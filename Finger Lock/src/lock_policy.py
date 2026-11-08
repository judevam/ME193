r"""Lock policy: how the safe turns a stream of CNN predictions into a decision.

Pure Python (no torch/cv2) so the exact same file can run on the UNO Q.

Flow
  LOCKED --arm (button)--> ARMED --3 digits--> UNLOCKED (if code matches)
                                          \--> DENIED  (wrong code)
                                               3 wrong codes in a row -> LOCKDOWN (alarm)

Rules for accepting a digit (1-5):
  * the CNN must predict the SAME digit with confidence >= conf_min
    for hold_frames frames in a row (one noisy frame never counts)
  * after a digit is accepted the hand must leave ("0" fist or "empty") for
    clear_frames frames before the next digit can start, so one held pose
    cannot be counted three times
  * the whole entry must finish within entry_timeout seconds
  * the code is stored only as a salted SHA-256 hash
"""
import hashlib
import hmac
import os

DIGITS = {"1", "2", "3", "4", "5"}
NON_DIGITS = {"0", "empty"}  # fist / no hand = "nothing entered right now"


class LockPolicy:
    def __init__(self, code, code_length=3, conf_min=0.75, hold_frames=8,
                 clear_frames=4, settle_secs=1.0, entry_timeout=25.0,
                 unlock_secs=10.0, denied_secs=3.0, max_attempts=3,
                 lockdown_secs=20.0, on_unlock=None, on_lock=None, on_alarm=None):
        assert len(code) == code_length and set(code) <= DIGITS, "code must be 3 digits 1-5"
        self.code_length = code_length
        self._salt = os.urandom(8)
        self._hash = self._digest(code)
        self.conf_min, self.hold_frames, self.clear_frames = conf_min, hold_frames, clear_frames
        self.settle_secs, self.entry_timeout = settle_secs, entry_timeout
        self.unlock_secs, self.denied_secs = unlock_secs, denied_secs
        self.max_attempts, self.lockdown_secs = max_attempts, lockdown_secs
        self.on_unlock, self.on_lock, self.on_alarm = on_unlock, on_lock, on_alarm
        self.events = []
        self.attempts = 0
        self.state = "LOCKED"
        self._reset_entry()
        self.until = 0.0  # end time of a timed state

    # ---- helpers --------------------------------------------------------
    def _digest(self, digits):
        return hashlib.sha256(self._salt + "".join(digits).encode()).digest()

    def _log(self, now, msg):
        self.events.append({"t": now, "msg": msg})
        del self.events[:-30]

    def _reset_entry(self):
        self.entered = []
        self.candidate = None
        self.hold = 0
        self.need_clear = False
        self.clear_count = 0
        self.armed_at = 0.0

    # ---- buttons --------------------------------------------------------
    def arm(self, now):
        """The physical button (or the on-screen one)."""
        if self.state == "LOCKED":
            self._reset_entry()
            self.state = "ARMED"
            self.armed_at = now
            self._log(now, "ARMED: show 3 numbers, remove your hand between each")

    def lock(self, now):
        if self.state in ("ARMED", "UNLOCKED", "DENIED"):
            self._to_locked(now, "LOCKED by user")

    def _to_locked(self, now, msg):
        was_open = self.state == "UNLOCKED"
        self.state = "LOCKED"
        self._reset_entry()
        self._log(now, msg)
        if was_open and self.on_lock:
            self.on_lock()

    # ---- main update: call once per camera frame --------------------------
    def update(self, label, conf, now):
        # timed states
        if self.state in ("UNLOCKED", "DENIED", "LOCKDOWN"):
            if now >= self.until:
                self._to_locked(now, "re-locked" if self.state == "UNLOCKED" else "ready")
            return
        if self.state != "ARMED":
            return
        if now - self.armed_at > self.entry_timeout:
            self._to_locked(now, "TIMEOUT: entry took too long")
            return
        if now - self.armed_at < self.settle_secs:
            return

        if self.need_clear:  # waiting for the hand to leave
            if label in NON_DIGITS and conf >= 0.5:
                self.clear_count += 1
            else:
                self.clear_count = 0
            if self.clear_count >= self.clear_frames:
                self.need_clear = False
                self.hold = 0
                self.candidate = None
            return

        digit = label if (label in DIGITS and conf >= self.conf_min) else None
        if digit is not None and digit == self.candidate:
            self.hold += 1
        elif digit is not None:
            self.candidate, self.hold = digit, 1
        else:
            self.candidate, self.hold = None, 0

        if self.hold >= self.hold_frames:
            self.entered.append(self.candidate)
            self._log(now, f"digit {len(self.entered)} locked in")
            self.candidate, self.hold = None, 0
            self.need_clear, self.clear_count = True, 0
            if len(self.entered) == self.code_length:
                self._decide(now)

    def _decide(self, now):
        ok = hmac.compare_digest(self._digest(self.entered), self._hash)
        self.entered_final = len(self.entered)
        if ok:
            self.state = "UNLOCKED"
            self.until = now + self.unlock_secs
            self.attempts = 0
            self._log(now, "ACCESS GRANTED: vault open")
            if self.on_unlock:
                self.on_unlock()
        else:
            self.attempts += 1
            self._log(now, f"ACCESS DENIED ({self.attempts}/{self.max_attempts})")
            if self.attempts >= self.max_attempts:
                self.state = "LOCKDOWN"
                self.until = now + self.lockdown_secs
                self.attempts = 0
                self._log(now, "SILENT ALARM: lockdown")
                if self.on_alarm:
                    self.on_alarm()
            else:
                self.state = "DENIED"
                self.until = now + self.denied_secs

    # ---- for the UI -------------------------------------------------------
    def remaining(self, now):
        if self.state == "ARMED":
            return max(0.0, self.entry_timeout - (now - self.armed_at))
        if self.state in ("UNLOCKED", "DENIED", "LOCKDOWN"):
            return max(0.0, self.until - now)
        return 0.0
