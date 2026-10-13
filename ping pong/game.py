"""The ping pong rules: ball, hit window, streak. Pure logic: no camera, BLE or MQTT.

Time is always passed in, so tests can fast-forward. The ball's distance is `z`:
0 = at the wall (far, drawn small), 1 = at you (drawn big).

    serve --(SERVE_DELAY)--> incoming --good swing in window--> returning --(travel)--> incoming ...
                                 |
                                 +--ball gets past you--> missed --(MISS_PAUSE)--> serve

Each incoming ball picks a lane (0 left, 1 center, 2 right). A hit needs a swing in the hit window
with the hand in that lane.

Only a ball getting past you ends the streak. A swing that isn't a hit (too early, wrong lane,
hand not seen) is a WHIFF: it's reported so the player gets feedback, but the ball keeps coming and
they can swing again. Whiffs are only reported in the second half of the ball's flight
(WHIFF_FROM_Z); swings while the ball is still far away are ignored.

On levels with a speed-up (config.LEVEL_SPEEDUP, hard), each hit in a row makes the ball a little
faster, down to config.MIN_TRAVEL_S; a miss resets it. The speed is set as each flight starts.

Some balls (config.LEVEL_SPIN_CHANCE) carry sidespin: they leave the wall aimed at the lane next
to `lane` (`from_lane`) and curve into `lane` late in the flight. Hits are judged on `lane`, where
the ball actually arrives; `aim_lane()` gives the in-between aim for drawing the curve.

Call `swing(t, hand_lane)` for each detected swing (with its own timestamp), then `update(now)`
once per frame. Both return a list of events: "serve", "hit", "miss", and the whiffs
"whiff_early", "whiff_lane", "whiff_no_hand". After a "miss", `miss_reason` says why: "late"
(never swung in time) or the last whiff's reason ("early", "lane", "no_hand").
"""

import random

import config

SERVE, INCOMING, RETURNING, MISSED = "serve", "incoming", "returning", "missed"
LANES = (0, 1, 2)
ANY_LANE = "any"   # hand_lane when pose is off: skip the lane check


class Game:
    def __init__(self, level=config.DEFAULT_LEVEL, rng=None):
        self.base_travel = config.LEVEL_TRAVEL_S[level]
        self.speedup = config.LEVEL_SPEEDUP.get(level, 1.0)
        self.travel = self.base_travel   # seconds wall <-> you for the current flight
        self.level = level
        self.early = config.HIT_EARLY_S
        self.late = config.HIT_LATE_S
        self.streak = 0
        self.best = 0
        self.last_offset = None   # last judged swing: seconds after the ball arrived (- = early)
        self.miss_reason = None   # why the last ball got past: "late", "early", "lane", "no_hand"
        self._whiff = None        # the current ball's last whiff reason
        self.state = None
        self.lane = 1    # lane the current ball ARRIVES in (what a hit is judged against)
        self.from_lane = 1   # lane it looks headed for when it leaves the wall (differs with spin)
        self.spin_chance = config.LEVEL_SPIN_CHANCE.get(level, 0.0)
        self._rng = rng or random.Random()
        self._t0 = 0.0   # when the current state started

    def start(self, now):
        self.streak = 0
        self._enter(SERVE, now)
        return ["serve"]

    # ── Inputs ───────────────────────────────────────────────────────────────

    def swing(self, t, hand_lane=ANY_LANE):
        """A swing happened at time t with the hand in hand_lane (None = hand not seen).
        Only counts while the ball is coming at you."""
        if self.state != INCOMING:
            return []   # ball at the wall / going away / just missed: swinging does nothing
        arrival = self._t0 + self.travel
        if t < self._t0 + config.WHIFF_FROM_Z * self.travel:
            return []   # ball still far away: a practice swing, ignored
        if t > arrival + self.late:
            return []   # after the window: update() reports the miss
        self.last_offset = t - arrival
        if t < arrival - self.early:
            return self._whiffed("early")
        if hand_lane is None:
            return self._whiffed("no_hand")
        if hand_lane != ANY_LANE and hand_lane != self.lane:
            return self._whiffed("lane")
        self.streak += 1
        self.best = max(self.best, self.streak)
        self._enter(RETURNING, t)
        return ["hit"]

    def update(self, now):
        """Advance time. Returns the events that happened up to `now`."""
        if self.state == SERVE and now >= self._t0 + config.SERVE_DELAY_S:
            self._enter(INCOMING, self._t0 + config.SERVE_DELAY_S)
        elif self.state == INCOMING and now > self._t0 + self.travel + self.late:
            return self._miss(self._t0 + self.travel + self.late)
        elif self.state == RETURNING and now >= self._t0 + self.travel:
            # bounced off the wall: comes straight back
            self._enter(INCOMING, self._t0 + self.travel)
        elif self.state == MISSED and now >= self._t0 + config.MISS_PAUSE_S:
            self._enter(SERVE, now)
            return ["serve"]
        return []

    # ── For drawing ──────────────────────────────────────────────────────────

    def ball_z(self, now):
        """0 = at the wall, 1 = at you; a bit over 1 while it's passing you late."""
        dt = now - self._t0
        if self.state == INCOMING:
            return dt / self.travel
        if self.state == RETURNING:
            return max(0.0, 1.0 - dt / self.travel)
        if self.state == MISSED:
            return 1.0 + (self.late + dt) / self.travel   # keeps flying past you
        return 0.0

    def time_to_arrival(self, now):
        """Seconds until the incoming ball reaches you (negative once it's late); None otherwise."""
        if self.state != INCOMING:
            return None
        return self._t0 + self.travel - now

    def in_hit_window(self, now):
        if self.state != INCOMING:
            return False
        arrival = self._t0 + self.travel
        return arrival - self.early <= now <= arrival + self.late

    # ── Internals ────────────────────────────────────────────────────────────

    def travel_for(self, streak):
        """Flight time at this streak: the level's base time, sped up per hit, never below the floor."""
        if self.speedup >= 1.0:
            return self.base_travel
        return max(config.MIN_TRAVEL_S, self.base_travel * self.speedup ** streak)

    @property
    def spin(self):
        """Which way the incoming ball curves: -1 left, +1 right, 0 no spin."""
        return (self.lane > self.from_lane) - (self.lane < self.from_lane)

    def aim_lane(self, s):
        """The lane (fractional) the incoming ball is heading for at progress s (0 wall .. 1 you):
        from_lane at first, curving into lane by the time it arrives."""
        s = min(max(s, 0.0), 1.0)
        return self.from_lane + (self.lane - self.from_lane) * s ** config.SPIN_CURVE

    @property
    def speed(self):
        """How fast the current flight is compared with the level's base speed (1.0 = normal)."""
        return self.base_travel / self.travel

    def _enter(self, state, t):
        self.state = state
        self._t0 = t
        if state in (INCOMING, RETURNING):   # a new flight: its speed comes from the streak
            self.travel = self.travel_for(self.streak)
        if state == INCOMING:
            self.lane = self._rng.choice(LANES)
            self.from_lane = self.lane
            if self.spin_chance > 0 and self._rng.random() < self.spin_chance:
                # looks headed for a neighbouring lane, curves into this one
                self.from_lane = self._rng.choice([n for n in (self.lane - 1, self.lane + 1) if n in LANES])
            self._whiff = None

    def _whiffed(self, reason):
        self._whiff = reason
        return ["whiff_" + reason]

    def _miss(self, t):
        self.miss_reason = self._whiff or "late"
        self.streak = 0
        self._enter(MISSED, t)
        return ["miss"]
