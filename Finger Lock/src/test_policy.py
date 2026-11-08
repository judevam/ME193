"""Offline tests for the lock policy (no camera, no model)."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from lock_policy import LockPolicy

DT = 0.1
def run(p, seq, t):
    """seq: list of (label, conf, n_frames)"""
    for label, conf, n in seq:
        for _ in range(n):
            t += DT
            p.update(label, conf, t)
    return t

def enter(p, t, digits, noisy=False):
    p.arm(t); t += 1.2
    for d in digits:
        t = run(p, [(d, 0.95, 10), ("empty", 0.9, 6)], t)
    return t

def case(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond

ok = True
# 1 correct code unlocks
p = LockPolicy("135"); t = enter(p, 0, "135")
ok &= case("correct code unlocks", p.state == "UNLOCKED")
t = run(p, [("empty", 0.9, 120)], t)
ok &= case("re-locks after timeout", p.state == "LOCKED")
# 2 wrong code denied
p = LockPolicy("135"); t = enter(p, 0, "134")
ok &= case("wrong code denied", p.state == "DENIED")
# 3 three wrong -> lockdown, and even the right code is refused during it
p = LockPolicy("135"); t = 0
for _ in range(3):
    t = enter(p, t, "111"); t = run(p, [("empty", .9, 40)], t)
ok &= case("3 wrong codes -> lockdown", p.state == "LOCKDOWN")
p.arm(t)
ok &= case("cannot arm during lockdown", p.state == "LOCKDOWN")
# 4 a held pose does not count 3 times
p = LockPolicy("111"); p.arm(0); t = run(p, [("1", .95, 12 + 30)], 1.2)
ok &= case("one held pose = one digit", len(p.entered) == 1 and p.state == "ARMED")
# 5 flickering / low confidence never counts
p = LockPolicy("135"); p.arm(0)
t = run(p, [("1", .95, 5), ("2", .95, 1), ("1", .95, 5), ("1", .5, 20)], 1.2)
ok &= case("flicker and low confidence ignored", len(p.entered) == 0)
# 6 timeout
p = LockPolicy("135"); p.arm(0); t = run(p, [("empty", .9, 300)], 0)
ok &= case("entry timeout -> locked", p.state == "LOCKED")
# 7 no arming, no unlocking
p = LockPolicy("135"); t = run(p, [("1", .99, 30), ("empty", .9, 10), ("3", .99, 30)], 0)
ok &= case("nothing happens until armed", p.state == "LOCKED" and not p.entered)
# 8 code stored hashed
p = LockPolicy("135")
ok &= case("code not stored in plain text", "135" not in repr(vars(p)))
sys.exit(0 if ok else 1)
