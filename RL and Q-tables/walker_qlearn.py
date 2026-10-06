"""Teach a LEGO Double Motor "walker" (links instead of wheels) to go straight with a Q-table.

The hub's built-in IMU gives the yaw (heading) error - 0 deg is the direction
the walker was facing when the episode started. Each control step:

    1. Define things      - ALPHA, EPSILON, GAMMA and the reward (see constants / reward()).
    2. Read the state     - bin the current yaw error into one of N_STATES rows.
    3. Choose an action   - roll a random number: < epsilon -> random action (explore),
                            otherwise the column with the max Q in this row (exploit).
    4. Execute the action - apply that left/right motor speed differential for STEP_TIME.
    5. Observe the result - read the new yaw (s'), which gives the reward r.
    6. Update Q           - Q(s,a) += ALPHA * (r + GAMMA * max(Q[s']) - Q(s,a)).
    7. Decay epsilon      - epsilon *= EPSILON_DECAY.
    8. Repeat from 3 with s' as the new state.

Because a walker wobbles side to side every stride, each action is held for
STEP_TIME (roughly one full gait cycle) so the yaw reading reflects the net
heading change instead of mid-stride sway.

The Q-table is saved to Q_TABLE_PATH after every episode and reloaded on the
next run, so training carries over between sessions.

Run from the repo root:
    python "RL and Q-tables/walker_qlearn.py"          # train on the robot
    python "RL and Q-tables/walker_qlearn.py" --sim    # train against a simulated walker (no hardware)
    python "RL and Q-tables/walker_qlearn.py" --run    # no learning: just exploit the saved table
"""

import argparse
import random
import time
from pathlib import Path

import legoeducation as le
import numpy as np

# --- 1. Define things ------------------------------------------------------

ALPHA = 0.3            # learning rate: how far each update moves Q toward the new estimate
GAMMA = 0.9            # discount factor: how much future reward counts vs. immediate reward
EPSILON = 1.0          # exploration rate: start fully random...
EPSILON_DECAY = 0.98   # ...and multiply by this after every step
EPSILON_MIN = 0.02     # floor so it never stops exploring entirely
EPSILON_RESUME = 0.3   # starting epsilon when continuing from a saved Q-table

# Connection Card plugged into the walker's Double Motor
CARD_COLOR = le.LEGO_COLOR_ORANGE
CARD_SERIAL = "7572"

# Which face of the hub points UP on the walker - yaw is measured around that
# axis. If the motor is mounted on its side, the default face reports body
# rocking instead of heading. Check with --yaw-test. None = leave hub default.
YAW_FACE = None        # e.g. le.DEVICE_FACE_TOP / FRONT / RIGHT / BOTTOM / BACK / LEFT

BASE_SPEED = 60        # forward speed (%) both motors share
STEP_TIME = 0.6        # seconds each action is held (~one gait cycle) before reading yaw
YAW_SAMPLES = 6        # yaw readings averaged across each step, so stride sway cancels out
STEPS_PER_EPISODE = 40
EPISODES = 20
FAIL_YAW = 90          # |yaw| beyond this ends the episode early (walked off course)
FAIL_PENALTY = -5
ON_COURSE = 5          # |yaw| within this counts as on course (+1 reward)

# Flip if a positive differential turns the walker the wrong way. It doesn't
# matter for learning (the Q-table figures out which action helps), only for
# making the printed action labels read correctly.
LEFT_SIGN, RIGHT_SIGN = 1, 1

# --- 2. States: yaw error bins (deg). Positive yaw = drifted clockwise/right.
YAW_BIN_EDGES = [-45, -25, -12, -ON_COURSE, ON_COURSE, 12, 25, 45]
N_STATES = len(YAW_BIN_EDGES) + 1   # 9 states: <-45, -45..-25, ..., -5..5 (on course), ..., >45

# --- Actions: speed differential d -> left = BASE + d, right = BASE - d.
# d > 0 speeds up the left side, steering right; d < 0 steers left.
# Kept small enough that the slower leg (BASE - 20 = 40%) still walks.
ACTIONS = [-20, -10, -5, 0, 5, 10, 20]
N_ACTIONS = len(ACTIONS)

Q_TABLE_PATH = Path(__file__).parent / "q_table.npy"


def yaw_to_state(yaw):
    """Turn a yaw error (deg) into a state (row) index."""
    return int(np.digitize(yaw, YAW_BIN_EDGES))


def reward(yaw):
    """Reward for landing at this yaw error: 0 is best, more negative the farther off course."""
    if abs(yaw) > FAIL_YAW:
        return FAIL_PENALTY
    if abs(yaw) <= ON_COURSE:
        return 1.0                      # bonus for staying in the on-course bin
    return -abs(yaw) / 30.0


def choose_action(Q, state, epsilon):
    """3. Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)
    row = Q[state]
    best = np.flatnonzero(row == row.max())  # break ties randomly so an all-zero row isn't biased
    return int(random.choice(best))


def update_q(Q, s, a, r, s_next, done):
    """6. Q_new(s,a) = Q_old(s,a) + alpha * (r + gamma * max Q(s') - Q_old(s,a))."""
    q_hat = 0.0 if done else Q[s_next].max()
    Q[s, a] += ALPHA * (r + GAMMA * q_hat - Q[s, a])


def wrap180(angle):
    return (angle + 180) % 360 - 180


# --- Robots ----------------------------------------------------------------

class RealWalker:
    def __init__(self):
        from lelib import doubleMotor
        self.dm = doubleMotor()
        self.dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
        if YAW_FACE is not None:
            self.dm.imu_set_yaw_face(YAW_FACE)

    def reset(self):
        self.stop()
        input("\nPoint the walker straight ahead, then press Enter to start the episode...")
        self.dm.reset_heading()
        time.sleep(0.2)
        self._avg_yaw = None   # first state comes from a fresh raw reading

    def drive(self, diff):
        """4. Apply the speed differential, hold it for one gait cycle while
        sampling yaw, and keep the average (one sway spike can't end the episode)."""
        left = LEFT_SIGN * max(-100, min(100, BASE_SPEED + diff))
        right = RIGHT_SIGN * max(-100, min(100, BASE_SPEED - diff))
        self.dm.movement_move_tank(int(left), int(right), blocking=False)
        samples = []
        for _ in range(YAW_SAMPLES):
            time.sleep(STEP_TIME / YAW_SAMPLES)
            samples.append(wrap180(self.dm.yaw()))
        self._avg_yaw = sum(samples) / len(samples)

    def yaw(self):
        """Stride-averaged yaw from the last drive(), or a raw reading before the first one."""
        if self._avg_yaw is not None:
            return self._avg_yaw
        return wrap180(self.dm.yaw())

    def stop(self):
        self.dm.motor_stop()

    def close(self):
        try:
            self.stop()
            self.dm.disconnect()
        except Exception as exc:
            print(f"Error while stopping/disconnecting: {exc}")


class SimWalker:
    """Rough stand-in for testing the learning loop without hardware: the walker
    has a built-in drift (one leg linkage slightly stronger) plus stride noise."""
    DRIFT = 4.0        # deg per step it veers on its own
    TURN_GAIN = 0.4    # deg per step per % of differential
    NOISE = 1.5

    def reset(self):
        self._yaw = random.uniform(-3, 3)

    def drive(self, diff):
        self._yaw += self.DRIFT + self.TURN_GAIN * diff + random.gauss(0, self.NOISE)

    def yaw(self):
        return self._yaw

    def stop(self):
        pass

    def close(self):
        pass


# --- Training loop -----------------------------------------------------------

def print_q_table(Q):
    header = "state (yaw deg)".ljust(18) + "".join(f"{d:>+8d}" for d in ACTIONS) + "    best"
    print(header)
    print("-" * len(header))
    edges = [-np.inf] + YAW_BIN_EDGES + [np.inf]
    for s in range(N_STATES):
        label = f"{edges[s]:>5.0f} .. {edges[s + 1]:<5.0f}"
        cells = "".join(f"{q:8.2f}" for q in Q[s])
        best = ACTIONS[int(Q[s].argmax())] if Q[s].any() else "-"
        print(label.ljust(18) + cells + f"    {best}")


def yaw_test(walker):
    """No learning: walk straight (d = 0) and print yaw ~10x per second, so you can
    check the reading tracks heading (should change slowly) rather than stride rocking."""
    walker.reset()
    walker.dm.movement_move_tank(LEFT_SIGN * BASE_SPEED, RIGHT_SIGN * BASE_SPEED, blocking=False)
    for i in range(50):
        print(f"t={i * 0.1:4.1f}s  yaw={walker.yaw():+7.1f}")
        time.sleep(0.1)
    walker.stop()


def run(walker, Q, learn, epsilon):
    if not learn:
        epsilon = 0.0
    for episode in range(1, EPISODES + 1):
        walker.reset()
        state = yaw_to_state(walker.yaw())                   # 2. read the state
        total_reward = 0.0

        for step in range(STEPS_PER_EPISODE):
            action = choose_action(Q, state, epsilon)        # 3. choose an action
            walker.drive(ACTIONS[action])                    # 4. execute it
            yaw = walker.yaw()                               # 5. observe s' and r
            next_state = yaw_to_state(yaw)
            r = reward(yaw)
            done = abs(yaw) > FAIL_YAW
            total_reward += r

            if learn:
                update_q(Q, state, action, r, next_state, done)        # 6. update Q
                epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)    # 7. decay epsilon

            print(f"ep {episode:2d} step {step:2d}  s={state} a={ACTIONS[action]:+3d}  "
                  f"yaw={yaw:+6.1f}  r={r:+5.2f}  eps={epsilon:.3f}")
            state = next_state                               # 8. loop with s'
            if done:
                print("  walked off course - ending episode")
                break

        walker.stop()
        print(f"Episode {episode}: total reward {total_reward:+.1f}")
        if learn:
            np.save(Q_TABLE_PATH, Q)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sim", action="store_true", help="use a simulated walker instead of the robot")
    parser.add_argument("--run", action="store_true", help="exploit the saved Q-table without learning")
    parser.add_argument("--fresh", action="store_true", help="ignore any saved Q-table and start from zeros")
    parser.add_argument("--yaw-test", action="store_true", help="walk straight and print yaw (no learning)")
    args = parser.parse_args()

    if args.yaw_test:
        walker = RealWalker()
        try:
            yaw_test(walker)
        finally:
            walker.close()
        return

    if Q_TABLE_PATH.exists() and not args.fresh:
        Q = np.load(Q_TABLE_PATH)
        epsilon = EPSILON_RESUME
        print(f"Loaded Q-table from {Q_TABLE_PATH.name} (epsilon starts at {epsilon})")
    else:
        Q = np.zeros((N_STATES, N_ACTIONS))
        epsilon = EPSILON

    walker = SimWalker() if args.sim else RealWalker()
    try:
        run(walker, Q, learn=not args.run, epsilon=epsilon)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        walker.close()
        print()
        print_q_table(Q)


if __name__ == "__main__":
    main()
