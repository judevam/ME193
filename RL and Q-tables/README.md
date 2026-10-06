# RL and Q-tables

Tabular Q-learning that teaches a LEGO Double Motor **walker** (leg linkages
instead of wheels) to walk in a straight line, using the hub's built-in IMU yaw
as the only sensor.

- [walker_qlearn.py](walker_qlearn.py) - the Q-learning loop.
- [lelib.py](lelib.py) - shared LEGO Education wrapper (copy of `aprilTags/lelib.py`),
  used here for `reset_heading()` / `yaw()`.

## The table

|                | action = speed differential `d` (left = BASE+d, right = BASE-d) |
|----------------|------------------------------------------------------------------|
| **state** = yaw error bin | `<-45, -45..-25, -25..-12, -12..-5, -5..5, 5..12, 12..25, 25..45, >45` deg (yaw averaged over each step) |
| **actions**    | `-20, -10, -5, 0, +5, +10, +20` % around BASE_SPEED = 60                               |
| **reward**     | `+1` within ±5°, else `-abs(yaw)/30`; `-5` and episode ends past ±90° |

## The loop (one step)

1. **Define things** - `ALPHA=0.3`, `GAMMA=0.9`, `EPSILON=1.0`, reward above.
2. **Read the state** - bin the current yaw into a row.
3. **Choose an action** - random number < ε → random action, else the row's max-Q action.
4. **Execute** - apply the differential for `STEP_TIME` (~one gait cycle, so stride sway averages out).
5. **Observe** - read the new yaw → `s'` and `r`.
6. **Update Q** - `Q(s,a) += α (r + γ·max Q(s') − Q(s,a))`
7. **Decay ε** - `ε *= 0.98` (floored at `EPSILON_MIN`).
8. Repeat from 3 with `s'`.

## Running

```powershell
python "RL and Q-tables/walker_qlearn.py" --sim     # try it with a simulated walker first
python "RL and Q-tables/walker_qlearn.py"           # train on the robot
python "RL and Q-tables/walker_qlearn.py" --run     # exploit the learned table, no learning
python "RL and Q-tables/walker_qlearn.py" --fresh   # start over from an all-zero table
```

On the robot, each episode waits for you to point the walker straight and press
Enter (that zeroes the heading). The table is saved to `q_table.npy` after each
episode and reloaded next run, and it's printed at the end.

Tune `BASE_SPEED` and `STEP_TIME` to your linkage's gait first. If a positive
differential turns the walker the wrong way, flip `LEFT_SIGN`/`RIGHT_SIGN`.
The learning still works either way; the flip only fixes the labels.
