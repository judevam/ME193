# Step Counter: plan

Extend the gait wearable so it **counts steps with a neural net (PyTorch)**, and **estimates
how far you walked**. Code: `StepCounter.py` (laptop, same folder and recordings as the
gait classifier). Same hardware (MPU-6050 + UNO Q), but for the step counter it goes on **one ankle** (outside,
just above the ankle bone), not the thigh. The thigh recordings in `data/` are for the gait
classifier only and are not used here.

## Assignment prompt

> Your goal here is to extend your walker to now count your steps using a neural net and
> PyTorch. Use every sensor you can to make it as accurate as possible. In the end I need to
> have three paragraphs explaining (in your and Claude's words) how the neural net is working.
> Also, I want to add the ability to estimate how far you walked, and give an overview of its
> limitations and accuracy (in the writeup).

## Deliverables

1. A neural net that counts steps from the wearable's IMU data (`StepCounter.py`).
2. A distance estimate from the step count.
3. Writeup: three paragraphs on how the net works (mine and Claude's words), plus
   limitations and accuracy of the step count and the distance.

## Decisions

| Question | Decision | Why |
|---|---|---|
| Placement | **One ankle** (user's choice). New recordings in `data_ankle/` with its own `steps.csv` | The ankle gives the sharpest heel-strike impacts. The thigh data has a different signal shape, so it can't train this net |
| Steps vs. events | The net marks **every foot landing**: one event = one step (`STEPS_PER_EVENT = 1`) | Measured on the first ankle recordings: the sensor foot makes a hard impact (3-4 g prominence) and the other foot a weaker one (about 1.5 g), alternating ~0.5 s apart, so all steps are visible. The first version assumed one event per stride and doubled it; that double-counted (e.g. 60 reported for 32 real steps) and was fixed 2026-10-31 |
| Sensors | All six IMU outputs plus three derived signals: accel magnitude, gyro magnitude, accel with gravity filtered out | "Every sensor you can." The MPU-6050 also has a temperature sensor, but it says nothing about steps. There is no magnetometer or barometer on this board |
| Net | Small 1-D convolutional net, 5 residual blocks, dilations 1-16, about 32k weights | Steps are repeating bumps in a time series; convolutions find bumps wherever they happen. Small, so it trains on a laptop CPU in a minute or so and could later run on the board |
| Output | Per sample: (a) probability that a step lands here, (b) none / walking / running | (a) gives the count: one peak = one step. (b) gives the gait for the distance step and lets the net learn to ignore standing and biking |
| Counting | Peaks of (a) above 0.35, at least 0.22 s apart | Turns the curve into a whole number |
| Training labels | You write the step count per recording in `data/steps.csv`; a peak detector finds candidate impacts and the N strongest become the targets | Marking every step by hand is too slow. The count is the one thing you can know for sure |
| Negatives | Standing and biking recordings: all-zero targets | The net has to learn that thigh motion that is not walking is not steps (air-biking is the hard one) |
| Evaluation | Leave-one-recording-out: count each recording with a net trained on the others. Baseline: the plain peak detector | Same reason as the gait classifier: windows from one recording look alike |
| Distance | steps x step length. Length per gait, optionally nudged by cadence, fitted from recordings where you measured the distance. Fallback: 0.415 x height (walking), 0.65 x height (running) | Simple, explainable, and you can measure the error directly |

## How the net works (for the writeup)

1. **Input.** 100 Hz recordings: for each time sample, nine numbers (the six IMU outputs and
   three derived ones), scaled to roughly -1..1.
2. **Convolutions.** A convolution slides a small learned pattern along the time axis and
   reports where the signal looks like it. Early layers learn short shapes (the sharp spike of
   a heel strike, the swing of the gyro). The dilated layers look farther out, so by the end
   each output sample has seen about 1.4 s around it, enough to tell a real step from a bump
   in the rhythm.
3. **Residual blocks.** Each block adds its result to its input, which makes the net easy to
   train even with several layers.
4. **Outputs.** One curve with a peak at every step, and a three-way gait guess per sample.
5. **Training.** Random 4 s crops of the recordings. The step curve is trained toward a small
   bump at each labeled step (binary cross-entropy, steps weighted 5x because they are rare);
   the gait guess uses cross-entropy. Augmentation: random gain (0.9-1.1x) and a little noise,
   so it doesn't memorize one stride strength. Adam with a one-cycle learning-rate schedule.
6. **Counting.** Run the whole recording through, find peaks, count them.

The three paragraphs: (1) what goes in and what comes out; (2) what the layers learn and why
the convolutions and dilations fit step data; (3) how it was trained and checked, and where it
fails. I write my version, Claude writes its version, and the writeup shows both.

## Phases

### Phase 0: setup (done)

- [x] `StepCounter.py` with `template`, `train`, `eval`, `count`
- [x] PyTorch (CPU) installed in `gaitVenv`, `torch` added to `requirements.txt`
- [x] Smoke test on the existing 14 gait recordings (runs end to end; no real step counts yet)

### Phase 1: record counted walks and runs

The existing recordings have no step counts, so they can't be scored. New protocol, using the
existing `gait.py record` (label walking or running):

1. Strap the sensor on the same ankle every time. Stand still, start recording.
2. Walk or run **exactly N steps**, counting out loud, every foot landing, both feet (any N
   works). Pick N = 50-100.
3. Stand still, stop recording.
4. If the distance is known (a hallway, a track lap, a measured 100 m), note it.

Collect at least:
- 8 walks: normal pace, slow, fast; one with a turn; some with distance measured
- 5 runs: easy and hard; some with distance measured
- 3 hard negatives: air-biking, fidgeting while standing, getting up from a chair
- Different paces matter more than many recordings at the same pace

Then, **into a separate folder so the thigh data stays out**:

```
python pull_data.py --out data_ankle --clear-board
python StepCounter.py --data data_ankle template     # then fill in data_ankle/steps.csv
python StepCounter.py --data data_ankle train
```

(columns `file, steps, distance_m`; `steps` is the total, both feet.)

### Phase 2: train and score

- [ ] `python StepCounter.py train`, then `eval`; read `models/step_scores.md`
- [ ] Compare against the peak-detector baseline. If the net isn't better, look at where it
  misses (plot the step curve over the signal) before changing anything
- [ ] Things to try if the count is off: more recordings at the missed pace, wider or narrower
  target bump, the 0.35 threshold, a bigger net

### Phase 3: distance

- [ ] Fill in `distance_m` for the measured walks and runs; `train` fits the step lengths
- [ ] `eval` reports distance error leave-one-out
- [ ] Check that step length really changes with cadence (slope) or whether one length per gait
  is as good

### Phase 3b: live view on the phone (done)

`python StepCounter.py serve` listens to the board's IMU stream and serves a page at
`http://<laptop IP>:8000` showing steps, distance, gait and cadence, with buttons to reset the
count and to start/stop labeled recordings. Counts lag by about 1 s (the net needs to see
past a step). `--replay FILE.csv` plays a recording in real time for testing without the board.

### Phase 4 (stretch): run it on the board

Export the weights to JSON and run the forward pass with numpy in `python/main.py`, the way
the gait models already work, so the LED matrix can show the step count. The net is small
(about 32k weights, convolutions only), so numpy is enough. Not needed for the writeup.

### Phase 5: writeup

- [ ] Three paragraphs on how the net works (mine and Claude's)
- [ ] Accuracy table from `step_scores.md`: net vs. peak detector, per recording and mean
- [ ] Distance accuracy, from the held-out recordings
- [ ] Limitations (below), backed by measurements where possible

## Expected limitations (check each against the data)

- **Labels are only as good as the count.** If I miscount by 2 steps in 100, the targets are
  off by 2%. The net can't be more accurate than my counting.
- **The targets are picked by a peak detector**, so the net partly learns to imitate it. The
  count constraint (exactly N) is what keeps it honest on totals; per-step timing is less sure.
- **The other foot is the weak signal.** The far foot's impact is only about 40% as strong
  as the sensor foot's (about 1.5 g vs. 3-4 g walking), so it is the one to go missing in
  shuffling, very light steps, or a limp that favors the sensor leg. Slow walks, where the
  other-foot impact is smallest, are the likeliest to under-count.
- **Ankle hardware limits.** Running heel strikes at the ankle can pass the board's +-8 g
  range, and fast leg swing can pass +-1000 deg/s; a clipped peak is still a peak, but check
  the raw data for flat tops. The enclosure is heavy for an ankle and can change the gait.
- **Air-biking looks like stepping at the ankle** (rhythmic, same cadence), so it needs
  recordings as negatives; expect it to be the net's hardest case.
- **Small dataset, one person, one sensor placement.** The score says how well it counts *my*
  walking with *this* mounting. A different person, strap position or shoes would need new
  data. Leave-one-recording-out tells how it handles a new session, not a new person.
- **Non-step motion.** Air-biking, stairs, hopping, and getting up and sitting down can look
  like steps. Stairs also have a different step length, which the distance doesn't know about.
- **Distance is steps x a typical length, not measured.** Errors: step length varies with
  speed, slope, fatigue and turning; it is fitted on a few recordings; and a count error of x%
  becomes at least an x% distance error. It can't tell a step forward from a step sideways or
  back. Expect distance to be less accurate than the count; the measured error is in the writeup.
- **No absolute position.** The MPU-6050 can't integrate acceleration into position (drift
  grows with time squared), which is why we don't try.
- **Walking and running only.** Biking gives zero steps and zero distance by design.

## Files

| File | What |
|---|---|
| `StepCounter.py` | features, the net, training, counting, distance, evaluation |
| `data_ankle/steps.csv` | your step counts and distances per recording (you fill this in) |
| `models/step_counter.pt` | trained weights and fitted step lengths (made by `train`) |
| `models/step_scores.md` | leave-one-out accuracy for the writeup (made by `eval`) |
