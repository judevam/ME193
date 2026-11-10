# Finger Lock: ME193 Neural Network Midterm

A safe that opens only for a secret 3-number finger pattern (each 1-5), read by a webcam and classified by a PyTorch CNN trained from scratch. Trained on the laptop, then deployed (inference only) to the Arduino UNO Q.

## Goal / flow

1. Walk up to the safe, push a button, LEDs light up (system armed).
2. Place a finger count (1-5) under the camera, three times in a row.
3. CNN predicts each count; the 3-digit sequence is compared to the secret code.
4. Match = unlock (servo/latch). No match = stays locked (feedback via LEDs).

## Rubric mapping

| Level | Requirement | Where it is covered |
|---|---|---|
| 1 | Model counts fingers on the computer + Excel spreadsheet page | Phases 1-3 |
| 2 | Safe opens only for my 3-digit code, using the computer | Phase 4 |
| 3 | Works on the UNO Q | Phase 5 |
| 4 | Awesome: works for other people's hands, more than asked, great marketing/layout | Phases 6-7 |
| - | "I told Cath" | Submit / tell Cath at the end |

## Hard constraints from the prompt

- Use the **new external webcam** only (not phone or laptop camera), the same camera for training and for the UNO Q.
- **PyTorch, network built from scratch** (no pretrained models).
- **Train on the computer, not on the UNO Q.** Move the trained model over.
- Needs something fun/cool we have not done in class (see "Wow factor").
- Stop-action movie with my own voice-over explaining the neural net (CNN and pooling included).

## Phase 0: Setup

- [ ] New folder `Finger Lock/` (this one) with its own virtual environment (`.venv`), `requirements.txt`, `.gitignore` (ignore `.venv`, raw data, large checkpoints as needed).
- [ ] Install: torch, torchvision, opencv-python, numpy, matplotlib, pandas, openpyxl (for the spreadsheet).
- [ ] Confirm the webcam works in OpenCV and note its index/resolution.
- [ ] New git branch for this project.

## Phase 1: Notion warm-up section (required BEFORE the box)

Fit a model to the V-shaped data from Anna's code (started in class):

- [ ] Math write-up: gradient descent for a single neuron (y = mx + b), then a ReLU layer, with the chain rule for dL/dm and dL/db.
- [ ] Excel spreadsheet: 5 iterations of m and b (prediction, error, loss, gradients, update), with live formulas.
- [ ] Python script: plot of the data with the fit, plus loss vs. epoch.
- [ ] Add as a section of the Notion page.

## Phase 2: Camera rig and data collection

- [ ] Build a fixed camera stand at a set distance over the desk; draw a red square on the desk. Only the red square region is captured (small image, no zoom issues).
- [ ] Capture script: press a key to save a cropped, labeled image (classes: 0 empty and 1-5 fingers, or just 1-5, to decide).
- [ ] Collect data across: both hands, different lighting, different positions inside the square, and multiple people (for the "different hands" rubric item).
- [ ] Keep a held-out test set from a person/session not used for training.
- [ ] Watch for the hard classes (the prompt warns that 3 is the hardest).

## Phase 3: Neural net in PyTorch

- [ ] Preprocessing: crop, resize small (grayscale vs. color to decide), normalize.
- [ ] Augmentation: random rotation, shift, small brightness changes, to enlarge the dataset.
- [ ] Architecture (from the hint): 3 conv layers of different sizes, pool after each, concatenate the pooled features, linear layer down to the class outputs, **cross-entropy loss**.
- [ ] Training loop: Adam or SGD, train/val/test split, loss and accuracy curves per epoch.
- [ ] Evaluation: confusion matrix, per-class accuracy, failure-case images.
- [ ] Save the model (state dict, plus an export to ONNX/TorchScript if the UNO Q needs it).
- [ ] Live webcam demo showing the predicted count.
- [ ] Understand and be able to explain: width/depth, backprop, gradient descent, convolution, pooling.

## Phase 4: Lock policy on the computer

- [ ] State machine: LOCKED -> (button) ARMED -> read digit 1 -> digit 2 -> digit 3 -> compare -> UNLOCKED or DENIED -> back to LOCKED.
- [ ] Decision rules: confidence threshold, require N consistent frames before accepting a digit, a "hand removed" gap between digits, timeout, lockout after repeated failures.
- [ ] Secret code stored as a hash rather than plain text (nice detail).
- [ ] Simulated safe on the computer first (on-screen status), then real hardware: button, LEDs, servo/latch.
- [ ] Test: opens only for my code, stays shut for wrong codes and for other people's attempts.

## Phase 5: Move to the UNO Q

- [ ] Same webcam plugged into the UNO Q.
- [ ] Run the trained model for inference only (check the framework/size/speed on the board).
- [ ] Wire the button, LEDs, and servo/latch to the board.
- [ ] Compare against the computer: accuracy, latency/FPS, memory, setup difficulty (this answers one of the 4 required sections).

## Phase 6: Wow factor (fun/cool, not done in class)

Ideas to pick from (decide later):

- Show the **live feature maps** from each conv layer on a screen so people see what the net sees.
- A **decoy mode**: wrong code triggers a fun response (buzzer, LED pattern).
- Test with other people's hands and report accuracy per person.
- Camera-based **tamper/liveness check** (for example the code must be entered with a short pause between digits).
- Visualizations of the filters / Grad-CAM heatmaps over finger images.

## Phase 7: Deliverables

- [ ] **Notion page**: video, GitHub link, intro text, good explanations, marketing/layout polish.
- [ ] **GitHub repo**: clean README, code, requirements, results.
- [ ] **Stop-action movie** explaining the neural net (input, convolution, pooling, linear layer, loss, backprop, gradient descent), with my own voice-over.
- [ ] **Four required sections:**
  1. Lock policy: how it makes decisions.
  2. At least 4 limitations of the lock.
  3. How the neural net works (the stop-action movie).
  4. UNO Q vs. computer comparison.
- [ ] Usual template questions (what I was proud of, etc.).
- [ ] Demo video of the lock working (computer and UNO Q).
- [ ] Tell Cath.

## Candidate limitations (for the required section)

- Lighting changes or shadows cause misclassification.
- Fixed camera position and the red square are required; a bumped camera breaks accuracy.
- Only 5 digits over 3 positions: 125 combinations, easy to brute force or observe (shoulder-surfing).
- Trained on limited hands (skin tone, size, rings, nails), so generalization is uncertain.
- Hard classes (such as 3) are confused with others.
- Photo or printout of a hand could fool it (no liveness check).
- Physical lock hardware (servo/latch) is weak.

## Suggested folder layout

```
Finger Lock/
  PLAN.md
  README.md
  requirements.txt
  .venv/                (not committed)
  warmup/               (V-data fit, Excel, plots)
  data/                 (raw + split images, not committed if large)
  src/                  (capture, dataset, model, train, evaluate, lock logic)
  models/               (saved weights)
  results/              (plots, confusion matrix, demo media)
  unoq/                 (board app / deployment files)
  media/                (stop-action frames, voice-over, video)
```

## Decisions so far

- **Classes:** 6 total: 0 (no hand / empty) plus 1-5 fingers. Matches the 6-output hint.
- **Hardware:** servo (primary) with a LEGO single motor as a possible alternate latch; LEDs and buttons available.
- **Camera:** USB webcam into a USB port/hub on the UNO Q (same camera for training and the board).
- **Warm-up Excel sheet:** Jude builds it; Claude only helps when asked.
- **Wow factor:** live feature-map display, decoy/alarm mode, per-person accuracy test. Grad-CAM heatmaps are a stretch add-on if time allows.
- **No PyTorch on the board:** PyTorch stays on the computer only (training and export). The UNO Q gets the trained weights exported to a lightweight format (ONNX or plain NumPy) and runs inference without installing PyTorch. Exact route decided on Day 4.
- **Grad-CAM:** stretch goal, after everything else works.
- **Camera stand:** not built yet; build it on Day 1 before collecting data.
- **Branch:** `lock`, created from `main`.
- **Deadline:** 5 days from 2026-11-05 (due about 2026-11-10).

## 5-day schedule (tentative)

| Day | Focus |
|---|---|
| 1 (Nov 5) | Venv/setup, warm-up V fit (math, Excel, plots), camera stand + red square, start collecting data |
| 2 | Finish data collection (multiple people), augmentation, build and train CNN, confusion matrix |
| 3 | Lock state machine on the computer + servo/LEDs/button, live feature-map display, test code |
| 4 | Move model to UNO Q, wire hardware, compare with computer, decoy/alarm mode, per-person test |
| 5 | Stop-action movie + voice-over, Notion page, README, demo video, tell Cath |

Risk: the UNO Q port is the most uncertain step, so keep Day 4 flexible and have the computer version fully working first.

## Open questions

- UNO Q inference route (PyTorch directly vs. ONNX): decide after checking what runs on the board.
- Exact servo/latch mechanism and box build.
- Notion access (the Notion connector needs authorizing in claude.ai connector settings if I should write to it).
