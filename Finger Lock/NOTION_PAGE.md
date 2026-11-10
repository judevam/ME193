# Vault 193: a safe that opens for three fingers

> **[TODO]** Paste the demo video here (the lock opening for the right code, staying shut for a wrong one).
> **GitHub:** [ME193, branch `lock`](https://github.com/judevam/ME193/tree/lock/Finger%20Lock) (training, lock policy, movie) and [ArduinoApps, branch `lock`](https://github.com/judevam/ArduinoApps/tree/lock/finger-lock-cam-test) (the app that runs on the UNO Q).

## Intro

I built a lock that you open with your hand. You walk up to the safe, push the button, and show a secret pattern of three different numbers (1 to 5) to a webcam, using the number of fingers you hold up. A neural network I built in PyTorch reads each number. If the three numbers match the code, a servo pulls the latch open. If not, the safe stays locked, and three wrong tries in a row set off a lockdown.

I trained the network on my laptop with a new USB webcam, then moved the trained weights to an Arduino UNO Q, which now runs the whole lock by itself: camera, neural network, decision logic, servo latch, and a live "vault" web page.

| | |
|---|---|
| Classes | 7: fist (0), 1, 2, 3, 4, 5 fingers, and an empty desk |
| Network | 3 convolution layers (8, 16, 32 filters), pooling after each, 1 linear layer; **12,407** learnable numbers |
| Data | 1,690 photos I took with the same webcam, 3 people |
| Accuracy | 90% on new frames from familiar sessions (222 of 248), **50% on a brand-new person's hand** (trained on the other two people) |
| On the UNO Q | NumPy only (no PyTorch), about 25 to 31 ms per photo, about 10 readings per second |

**Something fun that we didn't do in class:** a heist-themed live vault page, a live view of what each convolution layer sees, Grad-CAM heatmaps that show where the network looks, an alarm lockdown, and a status display on the UNO Q's built-in LED matrix.

---

## Part 1. Warm-up: fitting the V with gradient descent

*(Section requested before the box.)*

**The data.** Anna's V data: 120 points, x from 21 to 100, y from about -2 to 79. The points follow a V, roughly y = 2·|x − 60|, with a bottom near x = 60.

### The math

**Single neuron.** The model is a line, and the loss is the mean squared error over N points:

$$\hat y_i = m\,x_i + b \qquad\qquad L=\frac1N\sum_{i=1}^{N}(\hat y_i-y_i)^2$$

With error $e_i=\hat y_i-y_i$, the chain rule gives the gradients:

$$\frac{\partial L}{\partial m}=\frac2N\sum_i e_i\,x_i \qquad\qquad \frac{\partial L}{\partial b}=\frac2N\sum_i e_i$$

Gradient descent takes a small step against the gradient, with learning rate $\eta$:

$$m\leftarrow m-\eta\frac{\partial L}{\partial m}\qquad\qquad b\leftarrow b-\eta\frac{\partial L}{\partial b}$$

**ReLU neuron.** Put a ReLU after the linear step:

$$z_i=w\,x_i+b,\qquad a_i=\max(0,z_i),\qquad \hat y_i=a_i$$

ReLU's slope is 1 where z > 0 and 0 where z < 0, so it acts like a switch in the chain rule:

$$\frac{\partial L}{\partial w}=\frac2N\sum_i e_i\,\mathrm{ReLU}'(z_i)\,x_i\qquad\qquad \frac{\partial L}{\partial b}=\frac2N\sum_i e_i\,\mathrm{ReLU}'(z_i)$$

Points where z < 0 contribute no gradient. A neuron that is off for every point can't learn at all (a "dead" neuron).

**ReLU layer.** J ReLU units, then a weighted sum:

$$z_{ij}=w_j x_i+b_j,\quad a_{ij}=\max(0,z_{ij}),\quad \hat y_i=\sum_{j=1}^{J}v_j\,a_{ij}+c$$

$$\frac{\partial L}{\partial c}=\frac2N\sum_i e_i\qquad
\frac{\partial L}{\partial v_j}=\frac2N\sum_i e_i\,a_{ij}$$

$$\frac{\partial L}{\partial w_j}=\frac2N\sum_i e_i\,v_j\,\mathrm{ReLU}'(z_{ij})\,x_i\qquad
\frac{\partial L}{\partial b_j}=\frac2N\sum_i e_i\,v_j\,\mathrm{ReLU}'(z_{ij})$$

Each weight is updated the same way: $\theta\leftarrow\theta-\eta\,\partial L/\partial\theta$.

**Why two ReLUs make a V:**

$$\mathrm{ReLU}(x-x_0)+\mathrm{ReLU}(x_0-x)=|x-x_0|$$

### Spreadsheet: 5 iterations on m and b

**[TODO]** Embed the sheet (or a screenshot) and the downloaded Excel file.

I fit the single neuron to the data with x and y divided by 100 (so one learning rate works for both parameters), starting at m = 0, b = 0, learning rate 0.5:

| iteration | m | b | loss |
|---|---|---|---|
| 0 | 0.0000 | 0.0000 | 0.21592 |
| 1 | 0.2389 | 0.4060 | 0.07658 |
| 2 | 0.1364 | 0.2629 | 0.05697 |
| 3 | 0.1617 | 0.3243 | 0.05405 |
| 4 | 0.1399 | 0.3092 | 0.05345 |
| 5 | 0.1361 | 0.3223 | 0.05320 |

Why divide by 100? The gradient contains x, which is up to 100, so with raw numbers a learning rate of 0.1 makes the loss explode (it went from 2,159 to 9.4×10⁸ in one step). Scaling puts x and y near 0 to 1. The slope m is unchanged; b comes out 100 times smaller.

### Python: all three models, by hand

The script `neuralNets/fit_v_gd.py` writes out every gradient above (NumPy only) and trains each model by gradient descent. I checked the hand-written gradients against numerical ones (they agree to about 1e-6).

**[TODO]** Add `fit_v_data_fit.png` (data and fits) and `fit_v_loss.png` (loss against epoch).

| Model | Result | RMS error |
|---|---|---|
| Single neuron (a line) | m = -0.08, b = 45.4: a nearly flat line | 22.5 |
| ReLU neuron | follows one arm of the V, settles in a compromise | 34.2 |
| ReLU layer, 2 units | follows the V closely | **2.0** |

- **One neuron can't make a V.** A line can't bend, so the best it can do is a nearly flat line through the middle.
- **Two ReLUs can.** The ReLU layer's error (2.0) is about the noise in the data.
- **Loss against epoch:** the layer's loss stays flat for a few hundred epochs, then drops sharply around epoch 500 to 2,000, then levels off near the noise floor.
- **Local minima are real.** In 3 of 8 random starts the ReLU layer got stuck at a much higher loss, so the script tries several starts and keeps the best.

The finger-lock network uses exactly these ideas, with many more weights and convolution layers.

---

## Part 2. How the lock decides (the lock policy)

The neural network only guesses a number for each camera frame. A separate, simple piece of code, the lock policy, turns those guesses into a decision.

**States:** `LOCKED` → (button) → `ARMED` → three digits → `OPEN` if the code matches, or `DENIED`, and after 3 wrong codes in a row `LOCKDOWN`.

**Rules for accepting one digit (1 to 5):**
1. The network must predict the **same** digit for **8 frames in a row** (about 0.8 seconds).
2. Its confidence must be **at least 75%** on every one of those frames.
3. After the digit is accepted, the hand has to leave (the fist or the empty-desk class) for **4 frames** before the next digit can start. This is why I added the empty class: one held hand can't be counted three times.
4. All three digits must be entered within **25 seconds**, or the safe locks again.

**After three digits:**
- Correct code: the servo opens the latch for 10 seconds, then locks again.
- Wrong code: denied for 3 seconds, and the latch stays shut.
- Three wrong codes in a row: **lockdown**. Entry is disabled for 20 seconds, with an alarm banner on the page and a flashing X on the board's LED matrix.

**Safety details:** the code is never stored as text, only as a salted SHA-256 hash, and it is compared in constant time. The digits are not displayed, only dots.

Real predictions flicker (a frame of "1" in the middle of a run of "2"s). The 8-in-a-row rule exists so one noisy frame never counts. I tested the policy with 10 automated tests (correct code opens, wrong denied, lockdown works, a held pose counts once, flicker is ignored, and so on).

---

## Part 3. Limitations

1. **New hands are much harder.** On photos from familiar sessions the network gets 90%, but a person it has never seen only gets **50%**. Before I added Sophie to the training data, the earlier model scored just 41% on her. Her photos show why: she holds her fingers close together, her hand is larger in the frame, and it comes in at a different angle. Mine and Mia's are spread apart. The network learned "spread fingers at this position and size" more than "count the fingers". She showed the right number of fingers every time, but all 40 of her "2" photos were read as "3". Only 3 people are in the data, and the second and third have just 30 to 40 photos per number.
2. **The digit 3 is easily confused with 2.** In my tests, 12 of 40 test photos of 3 were read as 2, and 6 of the 40 photos of 2 were read as 1.
3. **It depends on a fixed camera and lighting.** The network was trained on the red square at one position, so a bumped camera, a shadow, or a different lamp changes the pixels it sees.
4. **Only 125 possible codes** (5 × 5 × 5), and the gestures are visible, so someone watching you could copy the code (shoulder surfing).
5. **No check that it's a real hand.** A printed photo of a hand held under the camera might fool it.
6. **The latch is only as strong as a micro servo.** It is a demonstration latch, not real security.

Ideas to improve: collect many more people and lighting conditions, use more digits or two-hand patterns, add a liveness check (the code must be entered with a short pause pattern), and a stronger latch.

---

## Part 4. How the neural network works

**[TODO]** Embed the stop-action movie here (`finger_lock_explainer_final.mp4`, the 5-minute cut with my voice-over). The steps below are what the movie explains, with real numbers from the trained network.

1. **A photo is a grid of numbers.** The webcam image is cropped to the red square, turned gray, and shrunk to 96 × 96 = 9,216 pixels, each divided by 255.
2. **A neuron** multiplies each input by a weight, adds them up with a bias, and applies ReLU. Learning means adjusting these weights.
3. **Width and depth:** width is how many neurons in a layer, depth is how many layers. Mine: three convolution layers of 8, 16 and 32 filters, then one linear layer.
4. **Convolution:** a 5 × 5 filter (25 learned weights) slides across the image, and at each position produces one number (the sum of 25 products plus a bias). The result is a feature map that lights up where the image matches the filter's pattern. The first layer's filters find edges; later ones combine them into finger shapes.
5. **ReLU and pooling:** ReLU removes negatives. 2 × 2 max-pooling keeps the biggest of each block, halving the map and making the network tolerant of small shifts.
6. **Putting it together:** every layer's maps are shrunk to 4 × 4 and lined up (8 × 16 + 16 × 16 + 32 × 16 = **896** numbers); one linear layer turns them into 7 scores, and softmax turns the scores into probabilities.
7. **Loss:** cross-entropy, $-\ln(p_\text{correct})$. Confident and right is nearly 0, confident and wrong is large.
8. **Backpropagation and gradient descent:** the chain rule passes the error backward through every layer to give each weight a gradient, and each weight takes a small step against it. In the movie I show one real weight update (a weight of 0.117 and a slope of -1.83). I trained with Adam, a smarter version of this step, for 150 epochs.
9. **Data and augmentation:** 1,690 photos of 3 people, with random rotation, shift, zoom, flip and brightness changes during training so the network sees new versions every epoch.
10. **What it learned (Grad-CAM):** when it is right about 3, 4 or 5, the network looks at the fingertips. For 0 and 1 it mostly looks at empty desk, because "no finger there" is evidence too. On a new hand that it gets wrong, it often looks at empty areas instead of the fingers, which fits the drop on new hands.

**[TODO]** Add `results/gradcam.png` and the feature-map screenshot from the vault page.

---

## Part 5. UNO Q vs computer

The weights are identical. On the laptop the network runs in PyTorch; on the UNO Q it runs in plain NumPy (I exported the weights and folded batch-norm into the filters). On 300 test photos the NumPy version gave the same answer as PyTorch on every one (probability difference about 2e-6).

| | Computer (laptop) | UNO Q |
|---|---|---|
| Software | PyTorch | NumPy only, no PyTorch |
| Time per photo | about 2 to 5 ms | about 25 to 31 ms (about 9× slower) |
| Readings per second | I capped it at 12 | I capped it at 10 (the board could go a bit higher) |
| Camera | same SK-C201 webcam | same SK-C201 webcam, through a USB hub |
| Extra jobs | none | also runs the servo latch, the LED matrix, and the web pages |
| Setup | pip install | export weights once, deploy the app, set it as the default app so it starts at power-up |

**Differences I ran into:** the board has a single USB-C port, so I use a powered hub for the camera; Linux renumbers the camera between boots, so the app searches for it; and the board has no Servo library for D2, so the sketch makes the servo pulses itself.

**[TODO]** Optionally run the same saved photos through both and compare accuracy and timing side by side.

---

## Part 6. The build

**[TODO]** Photos of the box, camera stand, servo latch, and wiring. Add the button and LEDs once wired.

- Camera: SK-C201 USB webcam on a fixed stand over a red square.
- Brain: Arduino UNO Q (the Linux side runs the network and the web pages; the microcontroller side drives the servo on D2 and the LED matrix).
- Latch: micro servo, calibrated to 70° (locked) and 15° (open).
- Vault page: live camera with target brackets, the detected number and a confidence bar for every class, three tumblers, a dial that turns as digits lock in, a security log, and a live feature-map panel.

---

## What I was proud of, what was hard, what I'd do next

**[TODO: write in your own words.]** Some things from the project you might want to mention, if they're true for you:
- the lock policy handling flickering predictions
- the loss of accuracy on new people, and what that taught you
- getting the same weights to run on the board without PyTorch
- the digit 3
