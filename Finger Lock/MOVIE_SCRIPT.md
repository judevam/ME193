# Narration script: How the Finger-Lock Neural Network Works

Read each block while its scene plays. The time next to the scene is when it starts in the video (`movie/out/finger_lock_explainer_silent.mp4`). The word budget is about 140 words a minute; if a scene's count is over budget, speak a little faster or cut a sentence.

This is the 5-minute cut (4:56). The numbers in the narration match what is on screen (real values from the trained network). If you retrain, the on-screen values change and these lines need updating.

## Scene 0: Hook  (0:00 - 0:08, 8s)

*17 words, budget 18*

How does a vault read your fingers? Show three numbers and the door opens. Here is how.

## Scene 1: A picture is just numbers  (0:08 - 0:23, 15s)

*31 words, budget 34*

To a computer, a photo is a grid of pixels. I shrink mine to ninety-six by ninety-six, gray, so nine thousand numbers between zero and one are all the network gets.

## Scene 2: One neuron  (0:23 - 0:50, 27s)

*54 words, budget 62*

A neuron is a tiny calculator: multiply each input by a weight, add them up, add a bias. Then ReLU: positive totals pass, negative totals become zero.

Two neurons, ReLU of x and ReLU of minus x, add up to a V, like our warm-up data. Learning means adjusting weights until the shape fits.

## Scene 3: Width and depth  (0:50 - 1:08, 18s)

*36 words, budget 41*

Width is how many neurons sit side by side. Depth is how many layers are chained. Ours has three convolution layers of eight, sixteen and thirty-two filters, then a linear layer: about twelve thousand learned numbers.

## Scene 4: Convolution  (1:08 - 1:43, 35s)

*79 words, budget 80*

The key idea is convolution. A filter is a small grid of weights, here five by five, and the network learns them.

Multiply each pixel under the filter by its weight and add up the twenty-five products: one point eight two, plus the bias, gives one point nine nine. That is one pixel of a new image.

Slide the filter everywhere and you get a feature map that lights up where the picture matches the pattern, here finger edges.

## Scene 5: ReLU and pooling  (1:43 - 2:07, 24s)

*40 words, budget 55*

ReLU turns every negative number into zero. Then max pooling: in each two by two block, keep only the biggest number. The map shrinks from ninety-six to forty-eight, the strongest signals survive, and small shifts of the hand matter less.

## Scene 6: The whole network  (2:07 - 2:35, 28s)

*48 words, budget 64*

Put it together: convolution and pooling three times, with eight, sixteen, then thirty-two filters. Shrink every map to four by four and line them up: eight hundred ninety-six numbers.

One linear layer turns them into seven scores, and softmax into probabilities. Here, about ninety-eight percent sure: a two.

## Scene 7: Loss  (2:35 - 2:55, 20s)

*42 words, budget 46*

To learn, the network needs a score for how wrong it is. Here the right answer is two, but it gave it only twenty-four percent. Cross-entropy loss is minus the log of that: one point four one. Training makes the average small.

## Scene 8: Backprop and gradient descent  (2:55 - 3:36, 41s)

*88 words, budget 94*

Backpropagation finds which weights caused the error. One feature, two point four two, one weight, zero point one one seven. At the score, the slope is probability minus one: minus zero point seven six. Times the feature: minus one point eight three.

Step against it: new weight equals weight minus learning rate times slope, here zero point three zero zero. Every weight does this; I used Adam, a smarter version.

The loss is a valley, and we walk downhill. In our real run, loss falls and accuracy climbs.

## Scene 9: Data and augmentation  (3:36 - 3:52, 16s)

*31 words, budget 36*

About sixteen hundred ninety photos of three people, with random rotations, shifts, zooms and flips. Ninety percent on new frames from familiar sessions, but only fifty on a new person's hand.

## Scene 10: What it learned  (3:52 - 4:14, 22s)

*44 words, budget 50*

Grad-CAM shows where the network looks. For three, four and five, the fingertips. For zero and one, mostly empty desk, where more fingers would be: no finger is evidence too. On a new hand it gets wrong, it often looks away from the fingers.

## Scene 11: The lock policy  (4:14 - 4:42, 28s)

*58 words, budget 64*

The network only guesses, frame by frame, so the lock policy decides. A digit counts only if the same number appears eight frames in a row at seventy-five percent confidence; a flicker never counts. Then the hand must leave.

Three right digits open the latch; a wrong code is denied, and three wrong codes trigger a twenty-second lockdown.

## Scene 12: Limits and wrap-up  (4:42 - 4:56, 14s)

*30 words, budget 32*

Limits: new hands, lighting, only one hundred twenty-five codes, a printed photo. On the UNO Q the same weights run in NumPy, about thirty milliseconds a photo. Thanks for watching.

---
Total video length: 4:56.

## After you record
Save your recording as one file (wav, mp3 or m4a), then run:

```
.venv\Scripts\python movie\add_voice.py my_voice.wav
```

Options: `--music 0.15` (background music volume under the voice), `--offset 1.0` (shift your voice later by 1 second). If your recording runs a little long or short, tell me and I can stretch or trim scene durations and re-render.