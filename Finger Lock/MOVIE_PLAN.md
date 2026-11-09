# Stop-Action Movie: How the Finger-Lock Neural Network Works

**Format:** animated explainer, about 7.5 minutes, silent video with burned-in captions, plus a timed narration script (`MOVIE_SCRIPT.md`) for the voice-over recorded later.
**Level:** real math with real numbers from this project's actual network and images (no made-up examples).
**Style (default, change if you want):** 1280x720, 12 frames per second "stop-action" feel (animation steps in discrete frames rather than smooth motion), dark vault/heist look matching the vault page (neon green, amber, red accents), captions at the bottom.

## What the movie must show (from the assignment)
neuron, width and depth, convolution, pooling, backpropagation, gradient descent, plus the lock itself.

## Real assets used
- Real webcam crops from `data/raw` (one image per number), preprocessed to 96x96 grayscale.
- The trained weights: first-layer 5x5 filters, second and third layer filters, linear layer.
- Real feature maps for a sample image (same renderer as the vault page).
- Real training curves (training will be re-run once with the per-epoch history saved to JSON).
- Real Grad-CAM heatmaps (`results/gradcam.png` machinery).
- Real lock policy rules (`src/lock_policy.py`), and a clip of the vault page.

## Scene list

| # | Scene | Time | What's on screen | Math / numbers shown |
|---|---|---|---|---|
| 0 | **Hook** | 0:20 | Vault page, a hand shows 2, then 5, then 1, door opens. "How does a neural network read fingers?" | none |
| 1 | **A picture is just numbers** | 0:35 | Real hand photo zooms into pixels, then grayscale numbers 0-255, then 96x96 grid = 9,216 inputs | 96 x 96 = 9,216 numbers, scaled to 0-1 |
| 2 | **One neuron** | 0:50 | A neuron with weights and bias, the ReLU graph, link back to the V-data warm-up | y = ReLU(w1·x1 + w2·x2 + b) with real values, e.g. inputs (0.8, 0.2), w (1.5, -2), b 0.3 |
| 3 | **Width and depth** | 0:40 | Neurons stacked in a layer (width) and layers in a chain (depth); our net: 8, 16, 32 filters, 3 layers; 11,000 weights total | parameter count: 1x8x5x5+8 = 208, etc. |
| 4 | **Convolution** | 1:00 | A real 5x5 filter slides over the real image, one output pixel computed live, feature map builds up | sum of 25 products, one real output value |
| 5 | **ReLU and pooling** | 0:45 | Feature map, negative values removed, 2x2 max-pool shrinks 48x48 to 24x24 | max of 4 real numbers |
| 6 | **The whole network** | 0:50 | Three stages with pooling after each, features from all three concatenated (896 numbers), linear layer, 7 scores | 56 x 16 = 896 features, 7 x 896 weights; softmax of real scores into probabilities |
| 7 | **How wrong am I? Loss** | 0:40 | Probabilities bar chart vs the true answer; cross-entropy | loss = -ln(p_true), e.g. p=0.7 gives 0.36, p=0.05 gives 3.0 |
| 8 | **Backprop and gradient descent** | 1:10 | Error flowing backward through the layers; one weight with its gradient; the downhill-ball picture; real loss curve over epochs | dL/dw by the chain rule for the last layer; w_new = w - lr · dL/dw with real numbers |
| 9 | **Teaching it with few photos** | 0:35 | Dataset (7 classes, 1,450 images), rotate / shift / zoom / brightness augmentation examples, test results (85% same session, 65% new person) | counts, accuracy numbers |
| 10 | **What did it learn?** | 0:45 | Real feature maps for a hand; Grad-CAM heatmaps (fingertips matter, empty space matters for 0/1/2) | none |
| 11 | **From guess to lock decision** | 0:50 | State machine (locked, armed, 3 digits, open / denied / lockdown), "same digit 8 frames in a row at 75% confidence", hand must leave between digits | the rules as numbers |
| 12 | **Limits and wrap-up** | 0:30 | Where it fails (3, new hands, lighting), UNO Q running it with NumPy, thanks | 29 ms per image on the board |
| | **Total** | **about 8:50** | | |

That is about 8:50. To get closer to 7:30 I would shorten scenes 1, 3, 9 and 12 (each by 10-20 seconds) and tighten the hook. Tell me which scenes you want longer or cut.

## Build plan
1. Save training history to JSON (one retrain, about 7 minutes) so the loss curve and weights in scene 8 are real.
2. Write a small rendering toolkit (`movie/`): draw frames with Pillow/OpenCV, caption bar, theme, helper to hold or animate frames, and pipe frames to ffmpeg.
3. Build and preview scenes one at a time (a short MP4 per scene), so any scene can be fixed without rebuilding everything.
4. Concatenate scenes into `movie/finger_lock_explainer.mp4` (silent, captions).
5. Write `MOVIE_SCRIPT.md`: narration text per scene with the on-screen time, sized so about 150 words per minute fits the timing.
6. After recording, mux the voice-over onto the video with ffmpeg (I can do this when you have the audio file).

## Decisions I need from you
- Visual style: vault/heist theme (default) or a clean whiteboard look?
- Final length target: about 7:30 (trim) or fine at about 9 minutes?
- Do you want background music (quiet, royalty-free) or voice only?
- Is 1280x720 fine, or do you want 1920x1080 (slower to render, bigger file)?

---
**Update:** the movie was cut to a 5-minute version (4:56). Per-scene running times are set in `movie/build.py` (`NEW_DUR`); the narration in `MOVIE_SCRIPT.md` was rewritten to match.
