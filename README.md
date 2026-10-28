# Pixel Lily

Lucy Liu · Intro to Machine Learning for the Arts · Assignment 8 · October 28, 2026

A pixel-art stargazer lily grown with two hands in p5.js + ml5.js. The left hand pinches the flower open (wider pinch = bigger bloom, longer petals), and snapping the pinch shut plucks the petals. The right hand looks after the stem: an open palm grows it, adds leaves and little lilies, lights up the sun and, kept open, brings up six more lilies out of the grass. A fist wilts it all, and holding the fist until everything dies makes the next open palm grow a different flower: lily → tulip → daisy → sunflower → rose → morning glory, then round again. The stem hand is read by a **CNN trained in the browser** on 64×64 crops of the right half of the camera. The pinch is read by **handPose**, because a distance isn't a category.

- [Process post](https://wl3512-lab.github.io/ml-arts-pixel-lily/)
- [Run the p5.js sketch](https://wl3512-lab.github.io/ml-arts-pixel-lily/sketch/)
- [Class submission page](https://github.com/ml5js/Intro-ML-Arts-IMA-F26/wiki/Assignment-8)

## Try it

1. Open the sketch and click **Start camera**.
2. Right hand on the right side of the pink line. Hold **open** (or key 1) and move your open palm around, then **fist** (2), then **none** (3) with no hand there. About 30 each.
3. **Train** (T). The loss should end well under the faint coin-flip line at 1.1.
4. Palm grows the stem, fist wilts it. Left hand: pinch to bloom, snap shut to pluck.

No camera: **Use mouse + keys**. Drag across the garden to pinch (fast to the left to pluck), hold O to grow, F to wilt. **Save model** / **Load model…** keep a trained CNN. A saved model copied into `sketch/model/` loads by itself.

## How it works

| Part | Details |
| --- | --- |
| Stem hand | Built on the class example [Training a Webcam Image Classifier with Convolutional Layers](https://editor.p5js.org/jackbdu/sketches/7Y6VDvUO6). `video.get()` of the right half (p5's `flipped: true` mirrors the real pixels) → 64×64×4 → pixels centered to −0.5…0.5 → ml5 `imageClassification` (conv 8@5×5 → max pool → conv 16@5×5 → max pool → dense softmax, 12,139 weights) → open / fist / none. 64 epochs, batch size 8, SGD at ml5's default 0.02. Guesses under 60% confidence count as none. |
| Flower hand | `ml5.handPose({ maxHands: 2, flipped: true })`. The hand whose wrist is on the left half. Pinch = thumb tip to index tip ÷ wrist to middle knuckle, mapped 0.2–1.0 → 0–1. A snap (≥ 0.55 to ≤ 0.08 within 0.4 s) plucks; closing slowly just closes the flower. |
| Plant | `sketch/lily-logic.js`, with no p5 or ml5 in it, so node can test it: stem height, a leaf every 15 px, wilt, smoothed bloom, pluck and a 3 s regrow, the sun, 3 little lilies on the stem, a 6-lily meadow that sprouts one after another once the stem is 60 px, fallen petals that dissolve after 10 s, and the species cycle: a fist that fully wilts the garden marks it dead, and the next open palm restarts it as a sprout of the next of 6 kinds. Until the CNN is trained, handPose fingertip reach stands in for the stem hand. |
| Pixels | Drawn one pixel at a time into a 120×150 buffer, scaled up with no smoothing. Ordered (Bayer) dithering for the sky and soil. |

Versions are pinned: **p5.js 1.11.10** and **ml5.js 1.4.0**. The first two commits are the unchanged class example and the first fixes, so the history shows where it started.

## Verification

- `node --test tests/lily-logic.test.cjs`: 19 tests, all passing.
- `tests/browser/mouse-run.sh <url> [shots]`: real mouse and key events in headless Chrome through grow → bloom → slow close (no pluck) → snap (one pluck, 6 petals fall and land) → wilt → revive. Log: `tests/results/mouse-run-2026-10-28.txt`.
- `tests/browser/camera-learn-run.sh <url> <clip.y4m>`: Chrome's fake camera plays a clip from `make-shapes-camera.py` (big disc / small disc / nothing on the stem-hand side). 120 crops through the real camera path, trained: loss 0.81 → 0.000, live guesses matched the clip 29/30, and the crop brightness rises with disc size, which shows the crop is the correct half. Log: `tests/results/camera-learn-run-2026-10-28.txt`.
- `tests/browser/camera-run.sh <url> [shots]`: plumbing with Chrome's default test pattern (handPose loads, hold-to-record, train, classification keeps running).
- **Not done yet: a live test with Lucy's own hands.**

Why the class example only ever said "a", and the settings tests behind centering and batch size 8, are written up in the process post.

## Sources and assistance

- [Week 8 materials](https://github.com/ml5js/Intro-ML-Arts-IMA-F26/tree/main/08-cnn); starting point by jackbdu (link above)
- Reading: [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) by Ujjwal Karn
- [ml5.neuralNetwork](https://docs.ml5js.org/#/reference/neural-network), [ml5.handPose](https://docs.ml5js.org/#/reference/handpose)
- Silkscreen font by Jason Kottke (Google Fonts)

**AI assistance:** Claude (Anthropic), October 28, 2026, diagnosed the one-label problem, ran the training experiments, and wrote the sketch, plant logic, tests, browser checks, the meadow, the six kinds of flower and documentation. The sun, little lilies, handPose stand-in and dissolving petals were added later the same day, also AI-assisted. The lily, the two-hand controls and the pixel style are Lucy's idea. No live hand test or completion of the assigned reading is claimed.
