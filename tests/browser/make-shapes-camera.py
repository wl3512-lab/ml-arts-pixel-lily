"""Make a fake-camera clip (.y4m) for camera-learn-run.sh.

640x480, 15 fps, a 6 s loop of three 2 s phases. The shape sits in the LEFT
half of the raw frame, which is the RIGHT half (stem-hand side) once p5's
flipped capture mirrors it. If the sketch cropped the wrong half, the CNN would
only ever see noise and could not learn the phases.
  phase 0 "open": big bright disc     phase 1 "fist": small bright disc
  phase 2 "none": nothing             a bright corner square marks the phase
Usage: python3 make-shapes-camera.py out.y4m
"""
import sys
import numpy as np

W, H, FPS, PHASE_FRAMES = 640, 480, 15, 30
rng = np.random.default_rng(8)
yy, xx = np.mgrid[0:H, 0:W]

with open(sys.argv[1], "wb") as f:
    f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
    for i in range(PHASE_FRAMES * 3):
        phase = i // PHASE_FRAMES
        y = (60 + rng.integers(0, 25, size=(H, W))).astype(np.uint8)  # dim noisy room
        cx = 160 + rng.integers(-40, 41)
        cy = 240 + rng.integers(-60, 61)
        r = {0: 95, 1: 40, 2: 0}[phase]
        if r:
            y[(xx - cx) ** 2 + (yy - cy) ** 2 < r * r] = 215
        # phase marker in the raw top-right corner (mirrored: top-left, flower side)
        y[10:40, W - 40 - phase * 50 : W - 10 - phase * 50] = 250
        f.write(b"FRAME\n")
        f.write(y.tobytes())
        f.write(np.full((H // 2) * (W // 2) * 2, 128, np.uint8).tobytes())
