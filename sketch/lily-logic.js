// Pure plant logic for Pixel Lily: no p5, no ml5, so node can test it.
// Inputs each frame: how open the pinch is (0..1, or null when the flower hand
// is out of view) and what the CNN says the stem hand is doing.
(function (root) {
  const STEM_MIN = 22; // garden pixels
  const STEM_MAX = 104;
  const GROW_SPEED = 14; // pixels per second while the palm is open
  const WILT_SPEED = 0.55; // per second while the hand is a fist
  const REVIVE_SPEED = 0.9; // per second while the palm is open
  const LEAF_EVERY = 15; // a leaf every 15 pixels of stem
  const SNAP_WINDOW = 0.4; // seconds: a pinch that closes this fast plucks
  const SNAP_FROM = 0.55; // the pinch had to be at least this open...
  const SNAP_TO = 0.08; // ...and end up this closed
  const REGROW_TIME = 3; // seconds for a new bud

  // thumb tip to index tip, divided by wrist to middle knuckle, so the same
  // pinch reads the same near or far from the camera. ml5 handPose keypoints:
  // 0 wrist, 4 thumb tip, 8 index tip, 9 middle knuckle.
  function pinchRatio(keypoints) {
    if (!keypoints || keypoints.length < 21) return null;
    const pick = [keypoints[0], keypoints[4], keypoints[8], keypoints[9]];
    if (pick.some((k) => !k || !Number.isFinite(k.x) || !Number.isFinite(k.y))) return null;
    const [wrist, thumb, index, knuckle] = pick;
    const palm = Math.hypot(knuckle.x - wrist.x, knuckle.y - wrist.y);
    if (palm < 1e-6) return null;
    return Math.hypot(index.x - thumb.x, index.y - thumb.y) / palm;
  }

  // fingertips touching reads about 0.1-0.2, a wide pinch about 1.0-1.3
  function pinchToOpen(ratio) {
    if (ratio === null || !Number.isFinite(ratio)) return null;
    return Math.min(1, Math.max(0, (ratio - 0.2) / 0.8));
  }

  function createPlant() {
    return {
      time: 0,
      height: STEM_MIN,
      wilt: 0,
      bloom: 0, // smoothed, what gets drawn
      target: 0, // latest pinch
      bud: 1, // 1 = full-size flower head, grows back from 0 after a pluck
      pinchHistory: [],
      plucks: 0,
    };
  }

  function leafCount(height) {
    return Math.max(0, Math.floor((height - 8) / LEAF_EVERY));
  }

  // leaf i sits LEAF_EVERY * (i + 1) up the stem and grows in over 10 pixels
  function leaves(height) {
    const out = [];
    for (let i = 0; i < leafCount(height); i++) {
      const at = LEAF_EVERY * (i + 1) - 4;
      out.push({ at, side: i % 2 === 0 ? 1 : -1, size: Math.min(1, (height - at) / 10) });
    }
    return out;
  }

  // advance the plant by dt seconds. Returns { plucked } so the drawing code
  // can spawn falling petals.
  function step(plant, input, dt) {
    plant.time += dt;
    const events = { plucked: false };

    // stem hand, from the CNN
    if (input.stem === "open") {
      plant.wilt = Math.max(0, plant.wilt - REVIVE_SPEED * dt);
      const healthy = 1 - plant.wilt;
      plant.height = Math.min(STEM_MAX, plant.height + GROW_SPEED * healthy * dt);
    } else if (input.stem === "fist") {
      plant.wilt = Math.min(1, plant.wilt + WILT_SPEED * dt);
    }

    // flower hand, from handPose
    if (input.pinch !== null && input.pinch !== undefined) {
      plant.target = input.pinch;
      plant.pinchHistory.push({ t: plant.time, v: input.pinch });
    }
    plant.pinchHistory = plant.pinchHistory.filter((p) => plant.time - p.t <= SNAP_WINDOW);

    const recentMax = Math.max(0, ...plant.pinchHistory.map((p) => p.v));
    if (plant.bud >= 1 && plant.target <= SNAP_TO && recentMax >= SNAP_FROM && plant.bloom > 0.3) {
      events.plucked = true;
      plant.plucks += 1;
      plant.bud = 0;
      plant.bloom = 0;
      plant.pinchHistory = [];
    }

    if (plant.bud < 1) {
      plant.bud = Math.min(1, plant.bud + dt / REGROW_TIME);
      plant.bloom = 0; // a new bud stays shut until it is full size
    } else {
      plant.bloom += (plant.target - plant.bloom) * Math.min(1, dt * 6);
    }
    return events;
  }

  const api = { pinchRatio, pinchToOpen, createPlant, step, leaves, leafCount, STEM_MIN, STEM_MAX, SNAP_WINDOW };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LilyLogic = api;
})(typeof window !== "undefined" ? window : globalThis);
