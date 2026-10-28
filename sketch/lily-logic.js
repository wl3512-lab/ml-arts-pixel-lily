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
  const SUN_RISE = 0.8; // per second while the palm is open
  const SUN_FADE = 0.12; // per second with no palm, so it dims slowly
  const SUN_FADE_FIST = 0.5;
  // little lilies on short stalks down the stem: they sprout once the stem is
  // 6 px past them and open over SIDE_GROW_TIME seconds of open palm
  const SIDE_FLOWERS = [
    { at: 62, side: -1 },
    { at: 76, side: 1 },
    { at: 90, side: -1 },
  ];
  const SIDE_SPROUT = 6;
  const SIDE_GROW_TIME = 2.5;
  // once the main stem is tall, more lilies come up out of the grass, one
  // after another while the palm stays open. x is from the main stem, height
  // is how tall each one gets, size scales its flower
  const MEADOW = [
    { x: -30, height: 34, size: 0.7 },
    { x: 32, height: 40, size: 0.75 },
    { x: -46, height: 24, size: 0.6 },
    { x: 46, height: 28, size: 0.6 },
    { x: -15, height: 16, size: 0.5 },
    { x: 17, height: 20, size: 0.5 },
  ];
  const MEADOW_START = 60; // main stem height before the first one sprouts
  const MEADOW_GROW_TIME = 3;
  const MEADOW_NEXT = 0.4; // the next one starts when this one is 40% grown
  const OPEN_REACH = 1.5; // see fingerReach
  const FIST_REACH = 1.25;
  const PETAL_STAY = 10; // seconds a fallen petal lies in the grass...
  const PETAL_DISSOLVE = 4; // ...before it dissolves away over this long

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

  // how far the four fingertips reach from the wrist compared with their
  // knuckles (5/9/13/17 knuckles, 8/12/16/20 tips): about 1.8-2 with the palm
  // open, around 1 in a fist
  function fingerReach(keypoints) {
    if (!keypoints || keypoints.length < 21) return null;
    const wrist = keypoints[0];
    let sum = 0;
    for (const [knuckle, tip] of [[5, 8], [9, 12], [13, 16], [17, 20]]) {
      const k = keypoints[knuckle];
      const t = keypoints[tip];
      if (!wrist || !k || !t || ![wrist.x, wrist.y, k.x, k.y, t.x, t.y].every(Number.isFinite)) return null;
      const base = Math.hypot(k.x - wrist.x, k.y - wrist.y);
      if (base < 1e-6) return null;
      sum += Math.hypot(t.x - wrist.x, t.y - wrist.y) / base;
    }
    return sum / 4;
  }

  // the stem hand read from handPose keypoints, used until the CNN is trained.
  // Half-curled fingers read as none, so the plant just waits.
  function palmState(keypoints) {
    const reach = fingerReach(keypoints);
    if (reach === null) return "none";
    return reach > OPEN_REACH ? "open" : reach < FIST_REACH ? "fist" : "none";
  }

  // 0 while a fallen petal is fresh, rising to 1 once it has dissolved
  function petalDissolve(age) {
    return Math.min(1, Math.max(0, (age - PETAL_STAY) / PETAL_DISSOLVE));
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
      sun: 0, // 0 = pale and unlit, 1 = blazing
      sideGrowth: SIDE_FLOWERS.map(() => 0), // 0 = not sprouted, 1 = fully open
      meadow: MEADOW.map(() => 0), // same, for the lilies in the grass
    };
  }

  function leafCount(height) {
    return Math.max(0, Math.floor((height - 8) / LEAF_EVERY));
  }

  // leaf i sits LEAF_EVERY * (i + 1) up the stem and grows in over 10 pixels
  // the little lilies that have sprouted, for drawing
  function sideFlowers(plant) {
    return SIDE_FLOWERS.map((f, i) => ({ ...f, grow: plant.sideGrowth[i] })).filter((f) => f.grow > 0);
  }

  // the lilies in the grass that have sprouted, for drawing
  function meadowFlowers(plant) {
    return MEADOW.map((f, i) => ({ ...f, grow: plant.meadow[i] })).filter((f) => f.grow > 0);
  }

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
      plant.sun = Math.min(1, plant.sun + SUN_RISE * dt);
      SIDE_FLOWERS.forEach((f, i) => {
        if (plant.height >= f.at + SIDE_SPROUT) {
          plant.sideGrowth[i] = Math.min(1, plant.sideGrowth[i] + (healthy * dt) / SIDE_GROW_TIME);
        }
      });
      if (plant.height >= MEADOW_START) {
        MEADOW.forEach((f, i) => {
          if (i === 0 || plant.meadow[i - 1] >= MEADOW_NEXT) {
            plant.meadow[i] = Math.min(1, plant.meadow[i] + (healthy * dt) / MEADOW_GROW_TIME);
          }
        });
      }
    } else if (input.stem === "fist") {
      plant.wilt = Math.min(1, plant.wilt + WILT_SPEED * dt);
      plant.sun = Math.max(0, plant.sun - SUN_FADE_FIST * dt);
    } else {
      plant.sun = Math.max(0, plant.sun - SUN_FADE * dt);
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

  const api = {
    pinchRatio, pinchToOpen, fingerReach, palmState, petalDissolve, createPlant, step, leaves, leafCount, sideFlowers,
    meadowFlowers, STEM_MIN, STEM_MAX, SNAP_WINDOW, SIDE_FLOWERS, MEADOW, MEADOW_START, PETAL_STAY, PETAL_DISSOLVE,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.LilyLogic = api;
})(typeof window !== "undefined" ? window : globalThis);
