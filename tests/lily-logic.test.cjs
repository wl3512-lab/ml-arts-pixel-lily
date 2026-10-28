// node --test tests/lily-logic.test.cjs
const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../sketch/lily-logic.js");

// a fake handPose hand: wrist at (x, y), palm `size` pixels tall, thumb and
// index tips `gap` palm-lengths apart
function hand(x, y, size, gap) {
  const k = Array.from({ length: 21 }, () => ({ x, y }));
  k[0] = { x, y };
  k[9] = { x, y: y - size };
  k[4] = { x: x - (gap * size) / 2, y: y - size * 1.4 };
  k[8] = { x: x + (gap * size) / 2, y: y - size * 1.4 };
  return k;
}

function run(plant, input, seconds, fps = 30) {
  let plucked = 0;
  for (let i = 0; i < Math.round(seconds * fps); i++) if (L.step(plant, input, 1 / fps).plucked) plucked++;
  return plucked;
}

test("pinch ratio ignores where the hand is and how close it is", () => {
  const a = L.pinchRatio(hand(100, 300, 80, 0.6));
  const b = L.pinchRatio(hand(500, 120, 200, 0.6));
  assert.ok(Math.abs(a - 0.6) < 1e-9);
  assert.ok(Math.abs(a - b) < 1e-9);
});

test("broken hands give null instead of a number", () => {
  const k = hand(100, 300, 80, 0.6);
  k[8] = { x: NaN, y: 10 };
  assert.equal(L.pinchRatio(k), null);
  assert.equal(L.pinchRatio(hand(100, 300, 0, 0.6)), null);
  assert.equal(L.pinchRatio([]), null);
  assert.equal(L.pinchToOpen(null), null);
});

test("pinch maps to 0..1 and clamps", () => {
  assert.equal(L.pinchToOpen(0.05), 0);
  assert.ok(Math.abs(L.pinchToOpen(0.6) - 0.5) < 1e-9);
  assert.equal(L.pinchToOpen(3), 1);
});

test("opening the pinch blooms the flower", () => {
  const p = L.createPlant();
  run(p, { pinch: 0.9, stem: "none" }, 1);
  assert.ok(p.bloom > 0.85, `bloom ${p.bloom}`);
});

test("closing the pinch slowly closes the flower without plucking", () => {
  const p = L.createPlant();
  run(p, { pinch: 0.9, stem: "none" }, 1);
  let plucked = 0;
  for (let v = 0.9; v >= 0; v -= 0.02) plucked += run(p, { pinch: v, stem: "none" }, 0.05);
  plucked += run(p, { pinch: 0, stem: "none" }, 1);
  assert.equal(plucked, 0);
  assert.ok(p.bloom < 0.05);
});

test("snapping the pinch shut plucks the petals once, then a bud regrows", () => {
  const p = L.createPlant();
  run(p, { pinch: 0.9, stem: "none" }, 1);
  const plucked = run(p, { pinch: 0, stem: "none" }, 0.2);
  assert.equal(plucked, 1);
  assert.equal(p.bud < 1, true);
  run(p, { pinch: 0.9, stem: "none" }, 1); // pinching open mid-regrow does not bloom it
  assert.equal(p.bloom, 0);
  run(p, { pinch: 0.9, stem: "none" }, 2.5);
  assert.equal(p.bud, 1);
  assert.ok(p.bloom > 0.5);
});

test("a snap from a closed flower does nothing", () => {
  const p = L.createPlant();
  assert.equal(run(p, { pinch: 0, stem: "none" }, 0.5), 0);
});

test("open palm grows the stem up to a limit and adds leaves", () => {
  const p = L.createPlant();
  const before = L.leafCount(p.height);
  run(p, { pinch: null, stem: "open" }, 3);
  assert.ok(p.height > L.STEM_MIN + 30);
  assert.ok(L.leafCount(p.height) > before);
  run(p, { pinch: null, stem: "open" }, 30);
  assert.equal(p.height, L.STEM_MAX);
  for (const leaf of L.leaves(p.height)) assert.ok(leaf.size > 0 && leaf.size <= 1);
});

test("a fist wilts the plant and an open palm brings it back", () => {
  const p = L.createPlant();
  run(p, { pinch: null, stem: "fist" }, 1);
  assert.ok(p.wilt > 0.5);
  const h = p.height;
  run(p, { pinch: null, stem: "fist" }, 3);
  assert.equal(p.wilt, 1);
  assert.equal(p.height, h); // wilting doesn't shrink it
  run(p, { pinch: null, stem: "open" }, 2);
  assert.equal(p.wilt, 0);
});

test("no hands means nothing changes", () => {
  const p = L.createPlant();
  run(p, { pinch: 0.8, stem: "open" }, 1);
  const snap = { h: p.height, w: p.wilt, b: p.bloom };
  run(p, { pinch: null, stem: "none" }, 2);
  assert.equal(p.height, snap.h);
  assert.equal(p.wilt, snap.w);
  assert.ok(Math.abs(p.bloom - snap.b) < 0.05);
});
