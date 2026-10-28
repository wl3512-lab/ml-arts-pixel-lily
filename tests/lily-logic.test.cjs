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

// a fake handPose hand with the four fingers either spread out or curled in
function palm(x, y, size, open) {
  const k = Array.from({ length: 21 }, () => ({ x, y }));
  const reach = open ? 1.9 : 0.9; // tip distance from the wrist, in knuckle lengths
  [[5, 8, -0.3], [9, 12, -0.1], [13, 16, 0.1], [17, 20, 0.3]].forEach(([knuckle, tip, lean]) => {
    k[knuckle] = { x: x + lean * size, y: y - size };
    k[tip] = { x: x + lean * size * reach, y: y - size * reach };
  });
  return k;
}

test("handPose reads an open palm and a fist the same near or far", () => {
  assert.equal(L.palmState(palm(400, 300, 60, true)), "open");
  assert.equal(L.palmState(palm(500, 200, 180, true)), "open");
  assert.equal(L.palmState(palm(400, 300, 60, false)), "fist");
  assert.equal(L.palmState(palm(500, 200, 180, false)), "fist");
  assert.equal(L.palmState([]), "none");
  const broken = palm(400, 300, 60, true);
  broken[12] = { x: NaN, y: 0 };
  assert.equal(L.palmState(broken), "none");
});

test("an open palm lights the sun, which fades slowly without it", () => {
  const p = L.createPlant();
  assert.equal(p.sun, 0);
  run(p, { pinch: null, stem: "open" }, 1.5);
  assert.equal(p.sun, 1);
  run(p, { pinch: null, stem: "none" }, 2);
  assert.ok(p.sun > 0.6 && p.sun < 1, `sun ${p.sun}`);
  run(p, { pinch: null, stem: "fist" }, 2);
  assert.equal(p.sun, 0);
});

test("little lilies sprout as the stem passes them and open with the palm", () => {
  const p = L.createPlant();
  assert.equal(L.sideFlowers(p).length, 0);
  run(p, { pinch: null, stem: "open" }, 3.5); // stem about 71 px: only the lowest one
  assert.equal(L.sideFlowers(p).length, 1);
  run(p, { pinch: null, stem: "open" }, 12);
  assert.equal(L.sideFlowers(p).length, L.SIDE_FLOWERS.length);
  assert.ok(p.sideGrowth.every((g) => g === 1));
  const before = [...p.sideGrowth];
  run(p, { pinch: null, stem: "fist" }, 2); // a fist doesn't undo them, it only wilts them
  assert.deepEqual(p.sideGrowth, before);
});

test("fallen petals stay 10 s, then dissolve over 4 s", () => {
  assert.equal(L.petalDissolve(0), 0);
  assert.equal(L.petalDissolve(9.9), 0);
  assert.ok(Math.abs(L.petalDissolve(12) - 0.5) < 1e-9);
  assert.equal(L.petalDissolve(14), 1);
  assert.equal(L.petalDissolve(60), 1);
});

test("more lilies come up out of the grass once the stem is tall, one after another", () => {
  const p = L.createPlant();
  run(p, { pinch: null, stem: "open" }, 2); // stem about 50 px, under MEADOW_START
  assert.ok(p.height < L.MEADOW_START);
  assert.equal(L.meadowFlowers(p).length, 0);
  run(p, { pinch: null, stem: "open" }, 2);
  const early = L.meadowFlowers(p).length;
  assert.ok(early >= 1 && early < L.MEADOW.length, `early ${early}`);
  assert.ok(p.meadow[0] > p.meadow[early - 1] || early === 1); // staggered, the first is furthest along
  run(p, { pinch: null, stem: "open" }, 20);
  assert.equal(L.meadowFlowers(p).length, L.MEADOW.length);
  assert.ok(p.meadow.every((g) => g === 1));
});

test("a fist or no hand doesn't grow the meadow", () => {
  const p = L.createPlant();
  run(p, { pinch: null, stem: "open" }, 4.5);
  const before = [...p.meadow];
  run(p, { pinch: null, stem: "fist" }, 3);
  run(p, { pinch: null, stem: "none" }, 3);
  assert.deepEqual(p.meadow, before);
});

function stepCount(plant, input, seconds, fps = 30) {
  const n = { plucked: 0, reborn: 0 };
  for (let i = 0; i < Math.round(seconds * fps); i++) {
    const e = L.step(plant, input, 1 / fps);
    if (e.plucked) n.plucked++;
    if (e.reborn) n.reborn++;
  }
  return n;
}

test("a fist kills the garden, and the next open palm grows a new kind of flower", () => {
  const p = L.createPlant();
  stepCount(p, { pinch: 0.8, stem: "open" }, 15);
  assert.equal(L.SPECIES[p.species], "lily");
  assert.ok(L.meadowFlowers(p).length > 0);
  assert.equal(stepCount(p, { pinch: null, stem: "fist" }, 3).reborn, 0); // dying doesn't switch yet
  assert.equal(p.dead, true);
  assert.equal(stepCount(p, { pinch: null, stem: "none" }, 2).reborn, 0); // nor does no hand
  const first = stepCount(p, { pinch: null, stem: "open" }, 1 / 30);
  assert.equal(first.reborn, 1);
  assert.equal(L.SPECIES[p.species], "tulip");
  assert.equal(p.dead, false);
  assert.equal(p.wilt, 0);
  assert.ok(p.height < L.STEM_MIN + 1); // starts again as a sprout
  assert.ok(p.bud < 0.1);
  assert.ok(p.sideGrowth.every((g) => g === 0) && p.meadow.every((g) => g === 0));
  assert.equal(stepCount(p, { pinch: null, stem: "open" }, 5).reborn, 0); // once per death
  assert.equal(L.SPECIES[p.species], "tulip");
});

test("opening before it's fully dead revives the same flower", () => {
  const p = L.createPlant();
  stepCount(p, { pinch: null, stem: "open" }, 3);
  stepCount(p, { pinch: null, stem: "fist" }, 1);
  assert.ok(p.wilt > 0.3 && p.wilt < 1);
  assert.equal(stepCount(p, { pinch: null, stem: "open" }, 2).reborn, 0);
  assert.equal(p.species, 0);
  assert.equal(p.wilt, 0);
});

test("the six kinds come round in order, then start over", () => {
  const p = L.createPlant();
  const seen = [L.SPECIES[p.species]];
  for (let i = 0; i < 6; i++) {
    stepCount(p, { pinch: null, stem: "fist" }, 2.5);
    stepCount(p, { pinch: null, stem: "open" }, 0.5);
    seen.push(L.SPECIES[p.species]);
  }
  assert.deepEqual(seen, ["lily", "tulip", "daisy", "sunflower", "rose", "morning glory", "lily"]);
  assert.equal(p.deaths, 6);
});
