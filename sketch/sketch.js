// Pixel Lily · Lucy Liu · ML Arts Assignment 8
//
// Built on the class example "Training a Webcam Image Classifier with
// Convolutional Layers" by jackbdu (https://editor.p5js.org/jackbdu/sketches/7Y6VDvUO6).
// The stem hand (right half of the camera) is read by that CNN, trained here in
// the browser on 64x64 crops: open / fist / none. Until it is trained, handPose
// finger keypoints stand in so the plant grows straight away. The flower hand (left half)
// is read by ml5 handPose, because a pinch is a distance, not a category.
// Plant rules live in lily-logic.js.

const imageWidth = 64;
const imageHeight = 64;
const imageChannels = 4;
const pixelsNum = imageWidth * imageHeight;

const LABELS = ["open", "fist", "none"];
const MAX_PER_LABEL = 80;
const SAMPLE_EVERY_MS = 110;
const CLASSIFY_EVERY_MS = 100;
const MIN_CONFIDENCE = 0.6;

const nnOptions = {
  inputs: [imageWidth, imageHeight, imageChannels],
  task: "imageClassification",
  debug: false,
};
// batchSize 8 + centered pixels: with the example's 4 and 0..1 pixels, training
// often never left coin-flip (see the post)
const trainingOptions = { epochs: 64, batchSize: 8 };

const GW = 120; // garden size in real pixels, scaled up with no smoothing
const GH = 150;
const GROUND = GH - 16;
const BASE_X = GW / 2;

let video = null;
let handPose = null;
let hands = [];
let mode = null; // "camera" | "mouse"

let classifier = null;
let examples = [];
let generation = 0; // late callbacks from an old model are ignored
let training = false;
let trained = false;
let lossHistory = [];
let stemGuess = { label: "none", confidence: 0 };
let holding = null;
let lastSampleAt = 0;
let lastCrop = null;
let lastCropAt = 0;

const keysDown = new Set();
let plant = LilyLogic.createPlant();
let lastInput = { pinch: null, stem: "none" };
let pinchHand = null;
let stemHand = null;
let falling = [];
let fallen = [];
let lastHead = null; // where the main flower was drawn, so plucked petals start there

let garden, gctx, skyLayer;
let gardenRect = { x: 0, y: 0, s: 1 };

const $ = (id) => document.getElementById(id);

// ---------- setup ----------

function setup() {
  const c = createCanvas(windowWidth, windowHeight);
  c.parent(document.querySelector("main"));
  c.elt.setAttribute("aria-label", "Pixel-art lily whose bloom and stem follow your hands");
  pixelDensity(1);
  noSmooth();
  garden = createGraphics(GW, GH);
  garden.pixelDensity(1);
  gctx = garden.drawingContext;
  skyLayer = makeSky();
  ml5.setBackend("webgl");
  wireUI();
  updateReadout();
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function wireUI() {
  $("startCamera").addEventListener("click", startCamera);
  $("useMouse").addEventListener("click", useMouse);
  for (const b of document.querySelectorAll("#labels button")) {
    b.addEventListener("pointerdown", (e) => { e.preventDefault(); startHolding(b.dataset.label); });
    for (const ev of ["pointerup", "pointerleave", "pointercancel"]) b.addEventListener(ev, stopHolding);
  }
  $("train").addEventListener("click", train);
  $("clear").addEventListener("click", clearExamples);
  $("save").addEventListener("click", () => classifier && trained && classifier.save("model"));
  $("load").addEventListener("change", (e) => loadModel(e.target.files, "your files"));
}

// ---------- input modes ----------

async function startCamera() {
  if (mode === "camera") return;
  $("startCamera").disabled = true;
  setStatus("Asking for the camera and loading handPose…");
  try {
    if (!window.ml5) throw new Error("ml5 did not load. Check your internet connection and reload.");
    await new Promise((resolve, reject) => {
      video = createCapture(VIDEO, { flipped: true }, resolve);
      video.elt.addEventListener("error", reject);
      setTimeout(() => reject(new Error("No camera picture after 15 s. Is another app using it?")), 15000);
    });
    // handPose reads the element's width/height attributes; video.size() sets them
    video.size(640, 480);
    video.hide();
    handPose = ml5.handPose({ maxHands: 2, flipped: true });
    await handPose.ready;
    handPose.detectStart(video, (results) => { hands = results; });
    mode = "camera";
    if (trained) classifyLoop(generation); // a model loaded before the camera was on
    setStatus("Camera on. Left of the pink line = flower hand, right = stem hand.");
    setTrainingEnabled(true);
    loadBundledModel();
  } catch (error) {
    setStatus(`Camera/model error: ${error.message || error}`, true);
    console.error(error);
    $("startCamera").disabled = false;
  }
}

function useMouse() {
  if (mode === "camera") {
    handPose?.detectStop();
    video?.remove();
    video = null;
    hands = [];
    $("startCamera").disabled = false;
  }
  mode = "mouse";
  setTrainingEnabled(false);
  setStatus("Mouse mode: drag across the garden to pinch, hold O / F for the stem.");
}

function setStatus(text, isError = false) {
  $("inputStatus").textContent = text;
  $("inputStatus").classList.toggle("error", isError);
}

// what the plant gets this frame: { pinch: 0..1 or null, stem: open|fist|none }
function readInputs() {
  let pinch = null;
  let stem = "none";
  pinchHand = null;
  stemHand = null;
  if (mode === "camera" && video) {
    // the flower hand is whichever hand has its wrist on the left half
    pinchHand = hands.find((h) => h.keypoints[0].x < video.width / 2) || null;
    stemHand = hands.find((h) => h.keypoints[0].x >= video.width / 2) || null;
    if (pinchHand) pinch = LilyLogic.pinchToOpen(LilyLogic.pinchRatio(pinchHand.keypoints));
    if (trained) {
      if (stemGuess.confidence >= MIN_CONFIDENCE) stem = stemGuess.label;
    } else if (stemHand) {
      stem = LilyLogic.palmState(stemHand.keypoints);
    }
  } else if (mode === "mouse" && mouseIsPressed && overGarden()) {
    pinch = constrain((mouseX - gardenRect.x) / (GW * gardenRect.s), 0, 1);
  }
  // O / F work in both modes, handy for testing the plant without a trained model
  if (keysDown.has("o")) stem = "open";
  if (keysDown.has("f")) stem = "fist";
  return { pinch, stem };
}

function overGarden() {
  const { x, y, s } = gardenRect;
  return mouseX >= x && mouseX <= x + GW * s && mouseY >= y && mouseY <= y + GH * s;
}

// ---------- the CNN (stem hand) ----------

// from the class example, changed to crop the right half of the camera and
// center the pixels on 0
function getNormalizedPixels() {
  const img = video.get(video.width / 2, 0, video.width / 2, video.height);
  img.pixelDensity(1);
  img.resize(imageWidth, imageHeight);
  img.loadPixels();
  lastCrop = img;
  lastCropAt = millis();
  const normalizedPixels = [];
  for (let i = 0; i < pixelsNum * imageChannels; i++) {
    normalizedPixels.push(img.pixels[i] / 255 - 0.5);
  }
  return normalizedPixels;
}

function cameraReady() {
  return mode === "camera" && video && video.loadedmetadata;
}

function startHolding(label) {
  if (!cameraReady() || training) return;
  holding = label;
  lastSampleAt = 0;
  document.querySelector(`#labels [data-label="${label}"]`).classList.add("holding");
}

function stopHolding() {
  holding = null;
  for (const b of document.querySelectorAll("#labels button")) b.classList.remove("holding");
}

function addExample(label) {
  if (counts()[label] >= MAX_PER_LABEL) return;
  examples.push({ inputs: getNormalizedPixels(), label });
  updateCounts();
}

function counts() {
  const c = { open: 0, fist: 0, none: 0 };
  for (const e of examples) c[e.label]++;
  return c;
}

function updateCounts() {
  const c = counts();
  for (const l of LABELS) $(`count-${l}`).textContent = c[l];
}

function clearExamples() {
  examples = [];
  updateCounts();
}

function setTrainingEnabled(on) {
  for (const b of document.querySelectorAll("#labels button")) b.disabled = !on;
  $("train").disabled = !on;
  $("clear").disabled = !on;
}

function train() {
  if (training || mode !== "camera") return;
  const c = counts();
  const ready = LABELS.filter((l) => c[l] >= 5);
  if (ready.length < 2) {
    $("trainStatus").textContent = "Add at least 5 examples for two labels first (30+ each works much better).";
    return;
  }
  stopHolding();
  generation++;
  const gen = generation;
  training = true;
  trained = false;
  lossHistory = [];
  classifier = ml5.neuralNetwork(nnOptions);
  for (const e of examples) classifier.addData(e.inputs, [e.label]);
  $("trainStatus").textContent = `Training on ${examples.length} pictures…`;
  classifier.train(
    trainingOptions,
    (epoch, logs) => {
      if (gen !== generation) return;
      lossHistory.push(logs.loss);
      $("trainStatus").textContent = `epoch ${epoch + 1}/${trainingOptions.epochs} · loss ${logs.loss.toFixed(3)}`;
      drawLoss();
    },
    () => {
      if (gen !== generation) return;
      training = false;
      trained = true;
      const first = lossHistory[0];
      const last = lossHistory[lossHistory.length - 1];
      $("trainStatus").textContent = `Trained on ${examples.length} pictures · loss ${first.toFixed(2)} → ${last.toFixed(3)}`
        + (last > 0.6 ? "\nStill high: add more varied examples and train again." : "");
      $("save").disabled = false;
      classifyLoop(gen);
    }
  );
}

async function classifyLoop(gen) {
  if (gen !== generation || !trained || !cameraReady()) return;
  try {
    const results = await classifier.classify(getNormalizedPixels());
    if (gen !== generation) return;
    stemGuess = { label: results[0].label, confidence: results[0].confidence };
  } catch (error) {
    console.warn(error);
  }
  setTimeout(() => classifyLoop(gen), CLASSIFY_EVERY_MS);
}

async function loadModel(source, from) {
  generation++;
  const gen = generation;
  try {
    const nn = ml5.neuralNetwork(nnOptions);
    await nn.load(source);
    if (gen !== generation) return;
    classifier = nn;
    trained = true;
    training = false;
    lossHistory = [];
    drawLoss();
    $("trainStatus").textContent = `Loaded a trained model from ${from}. Train again any time to replace it.`;
    $("save").disabled = false;
    classifyLoop(gen);
  } catch (error) {
    $("trainStatus").textContent = `Could not load the model: ${error.message || error}`;
    console.error(error);
  }
}

// a model saved with "Save model" and copied into sketch/model/ loads by itself
async function loadBundledModel() {
  try {
    const res = await fetch("model/model.json", { method: "HEAD" });
    if (res.ok && !trained && !training) loadModel("model/model.json", "this site (trained on Lucy's hand)");
  } catch (error) {
    // no bundled model, train one instead
  }
}

function drawLoss() {
  const cv = $("lossChart");
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (lossHistory.length < 2) return;
  const max = Math.max(1.2, ...lossHistory);
  ctx.strokeStyle = "#ff6fae";
  ctx.lineWidth = 2;
  ctx.beginPath();
  lossHistory.forEach((v, i) => {
    const x = (i / (trainingOptions.epochs - 1)) * cv.width;
    const y = cv.height - 3 - (v / max) * (cv.height - 6);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  // coin-flip for 3 labels is about 1.1
  const y = cv.height - 3 - (1.1 / max) * (cv.height - 6);
  ctx.fillStyle = "#ffffff55";
  ctx.fillRect(0, y, cv.width, 1);
}

// ---------- keys ----------

function keyPressed() {
  const k = key.toLowerCase();
  keysDown.add(k);
  if (k === "1" || k === "2" || k === "3") startHolding(LABELS[Number(k) - 1]);
  else if (k === "t") train();
  else if (k === "d") document.body.classList.toggle("clean");
}

function keyReleased() {
  const k = key.toLowerCase();
  keysDown.delete(k);
  if (k === "1" || k === "2" || k === "3") stopHolding();
}

// ---------- main loop ----------

function draw() {
  const dt = Math.min(deltaTime / 1000, 0.05);
  if (holding && cameraReady() && millis() - lastSampleAt > SAMPLE_EVERY_MS) {
    lastSampleAt = millis();
    addExample(holding);
  }
  if (cameraReady() && !trained && millis() - lastCropAt > 250) getNormalizedPixels(); // keeps the preview fresh

  lastInput = readInputs();
  const events = LilyLogic.step(plant, lastInput, dt);
  if (events.plucked) dropPetals();
  updatePetals(dt);

  renderGarden();
  background("#1f1226");
  placeGarden();
  image(garden, gardenRect.x, gardenRect.y, GW * gardenRect.s, GH * gardenRect.s);
  if (cameraReady()) drawCamera();
  if (frameCount % 6 === 0) updateReadout();
}

function placeGarden() {
  const wide = width > 760;
  const left = wide ? 400 : 0; // the panel
  const right = wide && cameraReady() ? 288 : 0; // the camera column
  const bottom = wide ? 0 : height * 0.5;
  const room = width - left - right;
  const s = Math.max(1, Math.floor(Math.min((room - 20) / GW, (height - bottom - 20) / GH)));
  gardenRect = {
    s,
    x: Math.floor(left + (room - GW * s) / 2),
    y: Math.floor((height - bottom - GH * s) / 2),
  };
}

function drawCamera() {
  const w = 256;
  const h = 192;
  const x = width - w - 16;
  const y = height - h - 16;
  push();
  image(video, x, y, w, h);
  stroke("#ff6fae");
  strokeWeight(3);
  line(x + w / 2, y, x + w / 2, y + h);
  noStroke();
  textSize(11);
  textFont("monospace");
  fill(0, 150);
  rect(x, y, w, 18);
  fill(255);
  text("flower hand", x + 6, y + 13);
  text("stem hand", x + w / 2 + 6, y + 13);
  if (pinchHand) {
    const k = pinchHand.keypoints;
    const sx = w / video.width;
    stroke("#ffd6e8");
    strokeWeight(2);
    line(x + k[4].x * sx, y + k[4].y * sx, x + k[8].x * sx, y + k[8].y * sx);
    noStroke();
    fill("#ff6fae");
    circle(x + k[4].x * sx, y + k[4].y * sx, 8);
    circle(x + k[8].x * sx, y + k[8].y * sx, 8);
  }
  if (stemHand && !trained) {
    const k = stemHand.keypoints;
    const sx = w / video.width;
    fill("#ffd23f");
    for (const i of [8, 12, 16, 20]) circle(x + k[i].x * sx, y + k[i].y * sx, 7);
  }
  // what the CNN actually gets: 64x64, squished
  if (lastCrop) {
    const t = 128;
    const ty = y - t - 12;
    image(lastCrop, x + w - t, ty, t, t);
    fill(255);
    noStroke();
    textAlign(RIGHT);
    text("what the CNN sees (64×64)", x + w, ty - 6);
    if (trained) {
      textSize(14);
      text(`${stemGuess.label} ${stemGuess.confidence.toFixed(2)}`, x + w - t - 10, ty + t);
    }
  }
  pop();
}

function updateReadout() {
  const p = plant;
  const pinch = lastInput.pinch === null ? "—" : lastInput.pinch.toFixed(2);
  const cnn = mode === "camera"
    ? (trained ? `CNN ${stemGuess.label} ${stemGuess.confidence.toFixed(2)}` : "handPose (CNN not trained yet)")
    : "keys";
  $("readout").textContent =
    `pinch ${pinch} → bloom ${p.bloom.toFixed(2)}${p.bud < 1 ? ` (regrowing ${Math.round(p.bud * 100)}%)` : ""}\n` +
    `stem hand: ${cnn} → ${lastInput.stem}\n` +
    `stem ${Math.round(p.height)}/${LilyLogic.STEM_MAX} · leaves ${LilyLogic.leafCount(p.height)} · wilt ${p.wilt.toFixed(2)} · plucked ${p.plucks}×\n` +
    `sun ${p.sun.toFixed(2)} · little lilies ${LilyLogic.sideFlowers(p).length}/${LilyLogic.SIDE_FLOWERS.length}` +
    ` · meadow ${LilyLogic.meadowFlowers(p).length}/${LilyLogic.MEADOW.length}`;
}

// ---------- pixel art ----------

const PAL = {
  sky: ["#ffe6f3", "#fbdcf4", "#f3d8fd", "#e8dbff"],
  soil: "#6b3d5c",
  soilDark: "#4e2a45",
  grass: "#7bd389",
  grassDark: "#3f9a5a",
  stem: "#4caf6a",
  stemDark: "#2f7d4a",
  leaf: "#5cc47c",
  leafDark: "#2f8a52",
  dead: "#a08c4c",
  deadDark: "#6f6034",
  petalEdge: "#fff1f7",
  petal: "#ff8cc0",
  stripe: "#ff3d8b",
  spot: "#9e1050",
  dusty: "#b79aa9",
  dustyDark: "#8e7282",
  bud: "#9fdc8c",
  budEdge: "#5aa860",
  blush: "#ffbfdc",
  throat: "#d9f27a",
  filament: "#f0ffd8",
  anther: "#c2410c",
  sparkle: "#ffffff",
  sunDim: "#f1d3e6", // unlit: a pale disc that sits in the pink sky
  sunDimEdge: "#dcb3cf",
  sun: "#ffd23f",
  sunEdge: "#ff9f1c",
  sunCore: "#fff6b8",
  sunGlow: "#fff1a8",
};

const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5];

function makeSky() {
  const g = createGraphics(GW, GH);
  g.pixelDensity(1);
  const ctx = g.drawingContext;
  for (let y = 0; y < GROUND; y++) {
    for (let x = 0; x < GW; x++) {
      // ordered dithering between 4 sky colors, top to bottom
      const f = (y / GROUND) * (PAL.sky.length - 1);
      const i = Math.floor(f);
      const pick = f - i > (BAYER[(y % 4) * 4 + (x % 4)] + 0.5) / 16 ? i + 1 : i;
      ctx.fillStyle = PAL.sky[Math.min(pick, PAL.sky.length - 1)];
      ctx.fillRect(x, y, 1, 1);
    }
  }
  for (let y = GROUND; y < GH; y++) {
    for (let x = 0; x < GW; x++) {
      const deep = (y - GROUND) / (GH - GROUND);
      ctx.fillStyle = deep > (BAYER[(y % 4) * 4 + (x % 4)] + 0.5) / 16 ? PAL.soilDark : PAL.soil;
      ctx.fillRect(x, y, 1, 1);
    }
  }
  for (let x = 0; x < GW; x++) {
    ctx.fillStyle = PAL.grass;
    ctx.fillRect(x, GROUND, 1, 2);
    if (hash(x, 1, 7) < 0.35) {
      ctx.fillStyle = hash(x, 2, 7) < 0.5 ? PAL.grass : PAL.grassDark;
      ctx.fillRect(x, GROUND - 1 - Math.floor(hash(x, 3, 7) * 3), 1, 2);
    }
  }
  return g;
}

function hash(a, b, c) {
  let h = Math.imul(a | 0, 374761393) ^ Math.imul(b | 0, 668265263) ^ Math.imul(c | 0, 2147483587);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}

function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16);
  const pb = parseInt(b.slice(1), 16);
  const ch = (s) => {
    const x = (pa >> s) & 255;
    const y = (pb >> s) & 255;
    return Math.round(x + (y - x) * t);
  };
  return "#" + ((ch(16) << 16) | (ch(8) << 8) | ch(0)).toString(16).padStart(6, "0");
}

function px(x, y, c) {
  gctx.fillStyle = c;
  gctx.fillRect(Math.round(x), Math.round(y), 1, 1);
}

function renderGarden() {
  gctx.drawImage(skyLayer.elt, 0, 0);
  drawSun(plant.sun);
  drawSparkles();

  const p = plant;
  const t = millis() / 1000;
  const w = p.wilt;
  drawMeadow(p, w, t);

  // stem: walk up one pixel at a time; wilting bends the top over
  const pts = [];
  let x = BASE_X;
  let y = GROUND + 1;
  for (let s = 0; s < p.height; s++) {
    const u = s / p.height;
    const ang = -HALF_PI + Math.sin(t * 1.3 + u * 2) * 0.05 * u + w * 2.5 * u * u;
    x += Math.cos(ang);
    y += Math.sin(ang);
    pts.push({ x, y, ang });
  }
  const stemC = mix(PAL.stem, PAL.dead, w);
  const stemD = mix(PAL.stemDark, PAL.deadDark, w);
  for (const q of pts) {
    px(q.x, q.y, stemC);
    px(q.x + 1, q.y, stemD);
  }

  for (const leaf of LilyLogic.leaves(p.height)) {
    const q = pts[Math.min(pts.length - 1, Math.max(0, Math.round(leaf.at)))];
    const ang = q.ang + leaf.side * (1.05 + w * 0.9);
    drawLeaf(q.x + (leaf.side > 0 ? 1 : 0), q.y, ang, 11 * leaf.size, leaf.side, w);
  }

  for (const f of LilyLogic.sideFlowers(p)) {
    const q = pts[Math.min(pts.length - 1, Math.max(0, Math.round(f.at)))];
    const ang = q.ang + f.side * (1.05 + w * 1.2); // out from the stem, drooping as it wilts
    const stalk = 13 * Math.min(1, f.grow * 2.5);
    let x = q.x + (f.side > 0 ? 1 : 0);
    let y = q.y;
    let a = ang;
    for (let u = 0; u < stalk; u += 0.5) {
      a = ang - f.side * 0.7 * (u / 13); // curves back up at the end
      x += Math.cos(a) * 0.5;
      y += Math.sin(a) * 0.5;
      px(x, y, stemC);
    }
    // a green bud for the first half of its growing, then it opens
    const head = { bud: Math.min(1, f.grow * 2), bloom: Math.max(0, f.grow * 2 - 1) };
    drawFlower(x, y, a + f.side * 0.2, head, w, 0.55);
  }

  const top = pts[pts.length - 1];
  const head = drawFlower(top.x, top.y, top.ang, p, w);
  if (head) lastHead = head;

  const now = millis() / 1000;
  for (const f of fallen) {
    const d = LilyLogic.petalDissolve(now - f.landed);
    drawPetal(f.x, f.y, f.ang, f.len, 1.6, 0, d > 0 ? fadeCols(f.cols, d) : f.cols, f.seed, d);
  }
  for (const f of falling) drawPetal(f.x, f.y, f.ang, f.len, 1.8, 0.3, f.cols, f.seed);

}

// lilies that come up out of the grass, behind the main plant: thinner
// stems, one leaf each, a bud that opens as they finish growing
function drawMeadow(p, w, t) {
  const stemC = mix(PAL.stem, PAL.dead, w);
  const clamp01 = (v) => Math.min(1, Math.max(0, v));
  for (const f of LilyLogic.meadowFlowers(p)) {
    const h = f.height * Math.min(1, f.grow * 1.6);
    const out = Math.sign(f.x); // they wilt away from the main stem
    let x = BASE_X + f.x;
    let y = GROUND + 1;
    let ang = -HALF_PI;
    let leafAt = null;
    for (let s = 0; s < h; s++) {
      const u = s / f.height;
      ang = -HALF_PI + Math.sin(t * 1.5 + f.x + u * 2) * 0.06 * u + out * w * 2.2 * u * u;
      x += Math.cos(ang);
      y += Math.sin(ang);
      px(x, y, stemC);
      if (s === Math.round(f.height * 0.4)) leafAt = { x, y, ang };
    }
    if (leafAt && f.grow > 0.35) {
      drawLeaf(leafAt.x, leafAt.y, leafAt.ang + out * (1.05 + w * 0.9), 7 * clamp01((f.grow - 0.35) / 0.3), out, w);
    }
    const head = { bud: clamp01((f.grow - 0.3) / 0.35), bloom: clamp01((f.grow - 0.65) / 0.35) };
    drawFlower(x, y, ang, head, w, f.size);
  }
}

// the sun lights up while the palm is open: it warms from a pale disc to gold,
// grows rays and a dithered glow
function drawSun(level) {
  const cx = GW - 22;
  const cy = 20;
  const r = 7;
  const t = millis() / 1000;
  const glow = r + 2 + 13 * level;
  for (let y = Math.floor(cy - glow); y <= cy + glow; y++) {
    for (let x = Math.floor(cx - glow); x <= cx + glow; x++) {
      const d = Math.hypot(x - cx, y - cy);
      if (d <= r || d > glow) continue;
      const g = level * (1 - (d - r) / (glow - r)) * 0.9;
      if (g > (BAYER[(y & 3) * 4 + (x & 3)] + 0.5) / 16) px(x, y, PAL.sunGlow);
    }
  }
  if (level > 0.1) {
    const len = 2 + 4 * level;
    for (let k = 0; k < 8; k++) {
      const a = (k / 8) * TWO_PI + t * 0.4;
      for (let u = r + 2; u < r + 2 + len; u += 0.5) px(cx + Math.cos(a) * u, cy + Math.sin(a) * u, PAL.sunEdge);
    }
  }
  const body = mix(PAL.sunDim, PAL.sun, level);
  const rim = mix(PAL.sunDimEdge, PAL.sunEdge, level);
  for (let y = -r; y <= r; y++) {
    for (let x = -r; x <= r; x++) {
      const d = Math.hypot(x, y);
      if (d > r + 0.3) continue;
      px(cx + x, cy + y, d > r - 1.2 ? rim : body);
    }
  }
  if (level > 0.5) {
    // a highlight up and to the left once it is properly lit
    for (const [x, y] of [[-3, -3], [-2, -3], [-3, -2], [-2, -4], [-4, -2]]) px(cx + x, cy + y, PAL.sunCore);
  }
}

function drawSparkles() {
  const t = millis() / 1000;
  for (let i = 0; i < 7; i++) {
    const on = Math.sin(t * (1.1 + hash(i, 5, 1)) + i * 2.3) > 0.55;
    if (!on) continue;
    const x = 6 + Math.floor(hash(i, 1, 1) * (GW - 12));
    const y = 4 + Math.floor(hash(i, 2, 1) * (GROUND * 0.55));
    px(x, y, PAL.sparkle);
    px(x - 1, y, "#ffd6e8");
    px(x + 1, y, "#ffd6e8");
    px(x, y - 1, "#ffd6e8");
    px(x, y + 1, "#ffd6e8");
  }
}

function drawLeaf(x0, y0, ang, len, side, wilt) {
  if (len < 1) return;
  const body = mix(PAL.leaf, PAL.dead, wilt);
  const rib = mix(PAL.leafDark, PAL.deadDark, wilt);
  let x = x0;
  let y = y0;
  for (let u = 0; u <= len; u += 0.5) {
    const tt = u / len;
    const a = ang + side * 0.5 * tt; // curls over at the tip
    const half = Math.sin(Math.PI * tt) * len * 0.22;
    const nx = -Math.sin(a);
    const ny = Math.cos(a);
    for (let v = -half; v <= half; v += 0.5) px(x + nx * v, y + ny * v, Math.abs(v) < 0.5 ? rib : body);
    x += Math.cos(a) * 0.5;
    y += Math.sin(a) * 0.5;
  }
}

// a dissolving petal also fades toward the sky behind it
function fadeCols(cols, k) {
  const to = PAL.sky[PAL.sky.length - 1];
  return {
    edge: mix(cols.edge, to, k),
    body: mix(cols.body, to, k),
    stripe: mix(cols.stripe, to, k),
    spot: cols.spot && mix(cols.spot, to, k),
  };
}

function petalColors(bloom, wilt) {
  // a shut bud is green; it turns pink as it opens
  const pinkness = Math.min(1, bloom * 3);
  return {
    edge: mix(mix(PAL.budEdge, PAL.petalEdge, pinkness), PAL.dusty, wilt),
    body: mix(mix(PAL.bud, PAL.petal, pinkness), PAL.dusty, wilt),
    stripe: mix(mix(PAL.blush, PAL.stripe, pinkness), PAL.dustyDark, wilt),
    spot: pinkness > 0.6 ? mix(PAL.spot, PAL.dustyDark, wilt) : null,
  };
}

// a lily petal: pointed at the tip, widest a third of the way out, white edge,
// hot pink stripe down the middle, dark spots near the throat
function drawPetal(x0, y0, ang, len, wid, curl, cols, seed, dissolve = 0) {
  let x = x0;
  let y = y0;
  for (let u = 0; u <= len; u += 0.5) {
    const tt = u / len;
    const a = ang + curl * tt * tt;
    const half = wid * Math.sin(Math.PI * Math.pow(tt, 0.65));
    const nx = -Math.sin(a);
    const ny = Math.cos(a);
    for (let v = -half; v <= half + 0.01; v += 0.5) {
      const edge = half > 0 ? Math.abs(v) / half : 0;
      let c = edge > 0.72 ? cols.edge : edge < 0.3 && tt > 0.1 ? cols.stripe : cols.body;
      if (cols.spot && tt > 0.12 && tt < 0.55 && hash(seed, Math.round(u * 2), Math.round(v * 2)) < 0.09) c = cols.spot;
      const qx = Math.round(x + nx * v);
      const qy = Math.round(y + ny * v);
      // dissolving drops pixels in a fixed random order until none are left
      if (dissolve > 0 && hash(seed + 31, qx, qy) < dissolve) continue;
      px(qx, qy, c);
    }
    x += Math.cos(a) * 0.5;
    y += Math.sin(a) * 0.5;
  }
}

const OPEN_SPREAD = [-2.62, -1.57, -0.52, 0.52, 1.57, 2.62]; // six petals, 60° apart
const BUD_SPREAD = [-0.16, -0.1, -0.03, 0.03, 0.1, 0.16];

// size 1 is the main lily; the little ones on the stem are drawn at 0.55.
// Returns where the head was drawn, so plucked petals can start there.
function drawFlower(hx, hy, headAng, p, wilt, size = 1) {
  if (p.bud <= 0.02) return null;
  const b = p.bloom * (1 - 0.55 * wilt);
  const ease = 1 - Math.pow(1 - b, 2);
  const len = p.bud * (12 + 10 * b) * size; // wider pinch, longer petals
  const wid = (1.6 + 2.6 * b) * size;
  const cols = petalColors(b, wilt);
  // draw the three back petals first so the front three overlap them
  for (const k of [0, 2, 4, 1, 3, 5]) {
    const offset = BUD_SPREAD[k] + (OPEN_SPREAD[k] - BUD_SPREAD[k]) * ease;
    const curl = Math.sign(OPEN_SPREAD[k]) * (0.9 * b + 0.6 * wilt); // lilies curl their petals back
    drawPetal(hx, hy, headAng + offset, len, wid, curl, cols, k + 1);
  }
  if (b > 0.3) {
    for (let k = 0; k < 6; k++) {
      const a = headAng + OPEN_SPREAD[k] * 0.78 * ease + 0.26;
      const L = len * 0.5 * b;
      for (let u = 1; u < L; u += 0.5) px(hx + Math.cos(a) * u, hy + Math.sin(a) * u, PAL.filament);
      px(hx + Math.cos(a) * L, hy + Math.sin(a) * L, PAL.anther);
      px(hx + Math.cos(a) * L + 1, hy + Math.sin(a) * L, PAL.anther);
    }
    px(hx, hy, mix(PAL.throat, PAL.dusty, wilt));
    px(hx + 1, hy, mix(PAL.throat, PAL.dusty, wilt));
    px(hx, hy + 1, mix(PAL.throat, PAL.dusty, wilt));
  }
  return { x: hx, y: hy, ang: headAng, len, cols };
}

function dropPetals() {
  if (!lastHead) return;
  for (let k = 0; k < 6; k++) {
    const a = lastHead.ang + OPEN_SPREAD[k];
    falling.push({
      x: lastHead.x + Math.cos(a) * 4,
      y: lastHead.y + Math.sin(a) * 4,
      vx: Math.cos(a) * 10 + (Math.random() - 0.5) * 6,
      vy: Math.sin(a) * 6 - 8,
      ang: a,
      spin: (Math.random() - 0.5) * 6,
      phase: Math.random() * 6,
      len: Math.max(6, lastHead.len * 0.55),
      cols: lastHead.cols,
      seed: 10 + k,
    });
  }
}

function updatePetals(dt) {
  const t = millis() / 1000;
  for (const f of falling) {
    f.vy = Math.min(f.vy + 30 * dt, 20); // slow, papery fall
    f.vx += Math.sin(t * 3 + f.phase) * 18 * dt;
    f.vx *= 1 - 1.2 * dt;
    f.x += f.vx * dt;
    f.y += f.vy * dt;
    f.ang += f.spin * dt;
  }
  for (const f of falling.filter((f) => f.y >= GROUND - 1)) {
    fallen.push({ x: f.x, y: GROUND - 1, ang: Math.cos(f.ang) > 0 ? 0.1 : Math.PI - 0.1, len: f.len, cols: f.cols, seed: f.seed, landed: t });
  }
  fallen = fallen.filter((f) => LilyLogic.petalDissolve(t - f.landed) < 1);
  falling = falling.filter((f) => f.y < GROUND - 1 && f.x > -10 && f.x < GW + 10);
  if (fallen.length > 36) fallen.splice(0, fallen.length - 36);
}

// for the automated browser checks
window.lilyState = () => ({
  mode,
  plant: { ...plant, pinchHistory: plant.pinchHistory.length },
  input: lastInput,
  counts: counts(),
  trained,
  training,
  stemGuess,
  lossHistory,
  falling: falling.length,
  fallen: fallen.length,
  hands: hands.length,
});
