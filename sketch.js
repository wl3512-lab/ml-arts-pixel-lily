// Starting point: "Training a Webcam Image Classifier with Convolutional Layers"
// by jackbdu, https://editor.p5js.org/jackbdu/sketches/7Y6VDvUO6
// Changed: 3 labels instead of A/B, the CNN only sees the right half of the camera,
// pixels centered on 0 and batchSize 8 so training stops getting stuck on one label.
// References:
// https://editor.p5js.org/ima_ml/sketches/bL6jONrFQ

const imageWidth = 64;
const imageHeight = 64;
const imageChannels = 4;
const pixelsNum = imageWidth * imageHeight;

let video;

let classifier;
const options = {
  inputs: [imageWidth, imageHeight, imageChannels],
  task: "imageClassification",
  // learningRate: 0.1,
  debug: true,
};
const trainingOptions = {
  epochs: 64,
  batchSize: 8,
};

let statusElement;
const labels = ["open", "fist", "none"];
const counts = { open: 0, fist: 0, none: 0 };
let countsElement;

function preload() {
  classifier = ml5.neuralNetwork(options);
}

function setup() {
  createCanvas(640, 480);
  pixelDensity(1);
  statusElement = createP("Right hand on the right side: add open, fist and none, then Train!").style('font-size: 2em;');
  for (const label of labels) {
    createButton(`Add ${label}`).mousePressed(() => addExample(label));
  }
  countsElement = createP("open 0 · fist 0 · none 0");
  createButton("Train").mousePressed(train);

  video = createCapture(VIDEO, { flipped: true });
  // video.size(imageWidth / pixelDensity(), imageHeight / pixelDensity());
  video.hide();

  ml5.setBackend("webgl");

  clearCanvas();
}

function draw() {
  image(video, 0, 0, width, height);
  // the right half is what the CNN sees
  stroke(255, 105, 180);
  strokeWeight(3);
  line(width / 2, 0, width / 2, height);
}

function addExample(label) {
  classifier.addData(getNormalizedPixels(), [label]);
  counts[label] += 1;
  countsElement.html(`open ${counts.open} · fist ${counts.fist} · none ${counts.none}`);
}

function getNormalizedPixels() {
  // only the right half of the camera, where the stem hand goes
  const img = video.get(video.width / 2, 0, video.width / 2, video.height);
  img.pixelDensity(1);
  img.resize(imageWidth, imageHeight);
  img.loadPixels();
  const normalizedPixels = [];
  for (let i = 0; i < pixelsNum * imageChannels; i++) {
    // -0.5..0.5 instead of 0..1: with 0..1 the network often never left coin-flip
    normalizedPixels.push(img.pixels[i] / 255 - 0.5);
  }
  return normalizedPixels;
}

function train() {
  // classifier.normalizeData();
  // console.log(classifier.neuralNetworkData);
  classifier.train(trainingOptions, doneTraining);
  // console.log(classifier.data);
}

function doneTraining() {
  // console.log("done");
  statusElement.html("training done!");
  classify();
}

function classify() {
  const inputs = getNormalizedPixels();
  // console.log(inputs);
  classifier.classify(inputs, gotResults);
}

function gotResults(results) {
  // console.log(results);
  const label = results[0].label;
  const confidence = nf(results[0].confidence, 0, 2);
  statusElement.html(`${label} ${confidence}`);
  classify();
}

function clearCanvas() {
  background(255);
}
