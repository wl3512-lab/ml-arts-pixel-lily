// Starting point: "Training a Webcam Image Classifier with Convolutional Layers"
// by jackbdu, https://editor.p5js.org/jackbdu/sketches/7Y6VDvUO6 (copied unchanged)
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
  batchSize: 4,
};

let statusElement;
let aCount = 0;
let bCount = 0;
let aCountElement;
let bCountElement;

function preload() {
  classifier = ml5.neuralNetwork(options);
}

function setup() {
  createCanvas(400, 400);
  pixelDensity(1);
  statusElement = createP("Add examples for A and B, then Train!").style('font-size: 2em;');;
  createButton("Add Example A").mousePressed(addExampleA);
  aCountElement = createP(aCount);
  createButton("Add Example B").mousePressed(addExampleB);
  bCountElement = createP(bCount);
  createButton("Train").mousePressed(train);

  video = createCapture(VIDEO, { flipped: true });
  // video.size(imageWidth / pixelDensity(), imageHeight / pixelDensity());
  video.hide();

  ml5.setBackend("webgl");

  clearCanvas();
}

function draw() {
  image(video, 0, 0, width, height);
}

function addExampleA() {
  const inputs = getNormalizedPixels();
  const outputs = ['a'];
  classifier.addData(inputs, outputs);
  aCount = aCount + 1;
  aCountElement.html(aCount);
}

function addExampleB() {
  const inputs = getNormalizedPixels();
  const outputs = ['b'];
  classifier.addData(inputs, outputs);
  bCount = bCount + 1;
  bCountElement.html(bCount);
}

function getNormalizedPixels() {
  const img = get();
  img.pixelDensity(1);
  img.resize(imageWidth, imageHeight);
  img.loadPixels();
  const normalizedPixels = [];
  for (let i = 0; i < pixelsNum * imageChannels; i++) {
    normalizedPixels.push(img.pixels[i] / 255);
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
