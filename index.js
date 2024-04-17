const cocoSsd = require('@tensorflow-models/coco-ssd');
const tf = require('@tensorflow/tfjs-node');
const fs = require('fs').promises;

// Load the Coco SSD model and image.
Promise.all([cocoSsd.load(), fs.readFile('image1.jpg')])
.then((results) => {
  const model = results[0];
  const imgTensor = tf.node.decodeImage(new Uint8Array(results[1]), 3);
  return model.detect(imgTensor);
})
.then((predictions) => {
  console.log(JSON.stringify(predictions, null, 2));
});