const video = document.getElementById("video");

Promise.all([
  // Small face detector
  faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
  // Register face parts
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  // Recognize face box location
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  // Recognize face expressions
  faceapi.nets.faceExpressionNet.loadFromUri("./models")
]).then(startVideo);

function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => {
      video.srcObject = stream;
    },
    err => {
      console.error(err);
    }
  );
}

video.addEventListener("play", () => {
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions()
      console.log(detections)
  }, 100);
});
