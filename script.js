// Get video tag by Id
const video = document.getElementById("video");

Promise.all([
  // Small face detector
  faceapi.nets.tinyFaceDetector.loadFromUri("./models"),
  // Recognize face parts
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  // Recognize face box location
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  // Recognize face expressions
  faceapi.nets.faceExpressionNet.loadFromUri("./models")
]).then(startVideo);

// Hook up web cam to video HTML element
function startVideo() {
  navigator.getUserMedia(
    { video: {} },
    stream => {
      // Get video from web cam and set as source for video HTML element
      video.srcObject = stream;
    },
    err => {
      console.error(err);
    }
  );
}

// When video starts playing start recognizing faces
video.addEventListener("play", () => {
  // Create canvas from video
  const canvas = faceapi.createCanvasFromMedia(video);
  // Put canvas on page
  document.body.append(canvas);
  // Reize canvas over video
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);
  // Get all faces inside "video" every 100 ms
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video,
        // Detect with tiny face API
        new faceapi.TinyFaceDetectorOptions())
      .withFaceLandmarks()
      .withFaceExpressions();
    console.log(detections);
    // Display elements inside canvas
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    // Clear canvas before drawing
    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
    // Draw detections on canvas
    faceapi.draw.drawDetections(canvas, resizedDetections);
    faceapi.draw.drawFaceLandmarks(canvas, resizedDetections)
    faceapi.draw.drawFaceExpressions(canvas, resizedDetections);
  }, 100);
});
