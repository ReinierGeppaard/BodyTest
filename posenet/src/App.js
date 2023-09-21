import React, { useRef, useEffect } from "react";
import "./App.css";
import * as tf from "@tensorflow/tfjs";
import * as posenet from "@tensorflow-models/posenet";
import Webcam from "react-webcam";
import { drawKeypoints, drawSkeleton } from "./utilities";

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    let isMounted = true; // Track whether the component is mounted

    // Load PoseNet model (async function)
    const loadPoseNet = async () => {
      const net = await posenet.load({
        inputResolution: { width: 320, height: 240 }, // Lower resolution for CPU
        scale: 0.5,
      });

      // Function to run pose estimation
      const detect = async () => {
        if (!isMounted) return; // Prevent running when unmounted

        if (
          typeof webcamRef.current !== "undefined" &&
          webcamRef.current !== null &&
          webcamRef.current.video.readyState === 4
        ) {
          const video = webcamRef.current.video;

          // Create a TensorFlow.js tensor from the video feed
          const input = tf.browser.fromPixels(video);
          const pose = await net.estimateSinglePose(input);

          // Dispose of the input tensor when done (important for GPU)
          input.dispose();

          if (isMounted) {
            const ctx = canvasRef.current.getContext("2d");
            const videoWidth = video.videoWidth;
            const videoHeight = video.videoHeight;

            // Set canvas dimensions
            webcamRef.current.video.width = videoWidth;
            webcamRef.current.video.height = videoHeight;
            canvasRef.current.width = videoWidth;
            canvasRef.current.height = videoHeight;

            // Draw keypoints and skeleton
            drawKeypoints(pose.keypoints, 0.3, ctx, videoWidth / 640);
            drawSkeleton(pose.keypoints, 0.3, ctx, videoWidth / 640);
          }
        }
      };

      // Call detect function periodically (adjust interval as needed)
      const intervalId = setInterval(detect, 50); // Increase frequency for smoother animation

      // Clean up the interval when the component unmounts
      return () => {
        isMounted = false; // Mark the component as unmounted
        clearInterval(intervalId);
      };
    };

    loadPoseNet();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <Webcam
          ref={webcamRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
        <canvas
          ref={canvasRef}
          style={{
            position: "absolute",
            marginLeft: "auto",
            marginRight: "auto",
            left: 0,
            right: 0,
            textAlign: "center",
            zIndex: 9,
            width: 640,
            height: 480,
          }}
        />
      </header>
    </div>
  );
}

export default App;
