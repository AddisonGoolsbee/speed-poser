// src/App.tsx
import { useRef, useEffect, type JSX } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";

// Define the connections between keypoints for MoveNet
const connections = [
  ["left_shoulder", "right_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

// Smoothing factor (0-1), higher means more smoothing
const SMOOTHING_FACTOR = 0.5;
// Number of frames to keep a keypoint after it's no longer detected
const PERSISTENCE_FRAMES = 10;

export default function App(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const lastKnownPositions = useRef<
    Map<string, { x: number; y: number; framesSinceLastSeen: number }>
  >(new Map());

  useEffect(() => {
    async function init() {
      await tf.ready();
      await tf.setBackend("webgl");
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        { modelType: "SinglePose.Lightning" }
      );
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      const ctx = canvasRef.current?.getContext("2d");
      const render = async () => {
        if (!ctx || !videoRef.current) return;
        ctx.drawImage(videoRef.current, 0, 0, 640, 480);
        const poses = await detector.estimatePoses(videoRef.current);

        // Increment frame counters and remove old positions
        for (const [key, value] of lastKnownPositions.current.entries()) {
          value.framesSinceLastSeen++;
          if (value.framesSinceLastSeen > PERSISTENCE_FRAMES) {
            lastKnownPositions.current.delete(key);
          }
        }

        if (poses[0]) {
          // Update positions with slight smoothing
          for (const keypoint of poses[0].keypoints) {
            if (keypoint.score && keypoint.score > 0.5 && keypoint.name) {
              const prevPos = lastKnownPositions.current.get(keypoint.name);
              if (prevPos) {
                // Apply smoothing
                const newX =
                  prevPos.x + (keypoint.x - prevPos.x) * (1 - SMOOTHING_FACTOR);
                const newY =
                  prevPos.y + (keypoint.y - prevPos.y) * (1 - SMOOTHING_FACTOR);
                lastKnownPositions.current.set(keypoint.name, {
                  x: newX,
                  y: newY,
                  framesSinceLastSeen: 0,
                });
              } else {
                lastKnownPositions.current.set(keypoint.name, {
                  x: keypoint.x,
                  y: keypoint.y,
                  framesSinceLastSeen: 0,
                });
              }
            }
          }

          // Draw connections
          ctx.strokeStyle = "#00FF00";
          ctx.lineWidth = 8;
          for (const [start, end] of connections) {
            const startPos = lastKnownPositions.current.get(start);
            const endPos = lastKnownPositions.current.get(end);

            if (startPos && endPos) {
              ctx.beginPath();
              ctx.moveTo(startPos.x, startPos.y);
              ctx.lineTo(endPos.x, endPos.y);
              ctx.stroke();
            }
          }

          // Draw head circle
          const nosePos = lastKnownPositions.current.get("nose");
          const leftEyePos = lastKnownPositions.current.get("left_eye");
          const rightEyePos = lastKnownPositions.current.get("right_eye");

          if (nosePos && leftEyePos && rightEyePos) {
            // Calculate head radius based on eye distance
            const eyeDistance = Math.hypot(
              rightEyePos.x - leftEyePos.x,
              rightEyePos.y - leftEyePos.y
            );
            const headRadius = eyeDistance * 1.5;

            ctx.beginPath();
            ctx.arc(
              nosePos.x,
              nosePos.y - headRadius / 4,
              headRadius,
              0,
              2 * Math.PI
            );
            ctx.strokeStyle = "#00FF00";
            ctx.lineWidth = 8;
            ctx.stroke();
          }
        }
        requestAnimationFrame(render);
      };
      render();
    }
    init();
  }, []);

  return (
    <div className="flex items-center justify-center h-screen">
      <canvas
        ref={canvasRef}
        width={640}
        height={480}
        className="transform -scale-x-100"
      />
      <video ref={videoRef} className="hidden" />
    </div>
  );
}
