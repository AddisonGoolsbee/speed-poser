// src/App.tsx
import { useRef, useEffect, type JSX } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";

export default function App(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

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
        if (poses[0]) {
          for (const { x, y, score } of poses[0].keypoints) {
            if (score && score > 0.5) {
              ctx.beginPath();
              ctx.arc(x, y, 5, 0, 2 * Math.PI);
              ctx.fill();
            }
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
