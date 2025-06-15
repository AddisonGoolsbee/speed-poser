// src/App.tsx
import { useRef, useEffect, useState, type JSX } from "react";
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
// Number of frames to hold pose for a point
const HOLD_FRAMES = 90;
// Minimum similarity threshold to count as matching
const MATCH_THRESHOLD = 0.85;

// Target pose (arms up with proper head position)
const TARGET_POSES = [
  {
    left_shoulder: {
      x: 0.57,
      y: 0.27,
    },
    left_elbow: {
      x: 0.64,
      y: 0.22,
    },
    right_elbow: {
      x: 0.39,
      y: 0.22,
    },
    left_hip: {
      x: 0.54,
      y: 0.53,
    },
    right_hip: {
      x: 0.47,
      y: 0.52,
    },
    right_knee: {
      x: 0.43,
      y: 0.69,
    },
    left_knee: {
      x: 0.59,
      y: 0.7,
    },
    right_shoulder: {
      x: 0.45,
      y: 0.27,
    },
    left_wrist: {
      x: 0.71,
      y: 0.15,
    },
    right_wrist: {
      x: 0.33,
      y: 0.14,
    },
    left_ear: {
      x: 0.54,
      y: 0.22,
    },
    nose: {
      x: 0.51,
      y: 0.22,
    },
    right_ear: {
      x: 0.49,
      y: 0.22,
    },
    left_ankle: {
      x: 0.61,
      y: 0.84,
    },
    right_ankle: {
      x: 0.4,
      y: 0.83,
    },
    left_eye: {
      x: 0.52,
      y: 0.21,
    },
    right_eye: {
      x: 0.5,
      y: 0.21,
    },
  },
  {
    right_hip: {
      x: 0.45,
      y: 0.66,
    },
    left_hip: {
      x: 0.54,
      y: 0.66,
    },
    right_knee: {
      x: 0.35,
      y: 0.8,
    },
    left_shoulder: {
      x: 0.56,
      y: 0.37,
    },
    left_knee: {
      x: 0.65,
      y: 0.82,
    },
    left_elbow: {
      x: 0.64,
      y: 0.34,
    },
    right_shoulder: {
      x: 0.42,
      y: 0.38,
    },
    left_ear: {
      x: 0.52,
      y: 0.3,
    },
    right_ear: {
      x: 0.46,
      y: 0.31,
    },
    right_eye: {
      x: 0.47,
      y: 0.3,
    },
    left_ankle: {
      x: 0.54,
      y: 0.92,
    },
    left_eye: {
      x: 0.5,
      y: 0.3,
    },
    left_wrist: {
      x: 0.57,
      y: 0.26,
    },
    right_ankle: {
      x: 0.44,
      y: 0.91,
    },
    right_wrist: {
      x: 0.41,
      y: 0.25,
    },
    right_elbow: {
      x: 0.35,
      y: 0.33,
    },
    nose: {
      x: 0.49,
      y: 0.31,
    },
  },
  {
    left_hip: {
      x: 0.53,
      y: 0.61,
    },
    right_hip: {
      x: 0.47,
      y: 0.62,
    },
    left_knee: {
      x: 0.54,
      y: 0.78,
    },
    right_knee: {
      x: 0.45,
      y: 0.78,
    },
    left_elbow: {
      x: 0.61,
      y: 0.48,
    },
    left_shoulder: {
      x: 0.55,
      y: 0.4,
    },
    right_shoulder: {
      x: 0.44,
      y: 0.4,
    },
    right_elbow: {
      x: 0.39,
      y: 0.48,
    },
    left_wrist: {
      x: 0.69,
      y: 0.47,
    },
    right_wrist: {
      x: 0.31,
      y: 0.47,
    },
    right_ankle: {
      x: 0.39,
      y: 0.87,
    },
    left_ear: {
      x: 0.52,
      y: 0.33,
    },
    left_ankle: {
      x: 0.61,
      y: 0.89,
    },
    nose: {
      x: 0.5,
      y: 0.33,
    },
    right_ear: {
      x: 0.47,
      y: 0.33,
    },
    left_eye: {
      x: 0.51,
      y: 0.32,
    },
    right_eye: {
      x: 0.48,
      y: 0.32,
    },
  },
  {
    left_shoulder: {
      x: 0.49,
      y: 0.46,
    },
    right_shoulder: {
      x: 0.46,
      y: 0.31,
    },
    left_knee: {
      x: 0.45,
      y: 0.76,
    },
    nose: {
      x: 0.52,
      y: 0.37,
    },
    right_ear: {
      x: 0.52,
      y: 0.36,
    },
    right_knee: {
      x: 0.31,
      y: 0.76,
    },
    left_ankle: {
      x: 0.48,
      y: 0.91,
    },
    right_ankle: {
      x: 0.35,
      y: 0.88,
    },
    right_wrist: {
      x: 0.61,
      y: 0.27,
    },
    right_elbow: {
      x: 0.52,
      y: 0.26,
    },
    left_hip: {
      x: 0.4,
      y: 0.55,
    },
    right_hip: {
      x: 0.33,
      y: 0.52,
    },
    left_wrist: {
      x: 0.55,
      y: 0.67,
    },
    left_elbow: {
      x: 0.51,
      y: 0.57,
    },
    right_eye: {
      x: 0.53,
      y: 0.36,
    },
    left_ear: {
      x: 0.52,
      y: 0.42,
    },
    left_eye: {
      x: 0.53,
      y: 0.39,
    },
  },
  {
    left_shoulder: {
      x: 0.6,
      y: 0.23,
    },
    right_hip: {
      x: 0.51,
      y: 0.48,
    },
    left_hip: {
      x: 0.58,
      y: 0.48,
    },
    left_elbow: {
      x: 0.69,
      y: 0.22,
    },
    right_shoulder: {
      x: 0.47,
      y: 0.27,
    },
    left_ear: {
      x: 0.55,
      y: 0.16,
    },
    nose: {
      x: 0.52,
      y: 0.16,
    },
    right_ear: {
      x: 0.49,
      y: 0.17,
    },
    left_eye: {
      x: 0.53,
      y: 0.15,
    },
    right_eye: {
      x: 0.5,
      y: 0.15,
    },
    left_knee: {
      x: 0.55,
      y: 0.67,
    },
    right_knee: {
      x: 0.41,
      y: 0.58,
    },
    left_wrist: {
      x: 0.73,
      y: 0.14,
    },
    right_elbow: {
      x: 0.43,
      y: 0.37,
    },
    left_ankle: {
      x: 0.53,
      y: 0.83,
    },
    right_wrist: {
      x: 0.35,
      y: 0.42,
    },
    right_ankle: {
      x: 0.41,
      y: 0.76,
    },
  },
  {
    left_hip: {
      x: 0.6,
      y: 0.65,
    },
    right_hip: {
      x: 0.53,
      y: 0.65,
    },
    left_knee: {
      x: 0.54,
      y: 0.77,
    },
    right_knee: {
      x: 0.43,
      y: 0.71,
    },
    left_elbow: {
      x: 0.46,
      y: 0.55,
    },
    left_shoulder: {
      x: 0.54,
      y: 0.44,
    },
    right_shoulder: {
      x: 0.46,
      y: 0.44,
    },
    right_elbow: {
      x: 0.39,
      y: 0.45,
    },
    left_ankle: {
      x: 0.62,
      y: 0.9,
    },
    right_ear: {
      x: 0.47,
      y: 0.36,
    },
    left_ear: {
      x: 0.52,
      y: 0.36,
    },
    nose: {
      x: 0.48,
      y: 0.36,
    },
    left_eye: {
      x: 0.5,
      y: 0.34,
    },
    right_eye: {
      x: 0.47,
      y: 0.35,
    },
    right_wrist: {
      x: 0.31,
      y: 0.42,
    },
    right_ankle: {
      x: 0.51,
      y: 0.84,
    },
    left_wrist: {
      x: 0.38,
      y: 0.58,
    },
  },
  {
    left_hip: {
      x: 0.54,
      y: 0.48,
    },
    right_hip: {
      x: 0.47,
      y: 0.5,
    },
    right_shoulder: {
      x: 0.42,
      y: 0.26,
    },
    right_elbow: {
      x: 0.36,
      y: 0.36,
    },
    left_shoulder: {
      x: 0.53,
      y: 0.22,
    },
    left_elbow: {
      x: 0.6,
      y: 0.15,
    },
    left_ear: {
      x: 0.51,
      y: 0.17,
    },
    right_ear: {
      x: 0.46,
      y: 0.17,
    },
    left_eye: {
      x: 0.5,
      y: 0.16,
    },
    right_eye: {
      x: 0.47,
      y: 0.16,
    },
    right_knee: {
      x: 0.48,
      y: 0.69,
    },
    left_knee: {
      x: 0.65,
      y: 0.58,
    },
    left_ankle: {
      x: 0.53,
      y: 0.59,
    },
    right_wrist: {
      x: 0.43,
      y: 0.42,
    },
    left_wrist: {
      x: 0.54,
      y: 0.11,
    },
    right_ankle: {
      x: 0.51,
      y: 0.88,
    },
    nose: {
      x: 0.48,
      y: 0.18,
    },
  },
  {
    left_hip: {
      x: 0.54,
      y: 0.52,
    },
    right_hip: {
      x: 0.33,
      y: 0.54,
    },
    right_shoulder: {
      x: 0.21,
      y: 0.06,
    },
    right_elbow: {
      x: 0.15,
      y: 0.56,
    },
    left_shoulder: {
      x: 0.65,
      y: 0.08,
    },
    left_knee: {
      x: 0.48,
      y: 0.97,
    },
    right_eye: {
      x: 0.34,
      y: 0.08,
    },
    nose: {
      x: 0.39,
      y: 0.18,
    },
    left_ear: {
      x: 0.54,
      y: 0,
    },
    left_eye: {
      x: 0.45,
      y: 0.08,
    },
    right_ear: {
      x: 0.28,
      y: 0,
    },
    left_elbow: {
      x: 0.75,
      y: 0.51,
    },
    right_knee: {
      x: 0.31,
      y: 0.99,
    },
  },
];

// Draw the target pose on a canvas
const drawTargetPose = (
  ctx: CanvasRenderingContext2D,
  targetPose: (typeof TARGET_POSES)[0]
) => {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  // Clear canvas completely
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "transparent";
  ctx.fillRect(0, 0, width, height);

  // Find the lowest point in the pose
  let lowestY = 0;
  for (const keypoint of Object.values(targetPose)) {
    lowestY = Math.max(lowestY, keypoint.y);
  }

  // Calculate the scale factor to move the lowest point to 90% of height
  const targetLowestY = 0.9; // 90% from top = 10% from bottom
  const scaleY = targetLowestY / lowestY;

  // Draw connections
  ctx.strokeStyle = "#87CEEB"; // Light blue color for target pose
  ctx.lineWidth = 24; // Even thicker lines for better visibility
  for (const [start, end] of connections) {
    const startPos = targetPose[start as keyof typeof targetPose];
    const endPos = targetPose[end as keyof typeof targetPose];

    if (startPos && endPos) {
      ctx.beginPath();
      ctx.moveTo(startPos.x * width, startPos.y * height * scaleY);
      ctx.lineTo(endPos.x * width, endPos.y * height * scaleY);
      ctx.stroke();
    }
  }

  // Draw head circle
  const nosePos = targetPose.nose;
  const leftEyePos = targetPose.left_eye;
  const rightEyePos = targetPose.right_eye;

  if (nosePos && leftEyePos && rightEyePos) {
    const eyeDistance = Math.hypot(
      (rightEyePos.x - leftEyePos.x) * width,
      (rightEyePos.y - leftEyePos.y) * height * scaleY
    );
    const headRadius = eyeDistance * 1.5;

    ctx.beginPath();
    ctx.arc(
      nosePos.x * width,
      nosePos.y * height * scaleY,
      headRadius,
      0,
      2 * Math.PI
    );
    ctx.stroke();
  }
};

export default function App(): JSX.Element {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const targetCanvasRef = useRef<HTMLCanvasElement>(null);
  const lastKnownPositions = useRef<
    Map<string, { x: number; y: number; framesSinceLastSeen: number }>
  >(new Map());
  const [similarity, setSimilarity] = useState(0);
  const [points, setPoints] = useState(0);
  const holdFramesRef = useRef(0);
  const lastLogTime = useRef(0);
  const currentTargetPoseRef = useRef(
    TARGET_POSES[Math.floor(Math.random() * TARGET_POSES.length)]
  );
  const [currentTargetPose, setCurrentTargetPose] = useState(
    currentTargetPoseRef.current
  );
  const [gameState, setGameState] = useState<"idle" | "playing" | "finished">(
    "idle"
  );
  const [timeLeft, setTimeLeft] = useState(30);
  const [finalScore, setFinalScore] = useState(0);

  // Timer effect
  useEffect(() => {
    let timer: number;
    if (gameState === "playing" && timeLeft > 0) {
      timer = window.setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 1) {
            setGameState("finished");
            setFinalScore(points);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => {
      if (timer) clearInterval(timer);
    };
  }, [gameState, timeLeft, points]);

  const startGame = () => {
    setGameState("playing");
    setTimeLeft(30);
    setPoints(0);
    setFinalScore(0);
    // Reset target pose
    const newTargetPose =
      TARGET_POSES[Math.floor(Math.random() * TARGET_POSES.length)];
    currentTargetPoseRef.current = newTargetPose;
    setCurrentTargetPose(newTargetPose);

    // Force redraw of target pose
    const targetCtx = targetCanvasRef.current?.getContext("2d");
    if (targetCtx) {
      targetCtx.clearRect(
        0,
        0,
        targetCtx.canvas.width,
        targetCtx.canvas.height
      );
      drawTargetPose(targetCtx, newTargetPose);
    }
  };

  const playAgain = () => {
    startGame();
  };

  // Calculate similarity between current pose and target pose
  const calculateSimilarity = (
    currentPose: Map<string, { x: number; y: number }>,
    targetPose: (typeof TARGET_POSES)[0]
  ) => {
    let totalDiff = 0;
    let count = 0;

    for (const [key, targetPos] of Object.entries(targetPose)) {
      const currentPos = currentPose.get(key);
      if (currentPos) {
        // Normalize positions to 0-1 range
        const currentX = currentPos.x / 640;
        const currentY = currentPos.y / 480;

        // Calculate difference
        const diffX = Math.abs(currentX - targetPos.x);
        const diffY = Math.abs(currentY - targetPos.y);
        totalDiff += diffX + diffY;
        count++;
      }
    }

    if (count === 0) return 0;

    // Convert difference to similarity (0-1)
    const avgDiff = totalDiff / count;
    return Math.max(0, 1 - avgDiff * 2);
  };

  // Draw target pose when component mounts or target pose changes
  useEffect(() => {
    if (gameState === "playing") {
      const targetCtx = targetCanvasRef.current?.getContext("2d");
      if (targetCtx) {
        // Clear the canvas first
        targetCtx.clearRect(
          0,
          0,
          targetCtx.canvas.width,
          targetCtx.canvas.height
        );
        drawTargetPose(targetCtx, currentTargetPoseRef.current);
      }
    }
  }, [gameState, currentTargetPose]);

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

          // Log current pose every 5 seconds
          const now = Date.now();
          if (now - lastLogTime.current > 5000) {
            const poseObj: Record<string, { x: number; y: number }> = {};
            for (const [key, value] of lastKnownPositions.current.entries()) {
              poseObj[key] = {
                x: Number((value.x / 640).toFixed(2)),
                y: Number((value.y / 480).toFixed(2)),
              };
            }
            console.log("Current pose coordinates:");
            console.log(JSON.stringify(poseObj, null, 2));
            lastLogTime.current = now;
          }

          // Calculate similarity with current target pose
          const currentSimilarity = calculateSimilarity(
            lastKnownPositions.current,
            currentTargetPoseRef.current
          );
          setSimilarity(currentSimilarity);

          // Update hold frames and points
          if (currentSimilarity > MATCH_THRESHOLD) {
            holdFramesRef.current++;
            if (holdFramesRef.current >= HOLD_FRAMES) {
              setPoints((p) => p + 1);
              holdFramesRef.current = 0;
              // Change target pose when points are earned
              const newTargetPose =
                TARGET_POSES[Math.floor(Math.random() * TARGET_POSES.length)];
              currentTargetPoseRef.current = newTargetPose;
              setCurrentTargetPose(newTargetPose);
              // Force immediate redraw of target pose
              const targetCtx = targetCanvasRef.current?.getContext("2d");
              if (targetCtx) {
                targetCtx.clearRect(
                  0,
                  0,
                  targetCtx.canvas.width,
                  targetCtx.canvas.height
                );
                drawTargetPose(targetCtx, newTargetPose);
              }
            }
          } else {
            holdFramesRef.current = 0;
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
    <div className="flex flex-col items-center justify-center h-screen bg-gray-900">
      {/* Camera Feed Layer */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-[1024px] h-[768px]">
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="w-full h-full object-contain transform -scale-x-100"
          />
          {gameState === "playing" && (
            <div className="absolute inset-0 w-full h-full pointer-events-none">
              <canvas
                ref={targetCanvasRef}
                width={1024}
                height={768}
                className="w-full h-full object-contain opacity-50 absolute top-0 left-0 transform -scale-x-100"
              />
            </div>
          )}
        </div>
      </div>

      {/* UI Layer */}
      <div className="relative w-full h-full">
        {/* Start Screen */}
        {gameState === "idle" && (
          <button
            onClick={startGame}
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-blue-500 hover:bg-blue-600 text-white font-bold py-4 px-8 rounded-lg text-2xl"
          >
            Start Game
          </button>
        )}

        {/* Game Screen */}
        {gameState === "playing" && (
          <>
            <div className="absolute top-4 left-4 text-white text-2xl">
              Points: {points}
            </div>
            <div className="absolute top-4 right-4 text-white text-2xl">
              Time: {timeLeft}s
            </div>
            <div className="absolute top-16 right-4 text-white text-2xl">
              Match: {Math.round(similarity * 100)}%
            </div>
            <div className="absolute top-24 right-4 w-48 h-4 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-200"
                style={{ width: `${similarity * 100}%` }}
              />
            </div>
          </>
        )}

        {/* Game Over Screen */}
        {gameState === "finished" && (
          <>
            <div className="absolute inset-0 bg-black bg-opacity-30" />
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-800 p-8 rounded-lg text-center">
              <h2 className="text-white text-4xl mb-4">Game Over!</h2>
              <p className="text-white text-2xl mb-6">
                Final Score: {finalScore}
              </p>
              <button
                onClick={playAgain}
                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-6 rounded-lg text-xl"
              >
                Play Again
              </button>
            </div>
          </>
        )}
      </div>

      <video ref={videoRef} className="hidden" />
    </div>
  );
}
