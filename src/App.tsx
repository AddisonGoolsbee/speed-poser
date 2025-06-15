import { useRef, useEffect, useState, type JSX } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import * as poseDetection from "@tensorflow-models/pose-detection";
import { connections, TARGET_POSES, CONTROL_POSES } from "./poses";
import { PoseDisplay } from "./components/PoseDisplay";

// Define types for pose data
type Keypoint = {
  x: number;
  y: number;
};

type Pose = {
  name: string;
  [key: string]: Keypoint | string;
};

// Smoothing factor (0-1), higher means more smoothing
const SMOOTHING_FACTOR = 0.5;
// Number of frames to keep a keypoint after it's no longer detected
const PERSISTENCE_FRAMES = 10;
// Number of frames to hold the start pose for a point
const START_POSE_HOLD_FRAMES = 180;
// Number of frames to hold pose for a point
const HOLD_FRAMES = 90;
// Minimum similarity threshold to count as matching
const MATCH_THRESHOLD = 0.85;

// Draw the target pose on a canvas
const drawTargetPose = (ctx: CanvasRenderingContext2D, targetPose: Pose) => {
  const width = ctx.canvas.width;
  const height = ctx.canvas.height;

  // Clear canvas completely
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "transparent";
  ctx.fillRect(0, 0, width, height);

  // Find the lowest point in the pose
  let lowestY = 0;
  for (const key in targetPose) {
    if (key === "name") continue;
    const keypoint = targetPose[key];
    if (keypoint && typeof keypoint === "object" && "y" in keypoint) {
      lowestY = Math.max(lowestY, keypoint.y);
    }
  }

  // Calculate the scale factor to move the lowest point to 90% of height
  const targetLowestY = 0.9; // 90% from top = 10% from bottom
  const scaleY = targetLowestY / lowestY;

  // Draw connections
  ctx.strokeStyle = "#33FFFF"; // Brighter blue color for target pose
  ctx.lineWidth = 24; // Even thicker lines for better visibility
  for (const [start, end] of connections) {
    const startPos = targetPose[start];
    const endPos = targetPose[end];
    if (
      startPos &&
      endPos &&
      typeof startPos === "object" &&
      "x" in startPos &&
      "y" in startPos &&
      typeof endPos === "object" &&
      "x" in endPos &&
      "y" in endPos
    ) {
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

  if (
    nosePos &&
    leftEyePos &&
    rightEyePos &&
    typeof nosePos === "object" &&
    "x" in nosePos &&
    "y" in nosePos &&
    typeof leftEyePos === "object" &&
    "x" in leftEyePos &&
    "y" in leftEyePos &&
    typeof rightEyePos === "object" &&
    "x" in rightEyePos &&
    "y" in rightEyePos
  ) {
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
  const startPoseHoldFrames = useRef(0);

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
    targetPose: Pose
  ) => {
    let totalDiff = 0;
    let count = 0;

    for (const [key, targetPos] of Object.entries(targetPose)) {
      if (key === "name") continue;
      const currentPos = currentPose.get(key);
      if (
        currentPos &&
        typeof targetPos === "object" &&
        "x" in targetPos &&
        "y" in targetPos
      ) {
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

          // Check for Start pose when in idle state
          if (gameState === "idle") {
            const startPoseSimilarity = calculateSimilarity(
              lastKnownPositions.current,
              CONTROL_POSES.Start
            );
            if (startPoseSimilarity > MATCH_THRESHOLD) {
              startPoseHoldFrames.current++;
              if (startPoseHoldFrames.current >= START_POSE_HOLD_FRAMES) {
                startGame();
                startPoseHoldFrames.current = 0;
              }
            } else {
              startPoseHoldFrames.current = 0;
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
            holdFramesRef.current = Math.max(0, holdFramesRef.current - 1);
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
        if (gameState === "playing") {
          const targetCtx = targetCanvasRef.current?.getContext("2d");
          if (targetCtx) {
            targetCtx.clearRect(
              0,
              0,
              targetCtx.canvas.width,
              targetCtx.canvas.height
            );
            drawTargetPose(targetCtx, currentTargetPoseRef.current);
          }
        } else if (gameState === "idle") {
          const targetCtx = targetCanvasRef.current?.getContext("2d");
          if (targetCtx) {
            targetCtx.clearRect(
              0,
              0,
              targetCtx.canvas.width,
              targetCtx.canvas.height
            );
            drawTargetPose(targetCtx, CONTROL_POSES.Start);
          }
        }
        requestAnimationFrame(render);
      };
      render();
    }
    init();
  }, [gameState]);

  return (
    <div className="flex flex-col items-center justify-center h-screen w-screen bg-[#111] overflow-hidden">
      {/* Camera Feed Layer */}
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative h-full w-auto">
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            className="h-full w-auto object-contain transform -scale-x-100"
          />
          {(gameState === "playing" || gameState === "idle") && (
            <div className="absolute inset-0 w-full h-full pointer-events-none">
              <canvas
                ref={targetCanvasRef}
                width={1024}
                height={768}
                className="h-full w-auto object-contain opacity-50 absolute top-0 left-0 transform -scale-x-100"
              />
            </div>
          )}
          {gameState === "playing" && (
            <div className="text-white text-5xl font-bold tracking-wider">
              <div className="absolute top-8 left-8 bg-black/50 px-6 py-2 rounded-xl flex items-center gap-2">
                <span>⭐️</span>
                <span>{points}</span>
              </div>
              <div className="absolute top-8 right-8 bg-black/50 px-6 py-2 rounded-xl flex items-center gap-2">
                <span>⏱️</span>
                <span>{timeLeft}</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* UI Layer */}
      <div className="relative w-full h-full">
        {/* Start Screen */}
        {gameState === "idle" && (
          <div className="flex flex-col items-center justify-start h-full mt-8">
            <PoseDisplay
              poseName="Superstar"
              similarity={calculateSimilarity(
                lastKnownPositions.current,
                CONTROL_POSES.Start
              )}
              holdFrames={startPoseHoldFrames.current}
              holdFramesRequired={START_POSE_HOLD_FRAMES}
              matchThreshold={MATCH_THRESHOLD}
            />
            <div className="text-white text-4xl mt-8">
              Hold this pose for 3 seconds to begin
            </div>
          </div>
        )}

        {/* Game Screen */}
        {gameState === "playing" && (
          <div className="flex flex-col items-center h-full mt-8">
            <PoseDisplay
              poseName={currentTargetPose.name}
              similarity={similarity}
              holdFrames={holdFramesRef.current}
              holdFramesRequired={HOLD_FRAMES}
              matchThreshold={MATCH_THRESHOLD}
            />
          </div>
        )}

        {/* Game Over Screen */}
        {gameState === "finished" && (
          <>
            <div className="absolute inset-0 bg-black bg-opacity-30" />
            <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-gray-800 p-12 rounded-lg text-center">
              <h2 className="text-white text-6xl mb-8">Game Over!</h2>
              <p className="text-white text-4xl mb-12">
                Final Score: {finalScore}
              </p>
              <button
                onClick={playAgain}
                className="bg-blue-500 hover:bg-blue-600 text-white font-bold py-4 px-8 rounded-lg text-2xl"
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
