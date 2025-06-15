import { type JSX } from "react";

type PoseDisplayProps = {
  poseName: string;
  similarity: number;
  holdFrames: number;
  holdFramesRequired: number;
  matchThreshold: number;
};

export const PoseDisplay = ({
  poseName,
  similarity,
  holdFrames,
  holdFramesRequired,
  matchThreshold,
}: PoseDisplayProps): JSX.Element => {
  return (
    <div className="text-center flex flex-row items-center justify-center gap-8 rounded-xl font-bold text-5xl text-white relative overflow-hidden py-2 px-4 tracking-wider">
      <div
        className="absolute inset-0 z-10"
        style={{
          backgroundColor:
            similarity >= matchThreshold
              ? "rgb(34, 197, 94)"
              : "rgb(239, 68, 68)",
          width: `${(holdFrames / holdFramesRequired) * 100}%`,
        }}
      />
      <div className="absolute inset-0 bg-black/50" />
      <div className="relative z-20 capitalize">{poseName}</div>
      <div className="relative z-20">{Math.round(similarity * 100)}%</div>
    </div>
  );
};
