import { getAdjacentKeyPoints } from "@tensorflow-models/posenet";

export function drawKeypoints(keypoints, minConfidence, ctx, scale = 1) {
  keypoints.forEach((keypoint) => {
    if (keypoint.score >= minConfidence) {
      const { y, x } = keypoint.position;
      ctx.beginPath();
      ctx.arc(x * scale, y * scale, 3, 0, 2 * Math.PI);
      ctx.fillStyle = "white";
      ctx.fill();
    }
  });
}

export function drawSkeleton(keypoints, minConfidence, ctx, scale = 1) {
  const adjacentKeyPoints = getAdjacentKeyPoints(
    keypoints,
    minConfidence
  );

  ctx.strokeStyle = "red";
  ctx.lineWidth = 2;

  adjacentKeyPoints.forEach(([leftKey, rightKey]) => {
    const x1 = leftKey.position.x * scale;
    const y1 = leftKey.position.y * scale;
    const x2 = rightKey.position.x * scale;
    const y2 = rightKey.position.y * scale;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
  });
}
