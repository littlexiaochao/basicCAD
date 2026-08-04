// =====================================================================
// 窄带溅射 Web Worker：负责一段三角形区间的溅射，返回该区间的局部
// distSq / sign，由主线程合并取最小值。
// =====================================================================
import { INF_COST, splatTriangleRange } from './splatKernel.js';

let positions = null;
let indices = null;

self.onmessage = (e) => {
  const msg = e.data;
  if (msg.type === 'init') {
    positions = msg.positions;
    indices = msg.indices;
    self.postMessage({ type: 'ready' });
    return;
  }
  if (msg.type === 'splat') {
    const {
      taskId,
      triStart,
      triEnd,
      resolution,
      minX, minY, minZ,
      cellX, cellY, cellZ,
      bandCells,
      collectNormals = false,
    } = msg;
    const total = resolution * resolution * resolution;
    const distSq = new Float32Array(total).fill(INF_COST);
    const sign = new Int8Array(total);
    const normals = collectNormals ? new Float32Array(total * 3) : null;
    splatTriangleRange(
      positions, indices,
      triStart, triEnd,
      resolution,
      minX, minY, minZ,
      cellX, cellY, cellZ,
      bandCells,
      distSq, sign, normals
    );
    // 转移缓冲区（零拷贝），主线程合并
    const transfer = [distSq.buffer, sign.buffer];
    if (normals) transfer.push(normals.buffer);
    self.postMessage(
      {
        type: 'result',
        taskId,
        distSq: distSq.buffer,
        sign: sign.buffer,
        normals: normals ? normals.buffer : null,
      },
      transfer
    );
  }
};
