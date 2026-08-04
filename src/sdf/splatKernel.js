// =====================================================================
// 窄带溅射共享内核（主线程与 Web Worker 共用，不依赖 three）
// ---------------------------------------------------------------------
// splatTriangleRange：把 [triStart, triEnd) 区间内每个三角形的精确
// 点-三角距离溅射到其 AABB + 窄带范围内的格点上，并记录最近面法线的
// 符号猜测（负=猜测外部，正=猜测内部）。
// =====================================================================

/** 窄带外（远处）格点的占位距离（平方） */
export const INF_COST = 1e20;

/**
 * 计算点到三角形的最近点（Ericson 最近点算法），结果写入 out。
 */
export function closestPointOnTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, out) {
  const abx = bx - ax;
  const aby = by - ay;
  const abz = bz - az;
  const acx = cx - ax;
  const acy = cy - ay;
  const acz = cz - az;
  const apx = px - ax;
  const apy = py - ay;
  const apz = pz - az;
  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;
  if (d1 <= 0 && d2 <= 0) {
    out[0] = ax; out[1] = ay; out[2] = az;
    return;
  }
  const bpx = px - bx;
  const bpy = py - by;
  const bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;
  if (d3 >= 0 && d4 <= d3) {
    out[0] = bx; out[1] = by; out[2] = bz;
    return;
  }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const v = d1 / (d1 - d3);
    out[0] = ax + v * abx; out[1] = ay + v * aby; out[2] = az + v * abz;
    return;
  }
  const cpx = px - cx;
  const cpy = py - cy;
  const cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;
  if (d6 >= 0 && d5 <= d6) {
    out[0] = cx; out[1] = cy; out[2] = cz;
    return;
  }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const w = d2 / (d2 - d6);
    out[0] = ax + w * acx; out[1] = ay + w * acy; out[2] = az + w * acz;
    return;
  }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / (d4 - d3 + d5 - d6);
    out[0] = bx + w * (cx - bx); out[1] = by + w * (cy - by); out[2] = bz + w * (cz - bz);
    return;
  }
  const denom = 1 / (va + vb + vc);
  const v = vb * denom;
  const w = vc * denom;
  out[0] = ax + abx * v + acx * w;
  out[1] = ay + aby * v + acy * w;
  out[2] = az + abz * v + acz * w;
}

/**
 * 对 [triStart, triEnd) 区间的三角形执行窄带溅射，就地更新 distSq / sign。
 * @param {Float32Array} positions 顶点位置数组（3 分量）
 * @param {Uint32Array|null} indices 索引数组；null 表示非索引
 * @param {number} triStart
 * @param {number} triEnd
 * @param {number} resolution 每轴格点数 N
 * @param {number} minX
 * @param {number} minY
 * @param {number} minZ
 * @param {number} cellX
 * @param {number} cellY
 * @param {number} cellZ
 * @param {number} bandCells 窄带宽度（格数）
 * @param {Float32Array|Float64Array} distSq 距离平方（就地更新）
 * @param {Int8Array} sign 符号猜测（就地更新）
 * @param {Float32Array|null} normals 可选：每格点的最近面单位法线（3N³，就地更新，
 *        与 distSq 的胜者一致），用于 Hermite 数据 / Dual Contouring 的 QEF 求解
 */
export function splatTriangleRange(
  positions, indices,
  triStart, triEnd,
  resolution,
  minX, minY, minZ,
  cellX, cellY, cellZ,
  bandCells,
  distSq, sign, normals = null
) {
  const n = resolution;
  const triCount = triEnd - triStart;
  if (triCount <= 0) return;

  // 预取区间内三角形顶点与单位法线
  const A = new Float64Array(triCount * 3);
  const B = new Float64Array(triCount * 3);
  const C = new Float64Array(triCount * 3);
  const Nx = new Float64Array(triCount);
  const Ny = new Float64Array(triCount);
  const Nz = new Float64Array(triCount);
  const v = new Float64Array(3);
  const readV = (t, k, out) => {
    const vi = indices ? indices[t * 3 + k] : t * 3 + k;
    out[0] = positions[vi * 3];
    out[1] = positions[vi * 3 + 1];
    out[2] = positions[vi * 3 + 2];
  };
  for (let j = 0; j < triCount; j++) {
    const t = triStart + j;
    readV(t, 0, v);
    A[j * 3] = v[0]; A[j * 3 + 1] = v[1]; A[j * 3 + 2] = v[2];
    readV(t, 1, v);
    B[j * 3] = v[0]; B[j * 3 + 1] = v[1]; B[j * 3 + 2] = v[2];
    readV(t, 2, v);
    C[j * 3] = v[0]; C[j * 3 + 1] = v[1]; C[j * 3 + 2] = v[2];
    const ax = A[j * 3], ay = A[j * 3 + 1], az = A[j * 3 + 2];
    const ex = B[j * 3] - ax, ey = B[j * 3 + 1] - ay, ez = B[j * 3 + 2] - az;
    const fx = C[j * 3] - ax, fy = C[j * 3 + 1] - ay, fz = C[j * 3 + 2] - az;
    let nx = ey * fz - ez * fy;
    let ny = ez * fx - ex * fz;
    let nz = ex * fy - ey * fx;
    const len = Math.hypot(nx, ny, nz);
    if (len > 1e-12) {
      nx /= len; ny /= len; nz /= len;
    } else {
      nx = 0; ny = 0; nz = 0;
    }
    Nx[j] = nx; Ny[j] = ny; Nz[j] = nz;
  }

  const q = new Float64Array(3);
  for (let j = 0; j < triCount; j++) {
    const ax = A[j * 3], ay = A[j * 3 + 1], az = A[j * 3 + 2];
    const bx = B[j * 3], by = B[j * 3 + 1], bz = B[j * 3 + 2];
    const cx = C[j * 3], cy = C[j * 3 + 1], cz = C[j * 3 + 2];
    const aMinX = Math.min(ax, bx, cx), aMaxX = Math.max(ax, bx, cx);
    const aMinY = Math.min(ay, by, cy), aMaxY = Math.max(ay, by, cy);
    const aMinZ = Math.min(az, bz, cz), aMaxZ = Math.max(az, bz, cz);
    const i0x = Math.max(0, Math.floor((aMinX - minX) / cellX) - bandCells);
    const i1x = Math.min(n - 1, Math.ceil((aMaxX - minX) / cellX) + bandCells);
    const i0y = Math.max(0, Math.floor((aMinY - minY) / cellY) - bandCells);
    const i1y = Math.min(n - 1, Math.ceil((aMaxY - minY) / cellY) + bandCells);
    const i0z = Math.max(0, Math.floor((aMinZ - minZ) / cellZ) - bandCells);
    const i1z = Math.min(n - 1, Math.ceil((aMaxZ - minZ) / cellZ) + bandCells);
    const nx = Nx[j], ny = Ny[j], nz = Nz[j];
    for (let ix = i0x; ix <= i1x; ix++) {
      const px = minX + ix * cellX;
      for (let iy = i0y; iy <= i1y; iy++) {
        const py = minY + iy * cellY;
        const rowBase = (ix * n + iy) * n;
        for (let iz = i0z; iz <= i1z; iz++) {
          const pz = minZ + iz * cellZ;
          closestPointOnTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz, q);
          const dx = px - q[0];
          const dy = py - q[1];
          const dz = pz - q[2];
          const d2 = dx * dx + dy * dy + dz * dz;
          const idx = rowBase + iz;
          if (d2 < distSq[idx]) {
            distSq[idx] = d2;
            if (nx === 0 && ny === 0 && nz === 0) {
              sign[idx] = 1;
            } else {
              sign[idx] = dx * nx + dy * ny + dz * nz >= 0 ? -1 : 1;
            }
            if (normals) {
              normals[idx * 3] = nx;
              normals[idx * 3 + 1] = ny;
              normals[idx * 3 + 2] = nz;
            }
          }
        }
      }
    }
  }
}
