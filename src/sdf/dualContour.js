import * as THREE from 'three';
import { sampleSDFValue } from './implicit.js';

// =====================================================================
// Hermite 数据 + Dual Contouring（双轮廓）等值面提取
// ---------------------------------------------------------------------
// Hermite 数据：每条穿越等值面的网格边，记录「交点位置 + 表面法线」。
//   法线直接复用窄带溅射时保存的最近面法线（sdf.normals），因此锐边 /
//   尖角两侧的相邻格点法线来自不同三角面，QEF 会把顶点推回真实棱线/角点
//   ——这正是 DC 相比 Marching Tetrahedra 能保持锐边特征的原因。
// 顶点：每个穿越等值面的网格单元求解二次误差函数（QEF）得到唯一顶点。
// 面：每条穿越边连接其周围至多 4 个单元的顶点，生成四边形（边界退化为三角）。
// 显示精度：step 控制粗网格步长（1=满分辨率，2/3=每 2/3 个格点取一个），
//   重新提取不需要重算 SDF，直接复用 sdf.data / sdf.normals。
// SDF 符号约定与 implicit.js 一致：内部为正，外部为负。
// =====================================================================

/** 判定一条边是否穿越等值面（d0/d1 为已减去 level 的符号距离） */
function crosses(d0, d1) {
  return (d0 <= 0 && d1 > 0) || (d0 > 0 && d1 <= 0);
}

/** 对称 3×3 矩阵 Jacobi 特征分解（m 行优先；v 的列是特征向量） */
function jacobi3(m, v) {
  for (let iter = 0; iter < 24; iter++) {
    let maxOff = 0;
    let p = 0;
    let q = 1;
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const o = Math.abs(m[a * 3 + b]);
        if (o > maxOff) {
          maxOff = o;
          p = a;
          q = b;
        }
      }
    }
    if (maxOff < 1e-14) break;
    const app = m[p * 3 + p];
    const aqq = m[q * 3 + q];
    const apq = m[p * 3 + q];
    const theta = (aqq - app) / (2 * apq);
    const t = Math.sign(theta) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1);
    const s = t * c;
    for (let k = 0; k < 3; k++) {
      if (k === p || k === q) continue;
      const mkp = m[k * 3 + p];
      const mkq = m[k * 3 + q];
      m[k * 3 + p] = c * mkp - s * mkq;
      m[k * 3 + q] = s * mkp + c * mkq;
      m[p * 3 + k] = m[k * 3 + p];
      m[q * 3 + k] = m[k * 3 + q];
    }
    m[p * 3 + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
    m[q * 3 + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
    m[p * 3 + q] = 0;
    m[q * 3 + p] = 0;
    for (let k = 0; k < 3; k++) {
      const vkp = v[k * 3 + p];
      const vkq = v[k * 3 + q];
      v[k * 3 + p] = c * vkp - s * vkq;
      v[k * 3 + q] = s * vkp + c * vkq;
    }
  }
}

/**
 * 求解单元的 QEF：minimize Σ (nᵢ·(x - pᵢ))²。
 * 先把原点移到样本质心再求最小范数解，退化（平面/直线）时不会跑飞；
 * 最后把顶点夹回单元包围盒，避免奇异 QEF 产生远离表面的顶点。
 */
function solveQEF(pts, norms, count, loX, loY, loZ, hiX, hiY, hiZ) {
  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < count; i++) {
    cx += pts[i * 3];
    cy += pts[i * 3 + 1];
    cz += pts[i * 3 + 2];
  }
  cx /= count;
  cy /= count;
  cz /= count;

  // A = Σ n nᵀ（对称），b = Σ n (n·(p - p̄))
  let a00 = 0;
  let a01 = 0;
  let a02 = 0;
  let a11 = 0;
  let a12 = 0;
  let a22 = 0;
  let b0 = 0;
  let b1 = 0;
  let b2 = 0;
  for (let i = 0; i < count; i++) {
    const nx = norms[i * 3];
    const ny = norms[i * 3 + 1];
    const nz = norms[i * 3 + 2];
    const px = pts[i * 3] - cx;
    const py = pts[i * 3 + 1] - cy;
    const pz = pts[i * 3 + 2] - cz;
    const nd = nx * px + ny * py + nz * pz;
    a00 += nx * nx;
    a01 += nx * ny;
    a02 += nx * nz;
    a11 += ny * ny;
    a12 += ny * nz;
    a22 += nz * nz;
    b0 += nx * nd;
    b1 += ny * nd;
    b2 += nz * nd;
  }

  const m = [a00, a01, a02, a01, a11, a12, a02, a12, a22];
  const v = [1, 0, 0, 0, 1, 0, 0, 0, 1];
  jacobi3(m, v);

  const lamMax = Math.max(Math.abs(m[0]), Math.abs(m[4]), Math.abs(m[8]));
  if (lamMax < 1e-14) return [cx, cy, cz];
  // 相对阈值取 1e-2：光滑表面单元内法线近似平行（λ₂/λ₁≈0.002~0.01），
  // 按秩 1 处理回退到样本质心（质心仍在表面上）；锐边/尖角单元的法线
  // 正交，特征值比接近 1，仍按满秩求解得到精确棱线/角点。阈值过小会把
  // 近零特征方向上的数值噪声放大，把 QEF 解推离表面并触发单元夹取。
  const eps = lamMax * 1e-2;
  let rank = 0;
  let d0 = 0;
  let d1 = 0;
  let d2 = 0;
  for (let e = 0; e < 3; e++) {
    const lam = m[e * 3 + e];
    if (Math.abs(lam) <= eps) continue;
    rank++;
    const dot = b0 * v[e * 3] + b1 * v[e * 3 + 1] + b2 * v[e * 3 + 2];
    const c = dot / lam;
    d0 += c * v[e * 3];
    d1 += c * v[e * 3 + 1];
    d2 += c * v[e * 3 + 2];
  }
  if (rank < 2) return [cx, cy, cz];

  let x = cx + d0;
  let y = cy + d1;
  let z = cz + d2;
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) {
    return [cx, cy, cz];
  }
  return [
    x < loX ? loX : x > hiX ? hiX : x,
    y < loY ? loY : y > hiY ? hiY : y,
    z < loZ ? loZ : z > hiZ ? hiZ : z,
  ];
}

/** 用 SDF 网格中心差分估算 p 处的梯度（朝向修正 / 法线兜底用） */
function gradientAt(sdf, px, py, pz) {
  const h = Math.min(sdf.cell.x, sdf.cell.y, sdf.cell.z) * 0.25;
  const gx = (sampleSDFValue(sdf, px + h, py, pz) - sampleSDFValue(sdf, px - h, py, pz)) / (2 * h);
  const gy = (sampleSDFValue(sdf, px, py + h, pz) - sampleSDFValue(sdf, px, py - h, pz)) / (2 * h);
  const gz = (sampleSDFValue(sdf, px, py, pz + h) - sampleSDFValue(sdf, px, py, pz - h)) / (2 * h);
  return [gx, gy, gz];
}

/**
 * Hermite 数据 + Dual Contouring 等值面提取。
 * @param {object} sdf meshToSDF 的返回值（含 collectNormals 时保存的溅射法线）
 * @param {object} [options]
 * @param {number} [options.step=1] 粗网格步长：1=满分辨率（最精细），2/3=每 2/3 个格点取一个
 * @param {number} [options.level=0] 等值面偏移（内部为正，level 为正时等值面外扩）
 * @returns {{ positions: Float32Array, indices: Uint32Array, vertexCount: number, triangleCount: number, edgeCount: number, cellCount: number }}
 */
export function dualContour(sdf, options = {}) {
  const { step = 1, level = 0 } = options;
  const k = Math.max(1, Math.round(step));
  const N = sdf.resolution;
  const data = sdf.data;
  const splatNormals = sdf.normals || null;
  const cell = sdf.cell;
  const minP = sdf.min;

  const nC = Math.floor((N - 1) / k) + 1; // 粗网格每轴格点数
  const nCells = nC - 1; // 粗网格每轴单元数
  const nCells3 = nCells * nCells * nCells;

  const ptIdx = (i, j, l) => (i * k * N + j * k) * N + l * k;
  const ptPos = (i, j, l, out) => {
    out[0] = minP.x + i * k * cell.x;
    out[1] = minP.y + j * k * cell.y;
    out[2] = minP.z + l * k * cell.z;
  };
  const cellIdx = (i, j, l) => (i * nCells + j) * nCells + l;

  /**
   * 粗边 (i,j,l)->(i+Δd,j+Δd,l+Δd) 的 4 个相邻单元（合法才给索引，否则 -1）。
   * 边的下端点粗格点坐标即 (i,j,l)，单元坐标必须落在 [0, nCells-1]。
   * 顺序为绕边一圈的循环序（面生成时按此顺序连四边形，避免蝴蝶结拓扑）。
   */
  const adjCells = (i, j, l, d, out) => {
    const ok = (a, b, c) => a >= 0 && b >= 0 && c >= 0 && a < nCells && b < nCells && c < nCells;
    if (d === 0) {
      out[0] = ok(i, j - 1, l - 1) ? cellIdx(i, j - 1, l - 1) : -1;
      out[1] = ok(i, j - 1, l) ? cellIdx(i, j - 1, l) : -1;
      out[2] = ok(i, j, l) ? cellIdx(i, j, l) : -1;
      out[3] = ok(i, j, l - 1) ? cellIdx(i, j, l - 1) : -1;
    } else if (d === 1) {
      out[0] = ok(i - 1, j, l - 1) ? cellIdx(i - 1, j, l - 1) : -1;
      out[1] = ok(i - 1, j, l) ? cellIdx(i - 1, j, l) : -1;
      out[2] = ok(i, j, l) ? cellIdx(i, j, l) : -1;
      out[3] = ok(i, j, l - 1) ? cellIdx(i, j, l - 1) : -1;
    } else {
      out[0] = ok(i - 1, j - 1, l) ? cellIdx(i - 1, j - 1, l) : -1;
      out[1] = ok(i - 1, j, l) ? cellIdx(i - 1, j, l) : -1;
      out[2] = ok(i, j, l) ? cellIdx(i, j, l) : -1;
      out[3] = ok(i, j - 1, l) ? cellIdx(i, j - 1, l) : -1;
    }
  };

  const hermitePts = [];
  const hermiteNorms = [];
  let edgeCount = 0;

  // ---------- Pass A：扫描粗边，收集 Hermite 数据并统计单元边数 ----------
  const cellEdgeCounts = new Uint32Array(nCells3);
  const p0 = [0, 0, 0];
  const p1 = [0, 0, 0];
  const adj = new Int32Array(4);

  const handleCrossing = (i, j, l, d, d0, d1) => {
    let t = 0.5;
    const denom = d0 - d1;
    if (Math.abs(denom) > 1e-30) t = d0 / denom;
    ptPos(i, j, l, p0);
    p1[0] = p0[0];
    p1[1] = p0[1];
    p1[2] = p0[2];
    if (d === 0) p1[0] = p0[0] + k * cell.x;
    else if (d === 1) p1[1] = p0[1] + k * cell.y;
    else p1[2] = p0[2] + k * cell.z;
    const px = p0[0] + (p1[0] - p0[0]) * t;
    const py = p0[1] + (p1[1] - p0[1]) * t;
    const pz = p0[2] + (p1[2] - p0[2]) * t;

    // 法线：优先取「负值侧（外部）端点」的溅射最近面法线。这条边穿越的
    // 表面正是外部端点刚离开的那个三角面，因此它的法线最接近真实表面法线；
    // 若用两端点平均，角点/棱线附近的“对角线”格点最近面有歧义，会把错误
    // 法线混入 QEF，导致顶点偏离真实角点。缺失时用另一端点的，再不行用
    // SDF 梯度兜底（QEF 对法线方向不敏感，正反皆可）。
    let nx = 0;
    let ny = 0;
    let nz = 0;
    if (splatNormals) {
      const i0 = ptIdx(i, j, l);
      const i1 = d === 0 ? ptIdx(i + 1, j, l) : d === 1 ? ptIdx(i, j + 1, l) : ptIdx(i, j, l + 1);
      const primary = d0 <= 0 ? i0 : i1; // 负值侧（外部）端点
      const fallback = primary === i0 ? i1 : i0;
      const readNormal = (idx) => {
        const ax = splatNormals[idx * 3];
        const ay = splatNormals[idx * 3 + 1];
        const az = splatNormals[idx * 3 + 2];
        const len = ax * ax + ay * ay + az * az;
        if (len <= 1e-12) return null;
        const inv = 1 / Math.sqrt(len);
        return [ax * inv, ay * inv, az * inv];
      };
      const n = readNormal(primary) || readNormal(fallback) || null;
      if (n) {
        nx = n[0];
        ny = n[1];
        nz = n[2];
      }
    }
    if (nx === 0 && ny === 0 && nz === 0) {
      const [gx, gy, gz] = gradientAt(sdf, px, py, pz);
      const glen = Math.hypot(gx, gy, gz);
      if (glen > 1e-12) {
        nx = gx / glen;
        ny = gy / glen;
        nz = gz / glen;
      } else {
        nx = 1;
      }
    }

    hermitePts.push(px, py, pz);
    hermiteNorms.push(nx, ny, nz);
    const id = edgeCount++;

    adjCells(i, j, l, d, adj);
    for (let c = 0; c < 4; c++) {
      if (adj[c] >= 0) cellEdgeCounts[adj[c]]++;
    }
    return id;
  };

  const checkEdge = (i, j, l, d, d0, d1) => {
    if (crosses(d0, d1)) handleCrossing(i, j, l, d, d0, d1);
  };

  for (let i = 0; i < nC; i++) {
    for (let j = 0; j < nC; j++) {
      for (let l = 0; l < nC; l++) {
        const idx = ptIdx(i, j, l);
        const dv = data[idx] - level;
        if (i < nCells) checkEdge(i, j, l, 0, dv, data[ptIdx(i + 1, j, l)] - level);
        if (j < nCells) checkEdge(i, j, l, 1, dv, data[ptIdx(i, j + 1, l)] - level);
        if (l < nCells) checkEdge(i, j, l, 2, dv, data[ptIdx(i, j, l + 1)] - level);
      }
    }
  }

  if (edgeCount === 0) {
    return {
      positions: new Float32Array(0),
      indices: new Uint32Array(0),
      vertexCount: 0,
      triangleCount: 0,
      edgeCount: 0,
      cellCount: 0,
    };
  }

  // ---------- Pass B：前缀和 → 每个单元列出它包含的穿越边 id ----------
  const cellOffset = new Uint32Array(nCells3 + 1);
  for (let i = 0; i < nCells3; i++) cellOffset[i + 1] = cellOffset[i] + cellEdgeCounts[i];
  const cellEdges = new Int32Array(cellOffset[nCells3]);
  const cursor = new Uint32Array(nCells3);
  cursor.set(cellOffset.subarray(0, nCells3));

  const pushEdgeToCells = (i, j, l, d, id) => {
    adjCells(i, j, l, d, adj);
    for (let c = 0; c < 4; c++) {
      if (adj[c] >= 0) cellEdges[cursor[adj[c]]++] = id;
    }
  };

  let passEdge = 0;
  for (let i = 0; i < nC; i++) {
    for (let j = 0; j < nC; j++) {
      for (let l = 0; l < nC; l++) {
        const idx = ptIdx(i, j, l);
        const dv = data[idx] - level;
        if (i < nCells && crosses(dv, data[ptIdx(i + 1, j, l)] - level)) {
          pushEdgeToCells(i, j, l, 0, passEdge++);
        }
        if (j < nCells && crosses(dv, data[ptIdx(i, j + 1, l)] - level)) {
          pushEdgeToCells(i, j, l, 1, passEdge++);
        }
        if (l < nCells && crosses(dv, data[ptIdx(i, j, l + 1)] - level)) {
          pushEdgeToCells(i, j, l, 2, passEdge++);
        }
      }
    }
  }

  // ---------- Pass C：每个活跃单元求解 QEF 得到一个顶点 ----------
  const cellVertex = new Int32Array(nCells3).fill(-1);
  const positions = new Float32Array(nCells3 * 3); // 按单元号占位，随后压缩
  const pts = new Float32Array(12 * 3);
  const norms = new Float32Array(12 * 3);
  let vertexCount = 0;

  for (let c = 0; c < nCells3; c++) {
    const start = cellOffset[c];
    const end = cellOffset[c + 1];
    if (start === end) continue;
    const nSamples = end - start;
    const nUse = Math.min(nSamples, 12);
    for (let s = 0; s < nUse; s++) {
      const eid = cellEdges[start + s];
      pts[s * 3] = hermitePts[eid * 3];
      pts[s * 3 + 1] = hermitePts[eid * 3 + 1];
      pts[s * 3 + 2] = hermitePts[eid * 3 + 2];
      norms[s * 3] = hermiteNorms[eid * 3];
      norms[s * 3 + 1] = hermiteNorms[eid * 3 + 1];
      norms[s * 3 + 2] = hermiteNorms[eid * 3 + 2];
    }
    const ck = Math.floor(c / (nCells * nCells));
    const cj = Math.floor(c / nCells) % nCells;
    const ci = c % nCells;
    const loX = minP.x + ck * k * cell.x;
    const loY = minP.y + cj * k * cell.y;
    const loZ = minP.z + ci * k * cell.z;
    const hiX = loX + k * cell.x;
    const hiY = loY + k * cell.y;
    const hiZ = loZ + k * cell.z;
    const [x, y, z] = solveQEF(pts, norms, nUse, loX, loY, loZ, hiX, hiY, hiZ);
    cellVertex[c] = vertexCount;
    positions[vertexCount * 3] = x;
    positions[vertexCount * 3 + 1] = y;
    positions[vertexCount * 3 + 2] = z;
    vertexCount++;
  }

  // ---------- Pass D：每条穿越边 → 连接周围单元的顶点生成面 ----------
  // 朝向：DC 生成的是四边形，用 SDF 梯度把每个三角面统一为「法线朝外」
  // （内部为正 → 外法线与梯度反方向）。避免依赖原始网格的绕序。
  const indices = [];
  const quad = new Int32Array(4);
  const v0 = new Float64Array(3);
  const v1 = new Float64Array(3);
  const v2 = new Float64Array(3);

  const pushTri = (a, b, c, gx, gy, gz) => {
    v0[0] = positions[a * 3];
    v0[1] = positions[a * 3 + 1];
    v0[2] = positions[a * 3 + 2];
    v1[0] = positions[b * 3];
    v1[1] = positions[b * 3 + 1];
    v1[2] = positions[b * 3 + 2];
    v2[0] = positions[c * 3];
    v2[1] = positions[c * 3 + 1];
    v2[2] = positions[c * 3 + 2];
    const e1x = v1[0] - v0[0];
    const e1y = v1[1] - v0[1];
    const e1z = v1[2] - v0[2];
    const e2x = v2[0] - v0[0];
    const e2y = v2[1] - v0[1];
    const e2z = v2[2] - v0[2];
    const nqx = e1y * e2z - e1z * e2y;
    const nqy = e1z * e2x - e1x * e2z;
    const nqz = e1x * e2y - e1y * e2x;
    if (nqx * gx + nqy * gy + nqz * gz > 0) {
      indices.push(a, c, b);
    } else {
      indices.push(a, b, c);
    }
  };

  const emitFace = (i, j, l, d) => {
    adjCells(i, j, l, d, quad);
    let nq = 0;
    for (let c = 0; c < 4; c++) {
      if (quad[c] >= 0 && cellVertex[quad[c]] >= 0) quad[nq++] = cellVertex[quad[c]];
    }
    if (nq < 3) return;
    // 面中心：平均顶点位置（粗网格单元角点，位于表面附近）
    const a = quad[0];
    const b = quad[1];
    const c = quad[2];
    let cx = positions[a * 3] + positions[b * 3];
    let cy = positions[a * 3 + 1] + positions[b * 3 + 1];
    let cz = positions[a * 3 + 2] + positions[b * 3 + 2];
    if (nq === 3) {
      cx += positions[c * 3];
      cy += positions[c * 3 + 1];
      cz += positions[c * 3 + 2];
      const inv = 1 / 3;
      const [gx, gy, gz] = gradientAt(sdf, cx * inv, cy * inv, cz * inv);
      pushTri(a, b, c, gx, gy, gz);
      return;
    }
    const dd = quad[3];
    cx += positions[c * 3] + positions[dd * 3];
    cy += positions[c * 3 + 1] + positions[dd * 3 + 1];
    cz += positions[c * 3 + 2] + positions[dd * 3 + 2];
    const inv = 1 / 4;
    const [gx, gy, gz] = gradientAt(sdf, cx * inv, cy * inv, cz * inv);
    pushTri(a, b, c, gx, gy, gz);
    pushTri(a, c, dd, gx, gy, gz);
  };

  for (let i = 0; i < nC; i++) {
    for (let j = 0; j < nC; j++) {
      for (let l = 0; l < nC; l++) {
        const idx = ptIdx(i, j, l);
        const dv = data[idx] - level;
        if (i < nCells && crosses(dv, data[ptIdx(i + 1, j, l)] - level)) {
          emitFace(i, j, l, 0);
        }
        if (j < nCells && crosses(dv, data[ptIdx(i, j + 1, l)] - level)) {
          emitFace(i, j, l, 1);
        }
        if (l < nCells && crosses(dv, data[ptIdx(i, j, l + 1)] - level)) {
          emitFace(i, j, l, 2);
        }
      }
    }
  }

  return {
    positions,
    indices: Uint32Array.from(indices),
    vertexCount,
    triangleCount: indices.length / 3,
    edgeCount,
    cellCount: vertexCount,
  };
}

/**
 * Dual Contouring 提取为 THREE.BufferGeometry（顶点法线由三角面平均）。
 * @param {object} sdf meshToSDF 的返回值
 * @param {object} [options] 同 dualContour
 * @returns {THREE.BufferGeometry}
 */
export function sdfToDualContourMesh(sdf, options = {}) {
  const { positions, indices, vertexCount } = dualContour(sdf, options);
  const geometry = new THREE.BufferGeometry();
  if (vertexCount === 0) return geometry;
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions.slice(0, vertexCount * 3), 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}
