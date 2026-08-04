import * as THREE from 'three';
import manifoldFactory from 'manifold-3d/manifold';
import { INF_COST, splatTriangleRange } from './splatKernel.js';

// =====================================================================
// 隐式表示（SDF）核心模块
// ---------------------------------------------------------------------
// 网格 → SDF：窄带溅射（splat）+ Web Worker 多核并行，与 OpenVDB / nTop
//             的 level set 生成思路一致——只在表面附近的窄带里计算精确距离，
//             远处用常量距离兜底（只参与符号洪泛分类，不影响等值面位置），
//             成本与网格分辨率解耦（O(表面积) 而非 O(N³)）。
// SDF → 网格：用 Manifold 3D（Google 维护的流形布尔/等值面库）的
//             levelSet（Marching Tetrahedra）提取等值面网格。
// 布尔运算：  Manifold 3D 的 union / difference / intersection
//             保证输出为流形（manifold）网格，适合后续切割与晶格化。
// 晶格化：    提供 gyroidSDF 等参数化隐式函数，可与模型 SDF 直接
//             min()/max() 组合（示例见 gyroidSDF / sampleLatticeSDF）。
// =====================================================================

// SDF 符号约定与 Manifold.levelSet 一致：内部为正，外部为负。

let _manifoldLocateFile = null;
let _manifoldPromise = null;

/**
 * 告诉本模块如何定位 manifold.wasm（Vite 下用 `?url` 导入的资产 URL）。
 * 必须在首次调用 getManifold() 之前设置。
 */
export function setManifoldLocateFile(fn) {
  _manifoldLocateFile = fn;
}

/** 懒初始化 Manifold WASM（仅首次调用时加载，之后复用单例） */
export async function getManifold() {
  if (!_manifoldPromise) {
    const p = (async () => {
      const mod = await manifoldFactory({
        locateFile: (file) => {
          if (file.endsWith('.wasm') && typeof _manifoldLocateFile === 'function') {
            return _manifoldLocateFile(file);
          }
          return file;
        },
      });
      mod.setup();
      return mod;
    })();
    // 初始化失败后允许下次重试
    _manifoldPromise = p;
    p.catch(() => {
      _manifoldPromise = null;
    });
  }
  return _manifoldPromise;
}

/**
 * 把对象树中所有可见 Mesh 合并成一个世界坐标下的索引三角网格。
 * @param {THREE.Object3D} root
 * @returns {{ geometry: THREE.BufferGeometry, bounds: THREE.Box3 }}
 */
export function mergeObjectGeometry(root) {
  const meshes = [];
  root.traverse((o) => {
    if (o.isMesh && o.geometry && o.visible !== false) meshes.push(o);
  });
  if (!meshes.length) throw new Error('当前模型没有可转换的三角网格');

  const positions = [];
  const indices = [];
  const bounds = new THREE.Box3();
  const va = new THREE.Vector3();

  for (const mesh of meshes) {
    const posAttr = mesh.geometry.attributes.position;
    if (!posAttr) continue;
    mesh.updateWorldMatrix(true, false);
    const matrix = mesh.matrixWorld;
    const localBase = positions.length / 3;
    for (let i = 0; i < posAttr.count; i++) {
      va.fromBufferAttribute(posAttr, i).applyMatrix4(matrix);
      positions.push(va.x, va.y, va.z);
      bounds.expandByPoint(va);
    }
    const idxAttr = mesh.geometry.index;
    if (idxAttr) {
      for (let i = 0; i < idxAttr.count; i++) {
        const v = idxAttr.getX(i);
        if (v >= 0 && v < posAttr.count) indices.push(localBase + v);
      }
    } else {
      for (let i = 0; i < posAttr.count; i++) indices.push(localBase + i);
    }
  }

  if (bounds.isEmpty()) throw new Error('模型包围盒为空，无法转换为隐式表示');

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setIndex(indices);
  geometry.computeBoundingBox();
  return { geometry, bounds };
}

const yieldToUI = () => new Promise((r) => setTimeout(r, 0));

/**
 * 判断三角网格是否封闭（水密）：
 * 每条边恰好被 2 个三角形共享、且无非流形边。退化（零面积）三角形不参与
 * 边的计数，也不影响封闭性——它们只是被焊接/量化压扁的细长条，不产生
 * 开放边界。顶点先按包围盒尺度量化合并，容忍 STL / STEP 数值噪声。
 * @param {THREE.BufferGeometry} geometry
 * @returns {{ closed: boolean, openEdges: number, nonManifoldEdges: number, degenerate: number }}
 */
export function analyzeMeshClosedness(geometry, epsRatio = 2e-4) {
  const posAttr = geometry.attributes.position;
  const index = geometry.index;
  if (!posAttr) {
    return { closed: false, openEdges: -1, nonManifoldEdges: -1, degenerate: -1 };
  }
  const p = posAttr.array;
  const nVerts = posAttr.count;
  const box = new THREE.Box3().setFromBufferAttribute(posAttr);
  const maxDim =
    Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z) || 1;
  const eps = maxDim * epsRatio;
  const vMap = new Map();
  const canonical = new Int32Array(nVerts);
  for (let i = 0; i < nVerts; i++) {
    const key = `${Math.round(p[i * 3] / eps)},${Math.round(p[i * 3 + 1] / eps)},${Math.round(p[i * 3 + 2] / eps)}`;
    let id = vMap.get(key);
    if (id === undefined) {
      id = vMap.size;
      vMap.set(key, id);
    }
    canonical[i] = id;
  }

  const triCount = index ? index.count / 3 : nVerts / 3;
  const edgeMap = new Map();
  const keyOf = (a, b) => (a < b ? `${a}:${b}` : `${b}:${a}`);
  const addEdge = (ia, ib) => {
    if (canonical[ia] === canonical[ib]) return;
    const k = keyOf(canonical[ia], canonical[ib]);
    const e = edgeMap.get(k);
    if (e) e.count++;
    else edgeMap.set(k, { count: 1 });
  };
  let degenerate = 0;
  if (index) {
    for (let t = 0; t < triCount; t++) {
      const ia = index.getX(t * 3);
      const ib = index.getX(t * 3 + 1);
      const ic = index.getX(t * 3 + 2);
      if (ia === ib || ib === ic || ia === ic) {
        degenerate++;
        continue;
      }
      addEdge(ia, ib);
      addEdge(ib, ic);
      addEdge(ic, ia);
    }
  } else {
    for (let t = 0; t < triCount; t++) {
      const ia = t * 3;
      const ib = t * 3 + 1;
      const ic = t * 3 + 2;
      addEdge(ia, ib);
      addEdge(ib, ic);
      addEdge(ic, ia);
    }
  }
  let openEdges = 0;
  let nonManifoldEdges = 0;
  for (const e of edgeMap.values()) {
    if (e.count === 1) openEdges++;
    else if (e.count > 2) nonManifoldEdges++;
  }
  const closed = openEdges === 0 && nonManifoldEdges === 0;
  return { closed, openEdges, nonManifoldEdges, degenerate };
}

/**
 * 直接分析对象树（模型组）的封闭性：合并所有网格后做水密检测。
 * 供导入模型时立即显示“封闭 / 开放”，转换隐式时复用同一判定。
 * @param {THREE.Object3D} root
 * @returns {{ closed: boolean, openEdges: number, nonManifoldEdges: number, degenerate: number }}
 */
export function analyzeRootClosedness(root) {
  const { geometry } = mergeObjectGeometry(root);
  const result = analyzeMeshClosedness(geometry);
  geometry.dispose();
  return result;
}

/**
 * 窄带溅射（内联版）：把每个三角形的精确距离“溅射”到其 AABB 附近的格点上。
 * 只计算表面附近 bandCells 个格点范围内的精确距离，远处保持 INF_COST。
 * 符号（最近面法线侧：负=外部猜测）只在溅射到的格点记录，远处的符号由
 * 洪泛法按连通性修正。分块执行并在块间让出主线程，保持 UI 响应。
 */
async function splatNarrowBand(geometry, { resolution, min, cell, bandCells, onProgress, signal, collectNormals = false }) {
  const pos = geometry.attributes.position.array;
  const index = geometry.index ? geometry.index.array : null;
  const n = resolution;
  const total = n * n * n;
  const distSq = new Float32Array(total).fill(INF_COST);
  const sign = new Int8Array(total);
  const normals = collectNormals ? new Float32Array(total * 3) : null;
  const triCount = index ? index.length / 3 : pos.length / 9;

  const CHUNK = 2048;
  for (let start = 0; start < triCount; start += CHUNK) {
    const end = Math.min(triCount, start + CHUNK);
    splatTriangleRange(
      pos, index,
      start, end,
      n,
      min.x, min.y, min.z,
      cell.x, cell.y, cell.z,
      bandCells,
      distSq, sign, normals
    );
    if (typeof onProgress === 'function') onProgress(0.05 + (end / triCount) * 0.85);
    await yieldToUI();
    if (signal && signal.aborted) {
      throw new DOMException('隐式转换已取消', 'AbortError');
    }
  }
  return { distSq, sign, normals };
}

// ---------- Web Worker 并行溅射 ----------

let _workerSeq = 0;
let _workerPoolPromise = null;

/** 计算 Worker 数量：留一个核给主线程，上限 8；分辨率越高每个 Worker 的
 *  全场缓冲越大，按分辨率动态下调数量以控制内存。 */
function getWorkerCount(resolution = 0) {
  const cores = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  const byCores = Math.min(Math.max(1, cores - 1), 8);
  const byMemory =
    resolution >= 224 ? 4 :
    resolution >= 160 ? 6 :
    8;
  return Math.min(byCores, byMemory);
}

/**
 * 预热 Worker 池（只创建 Worker，不绑定几何）。在模型加载时调用，
 * 让首次隐式转换不用承担创建开销。
 */
export function warmSplatWorkers() {
  if (typeof Worker === 'undefined' || typeof window === 'undefined') return;
  const count = getWorkerCount();
  if (count >= 2) {
    getWorkerPool(count).catch(() => {
      _workerPoolPromise = null;
    });
  }
}

/** 懒创建并复用 Worker 池（三角形区间相互独立，天然可并行） */
async function getWorkerPool(count) {
  if (!_workerPoolPromise) {
    _workerPoolPromise = (async () => {
      const workers = [];
      for (let i = 0; i < count; i++) {
        const w = new Worker(new URL('./splatWorker.js', import.meta.url), { type: 'module' });
        w._id = ++_workerSeq;
        w._dead = false;
        w._pending = new Map();
        w.onmessage = (e) => {
          const msg = e.data;
          const p = w._pending.get(msg.taskId);
          if (p) {
            w._pending.delete(msg.taskId);
            p.resolve(msg);
          }
        };
        w.onerror = (err) => {
          w._dead = true;
          for (const [, p] of w._pending) p.reject(new Error(`Worker 错误：${err.message || err.type}`));
          w._pending.clear();
        };
        workers.push(w);
      }
      return workers;
    })();
  }
  return _workerPoolPromise;
}

/** 向 Worker 发送消息并等待响应（按 taskId 匹配） */
function postToWorker(worker, msg) {
  return new Promise((resolve, reject) => {
    worker._pending.set(msg.taskId, { resolve, reject });
    worker.postMessage(msg);
  });
}

/**
 * 窄带溅射（Web Worker 并行版）：三角形区间按核数拆分，各 Worker 返回
 * 局部 distSq / sign，主线程合并取最小值。结果与内联版完全一致。
 * 不满足并行条件（模型太小 / 无 Worker / 单核）时回退内联。
 */
async function splatNarrowBandParallel(geometry, options) {
  const { resolution, min, cell, bandCells, onProgress, signal, useWorkers = 'auto', collectNormals = false } = options;
  const pos = geometry.attributes.position.array;
  const index = geometry.index ? geometry.index.array : null;
  const triCount = index ? index.length / 3 : pos.length / 9;
  const total = resolution * resolution * resolution;
  const workerCount = getWorkerCount(resolution);
  const wantWorkers =
    useWorkers === true ||
    (useWorkers === 'auto' && triCount >= 4000 && workerCount >= 2);
  if (!wantWorkers || typeof Worker === 'undefined' || typeof window === 'undefined') {
    return splatNarrowBand(geometry, options);
  }

  const pool = await getWorkerPool(workerCount);
  // 每个 Worker 先收到几何数据；消息按序处理，init 一定先于 splat
  for (const w of pool) {
    if (w._dead) continue;
    w.postMessage({ type: 'init', positions: pos, indices: index });
  }

  const taskCount = workerCount * 2;
  const per = Math.ceil(triCount / taskCount);
  const tasks = [];
  for (let i = 0; i < taskCount; i++) {
    const start = i * per;
    if (start >= triCount) break;
    tasks.push({ start, end: Math.min(triCount, (i + 1) * per) });
  }

  const distSq = new Float32Array(total).fill(INF_COST);
  const sign = new Int8Array(total);
  const normals = collectNormals ? new Float32Array(total * 3) : null;
  let completed = 0;
  let settled = false;

  await new Promise((resolve, reject) => {
    let next = 0;
    const finish = (fn, arg) => {
      if (!settled) {
        settled = true;
        fn(arg);
      }
    };
    const assign = (worker) => {
      if (settled || worker._dead) return;
      if (next >= tasks.length) return;
      const task = tasks[next++];
      const taskId = task.start;
      postToWorker(worker, {
        type: 'splat',
        taskId,
        triStart: task.start,
        triEnd: task.end,
        resolution,
        minX: min.x, minY: min.y, minZ: min.z,
        cellX: cell.x, cellY: cell.y, cellZ: cell.z,
        bandCells,
        collectNormals,
      })
        .then((res) => {
          // 合并局部结果（取最小距离与对应符号）
          const d = new Float32Array(res.distSq);
          const s = new Int8Array(res.sign);
          const nl = res.normals ? new Float32Array(res.normals) : null;
          for (let i = 0; i < d.length; i++) {
            if (d[i] < distSq[i]) {
              distSq[i] = d[i];
              sign[i] = s[i];
              if (nl) {
                normals[i * 3] = nl[i * 3];
                normals[i * 3 + 1] = nl[i * 3 + 1];
                normals[i * 3 + 2] = nl[i * 3 + 2];
              }
            }
          }
          completed++;
          if (typeof onProgress === 'function') onProgress(0.05 + (completed / tasks.length) * 0.85);
          if (signal && signal.aborted) {
            finish(reject, new DOMException('隐式转换已取消', 'AbortError'));
            return;
          }
          assign(worker);
          if (completed === tasks.length) finish(resolve);
        })
        .catch((err) => finish(reject, err));
    };
    for (const w of pool) assign(w);
    if (settled === false && next === 0) {
      // 所有 Worker 都已失效，回退内联
      finish(reject, new Error('Web Worker 全部不可用'));
    }
  });

  return { distSq, sign, normals, parallel: true, workers: workerCount };
}

/**
 * 由窄带溅射结果生成无符号距离场：窄带内为精确距离，远处用常量 farDist
 * （只影响洪泛符号分类与等值面之外的单调性，不影响表面位置）。
 */
function buildDistanceField(distSq, sign, far) {
  const dist = new Float32Array(distSq.length);
  for (let i = 0; i < dist.length; i++) {
    dist[i] = distSq[i] >= INF_COST * 0.5 ? far : Math.sqrt(distSq[i]);
  }
  return { dist, sign };
}

/**
 * 洪泛法符号修正（对封闭网格）：
 * 从网格边界（必为外部）沿 dist>eps 的区域扩散；所有可达格点标为外部，
 * 不可达的 dist>eps 格点必然位于封闭曲面内部。表面附近（dist<=eps）的
 * 格点保留最近面法线的符号猜测。这一修正可消除符号翻转造成的点状/杆状碎屑。
 * @returns {Float32Array} 有符号距离（内部为正，外部为负）
 */
function correctSignByFloodFill(dist, sign, resolution, eps) {
  const n = resolution;
  const total = n * n * n;
  const visited = new Uint8Array(total);
  const queue = new Int32Array(total);
  const at = (x, y, z) => (x * n + y) * n + z;
  let head = 0;
  let tail = 0;
  const push = (i) => {
    if (!visited[i]) {
      visited[i] = 1;
      queue[tail++] = i;
    }
  };
  // 边界格点（远离表面的外部起点）
  for (let x = 0; x < n; x++) {
    for (let y = 0; y < n; y++) {
      for (let z = 0; z < n; z++) {
        if (
          (x === 0 || y === 0 || z === 0 || x === n - 1 || y === n - 1 || z === n - 1) &&
          dist[at(x, y, z)] > eps
        ) {
          push(at(x, y, z));
        }
      }
    }
  }
  // 6 邻域扩散，只穿过 dist>eps 的“远处”区域
  while (head < tail) {
    const i = queue[head++];
    const x = ((i / n) / n) | 0;
    const y = ((i / n) | 0) % n;
    const z = i % n;
    if (x > 0) {
      const j = at(x - 1, y, z);
      if (dist[j] > eps) push(j);
    }
    if (x < n - 1) {
      const j = at(x + 1, y, z);
      if (dist[j] > eps) push(j);
    }
    if (y > 0) {
      const j = at(x, y - 1, z);
      if (dist[j] > eps) push(j);
    }
    if (y < n - 1) {
      const j = at(x, y + 1, z);
      if (dist[j] > eps) push(j);
    }
    if (z > 0) {
      const j = at(x, y, z - 1);
      if (dist[j] > eps) push(j);
    }
    if (z < n - 1) {
      const j = at(x, y, z + 1);
      if (dist[j] > eps) push(j);
    }
  }
  const signed = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    if (dist[i] > eps) {
      signed[i] = visited[i] ? -dist[i] : dist[i];
    } else {
      signed[i] = sign[i] * dist[i];
    }
  }
  return signed;
}

/**
 * 由无符号距离 + 符号猜测生成最终 SDF：
 * 开放网格 → 壳体（thickness - dist）；封闭网格 → 洪泛符号修正后的有符号距离。
 */
function finalizeSignedField(dist, sign, mode, thickness, resolution, cell) {
  if (mode === 'open') {
    const data = new Float32Array(dist.length);
    for (let i = 0; i < dist.length; i++) data[i] = thickness - dist[i];
    return data;
  }
  const eps = Math.max(cell.x, cell.y, cell.z) * 1.25;
  return correctSignByFloodFill(dist, sign, resolution, eps);
}

/**
 * 将网格转换为规则网格 SDF（隐式表示）。
 * - 自动检测封闭性：封闭网格 → 稳健有符号距离（洪泛符号修正）；
 *   开放网格 → 壳体模式（无符号距离 + 厚度，两侧对称薄壁，无需符号判定）。
 * @param {THREE.Object3D} root 模型对象树
 * @param {object} [options]
 * @param {number} [options.resolution=64] 每轴采样数（N³）
 * @param {number} [options.padding=0.12] 包围盒外扩比例（相对最大边长）
 * @param {number} [options.thickness=0.04] 壳体厚度（开放网格的薄壁半宽，世界单位）
 * @param {'closed'|'open'|null} [options.forceMode] 强制转换模式（默认自动检测）
 * @param {'auto'|boolean} [options.useWorkers] 是否使用 Web Worker 并行溅射（默认 auto）
 * @param {boolean} [options.collectNormals=false] 是否同时保存每个格点的最近面法线
 *        （Hermite 数据，供 Dual Contouring 提取锐边使用；额外占用 3×N³×4B 内存）
 * @param {(progress:number)=>void} [options.onProgress] 进度回调 0~1
 * @param {AbortSignal} [options.signal] 取消信号
 * @returns {Promise<object>} { data, normals, resolution, min, max, cell, bounds, mode, thickness, closedness, parallel }
 */
export async function meshToSDF(root, options = {}) {
  const {
    resolution = 64,
    padding = 0.12,
    thickness = 0.04,
    forceMode = null,
    useWorkers = 'auto',
    collectNormals = false,
    onProgress = null,
    signal = null,
  } = options;
  if (!Number.isInteger(resolution) || resolution < 16 || resolution > 256) {
    throw new Error(`分辨率需为 16~256 的整数（当前 ${resolution}）`);
  }
  if (!(thickness > 0)) {
    throw new Error('壳体厚度需为正数');
  }

  const { geometry, bounds } = mergeObjectGeometry(root);
  const size = bounds.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const pad = maxDim * padding;
  const min = new THREE.Vector3(bounds.min.x - pad, bounds.min.y - pad, bounds.min.z - pad);
  const max = new THREE.Vector3(bounds.max.x + pad, bounds.max.y + pad, bounds.max.z + pad);
  const cell = new THREE.Vector3().subVectors(max, min).divideScalar(resolution - 1);

  const closedness = analyzeMeshClosedness(geometry);
  const mode = forceMode || (closedness.closed ? 'closed' : 'open');
  if (mode !== 'closed' && mode !== 'open') {
    throw new Error(`未知隐式模式：${mode}`);
  }

  // 窄带溅射（OpenVDB / nTop 同款思路：只在表面附近算精确距离）。
  // 带宽需覆盖壳体厚度 + 洪泛符号修正所需的过渡带，远处用常量兜底。
  const cellMax = Math.max(cell.x, cell.y, cell.z);
  const bandCells =
    mode === 'open'
      ? Math.min(10, Math.max(4, Math.ceil((thickness + cellMax * 1.25) / cellMax) + 1))
      : 3;
  let splatResult;
  try {
    splatResult = await splatNarrowBandParallel(geometry, {
      resolution,
      min,
      cell,
      bandCells,
      onProgress,
      signal,
      useWorkers,
      collectNormals,
    });
  } catch (err) {
    console.warn('[basicCAD] Web Worker 溅射失败，回退内联：', err);
    splatResult = await splatNarrowBand(geometry, {
      resolution,
      min,
      cell,
      bandCells,
      onProgress,
      signal,
      collectNormals,
    });
  }
  const { distSq, sign, normals } = splatResult;
  const { dist } = buildDistanceField(distSq, sign, min.distanceTo(max));
  const data = finalizeSignedField(dist, sign, mode, thickness, resolution, cell);

  geometry.dispose();
  return {
    data,
    normals,
    resolution,
    min,
    max,
    cell,
    bounds: { min: min.clone(), max: max.clone() },
    mode,
    thickness,
    closedness,
    parallel: !!splatResult.parallel,
  };
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * 在 SDF 网格上做三线性插值采样，返回该点的有符号距离（内部为正）。
 * @param {object} sdf meshToSDF 的返回值
 * @param {number} x
 * @param {number} y
 * @param {number} z
 */
export function sampleSDFValue(sdf, x, y, z) {
  const { data, resolution, min, cell } = sdf;
  const last = resolution - 1;
  const gx = clamp((x - min.x) / cell.x, 0, last);
  const gy = clamp((y - min.y) / cell.y, 0, last);
  const gz = clamp((z - min.z) / cell.z, 0, last);
  const ix = Math.floor(gx);
  const iy = Math.floor(gy);
  const iz = Math.floor(gz);
  const fx = gx - ix;
  const fy = gy - iy;
  const fz = gz - iz;
  const ix1 = ix < last ? ix + 1 : ix;
  const iy1 = iy < last ? iy + 1 : iy;
  const iz1 = iz < last ? iz + 1 : iz;
  const at = (xi, yi, zi) => data[(xi * resolution + yi) * resolution + zi];
  const c000 = at(ix, iy, iz);
  const c100 = at(ix1, iy, iz);
  const c010 = at(ix, iy1, iz);
  const c110 = at(ix1, iy1, iz);
  const c001 = at(ix, iy, iz1);
  const c101 = at(ix1, iy, iz1);
  const c011 = at(ix, iy1, iz1);
  const c111 = at(ix1, iy1, iz1);
  const c00 = c000 + (c100 - c000) * fx;
  const c10 = c010 + (c110 - c010) * fx;
  const c01 = c001 + (c101 - c001) * fx;
  const c11 = c011 + (c111 - c011) * fx;
  const c0 = c00 + (c10 - c00) * fy;
  const c1 = c01 + (c11 - c01) * fy;
  return c0 + (c1 - c0) * fz;
}

/**
 * 用 Manifold.levelSet（Marching Tetrahedra）从 SDF 提取等值面网格。
 * @param {object} sdf meshToSDF 的返回值
 * @param {object} [options]
 * @param {number} [options.edgeLength=0.07] 输出网格的目标边长（越小越精细）
 * @param {number} [options.level=0] 等值面偏移（正=向内收缩）
 * @returns {Promise<THREE.BufferGeometry>}
 */
export async function sdfToThreeMesh(sdf, options = {}) {
  const { edgeLength = 0.07, level = 0 } = options;
  const mod = await getManifold();
  const bounds = {
    min: [sdf.min.x, sdf.min.y, sdf.min.z],
    max: [sdf.max.x, sdf.max.y, sdf.max.z],
  };
  const man = mod.Manifold.levelSet(
    (p) => sampleSDFValue(sdf, p[0], p[1], p[2]),
    bounds,
    edgeLength,
    level
  );
  const mesh = man.getMesh();
  man.delete();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    'position',
    new THREE.Float32BufferAttribute(mesh.vertProperties, mesh.numProp)
  );
  geometry.setIndex(new THREE.BufferAttribute(mesh.triVerts, 1));
  geometry.computeVertexNormals();
  return geometry;
}

/**
 * 丢弃三角网格中的极小连通分量（碎屑）。
 * 顶点先量化合并再并查集分组；只保留三角形数 >= max(minTriangles,
 * 总数×minRatio) 的分量。若没有需要过滤的，返回原几何引用。
 * @param {THREE.BufferGeometry} geometry
 * @returns {THREE.BufferGeometry}
 */
export function filterSmallComponents(geometry, options = {}) {
  const { minTriangles = 24, minRatio = 0.002 } = options;
  const index = geometry.index;
  const posAttr = geometry.attributes.position;
  if (!index || !posAttr) return geometry;
  const nTris = index.count / 3;
  if (nTris === 0) return geometry;
  const threshold = Math.max(minTriangles, Math.ceil(nTris * minRatio));
  if (nTris <= threshold) return geometry;

  const p = posAttr.array;
  const nVerts = posAttr.count;
  const box = new THREE.Box3().setFromBufferAttribute(posAttr);
  const maxDim =
    Math.max(box.max.x - box.min.x, box.max.y - box.min.y, box.max.z - box.min.z) || 1;
  const eps = maxDim * 2e-4;
  const vMap = new Map();
  const canonical = new Int32Array(nVerts);
  for (let i = 0; i < nVerts; i++) {
    const key = `${Math.round(p[i * 3] / eps)},${Math.round(p[i * 3 + 1] / eps)},${Math.round(p[i * 3 + 2] / eps)}`;
    let id = vMap.get(key);
    if (id === undefined) {
      id = vMap.size;
      vMap.set(key, id);
    }
    canonical[i] = id;
  }

  const parent = new Int32Array(vMap.size);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x) => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  const union = (a, b) => {
    const ra = find(a);
    const rb = find(b);
    if (ra !== rb) parent[rb] = ra;
  };
  for (let t = 0; t < nTris; t++) {
    const a = canonical[index.getX(t * 3)];
    const b = canonical[index.getX(t * 3 + 1)];
    const c = canonical[index.getX(t * 3 + 2)];
    union(a, b);
    union(b, c);
  }

  const triCount = new Int32Array(parent.length);
  for (let t = 0; t < nTris; t++) {
    triCount[find(canonical[index.getX(t * 3)])]++;
  }
  const keepTri = new Uint8Array(nTris);
  let kept = 0;
  for (let t = 0; t < nTris; t++) {
    if (triCount[find(canonical[index.getX(t * 3)])] >= threshold) {
      keepTri[t] = 1;
      kept++;
    }
  }
  if (kept === nTris) return geometry;

  // 重建：只保留保留三角形用到的顶点
  const vtxMap = new Map();
  const newPos = [];
  const newIdx = [];
  for (let t = 0; t < nTris; t++) {
    if (!keepTri[t]) continue;
    for (let k = 0; k < 3; k++) {
      const v = index.getX(t * 3 + k);
      let nv = vtxMap.get(v);
      if (nv === undefined) {
        nv = newPos.length / 3;
        vtxMap.set(v, nv);
        newPos.push(p[v * 3], p[v * 3 + 1], p[v * 3 + 2]);
      }
      newIdx.push(nv);
    }
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.Float32BufferAttribute(newPos, 3));
  out.setIndex(newIdx);
  out.computeVertexNormals();
  return out;
}

/**
 * 将两个 SDF 网格做布尔运算（需相同 resolution / bounds）。
 * 约定：内部为正，外部为负。
 * @param {'union'|'intersect'|'subtract'} op 并集 / 交集 / 差集（a 减 b）
 */
export function combineSDF(op, a, b, out = new Float32Array(a.data.length)) {
  if (a.data.length !== b.data.length) {
    throw new Error('两个 SDF 网格尺寸不一致，无法组合');
  }
  const A = a.data;
  const B = b.data;
  if (op === 'union') {
    for (let i = 0; i < A.length; i++) out[i] = Math.max(A[i], B[i]);
  } else if (op === 'intersect') {
    for (let i = 0; i < A.length; i++) out[i] = Math.min(A[i], B[i]);
  } else if (op === 'subtract') {
    for (let i = 0; i < A.length; i++) out[i] = Math.min(A[i], -B[i]);
  } else {
    throw new Error(`未知布尔操作：${op}`);
  }
  return out;
}

/**
 * Gyroid 三周期极小曲面（TPMS）隐式函数：
 *   sin(x)·cos(y) + sin(y)·cos(z) + sin(z)·cos(x) = 0
 * 返回带厚度的壳体 SDF（|f| - thickness 的近似），正值为壳体内。
 * 配合 combineSDF('intersect', modelSDF, latticeSDF) 即可把晶格
 * 嵌进模型内部（后续晶格化功能的基础）。
 * @param {[number,number,number]} p
 * @param {number} period 晶格周期（世界单位）
 * @param {number} thickness 壳厚（世界单位）
 */
export function gyroidSDF(p, period = 0.5, thickness = 0.06) {
  const k = (Math.PI * 2) / period;
  const x = p[0] * k;
  const y = p[1] * k;
  const z = p[2] * k;
  const f =
    Math.sin(x) * Math.cos(y) +
    Math.sin(y) * Math.cos(z) +
    Math.sin(z) * Math.cos(x);
  return Math.abs(f) - thickness;
}

/**
 * 把任意参数化隐式函数采样为与 sdf 相同网格的 SDF。
 * @param {object} sdf meshToSDF 的返回值（提供网格定义）
 * @param {(p:[number,number,number])=>number} fn 隐式函数（内部为正）
 */
export function sampleLatticeSDF(sdf, fn, out = new Float32Array(sdf.data.length)) {
  const { resolution, min, cell } = sdf;
  const p = [0, 0, 0];
  let i = 0;
  for (let ix = 0; ix < resolution; ix++) {
    p[0] = min.x + ix * cell.x;
    for (let iy = 0; iy < resolution; iy++) {
      p[1] = min.y + iy * cell.y;
      for (let iz = 0; iz < resolution; iz++) {
        p[2] = min.z + iz * cell.z;
        out[i++] = fn(p);
      }
    }
  }
  return out;
}
