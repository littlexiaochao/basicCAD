// =====================================================================
// 网格导出：把提取出的隐式表面网格写成 STL（二进制 / ASCII）或 OBJ。
// ---------------------------------------------------------------------
// 输入是 THREE.BufferGeometry（position + index + 可选 normal），
// 输出分别为 ArrayBuffer（二进制 STL）或字符串（OBJ / ASCII STL）。
// 坐标取原样（隐式网格已是世界坐标），单位不变。
// =====================================================================

/** 读取三角形顶点（支持索引 / 非索引），返回 [ax,ay,az, bx,by,bz, cx,cy,cz] */
function readTriangle(posAttr, index, t, out) {
  const read = (v, o) => {
    out[o] = posAttr.getX(v);
    out[o + 1] = posAttr.getY(v);
    out[o + 2] = posAttr.getZ(v);
  };
  if (index) {
    read(index.getX(t * 3), 0);
    read(index.getX(t * 3 + 1), 3);
    read(index.getX(t * 3 + 2), 6);
  } else {
    read(t * 3, 0);
    read(t * 3 + 1, 3);
    read(t * 3 + 2, 6);
  }
}

/** 三角形个数（索引优先，否则按非索引顶点数） */
export function geometryTriangleCount(geometry) {
  const posAttr = geometry.attributes.position;
  if (!posAttr) return 0;
  return geometry.index ? geometry.index.count / 3 : posAttr.count / 3;
}

/** 由三角形顶点计算单位法线（右手定则），结果写入 n 数组 */
function faceNormal(t, n) {
  const ax = t[0];
  const ay = t[1];
  const az = t[2];
  const abx = t[3] - ax;
  const aby = t[4] - ay;
  const abz = t[5] - az;
  const acx = t[6] - ax;
  const acy = t[7] - ay;
  const acz = t[8] - az;
  let nx = aby * acz - abz * acy;
  let ny = abz * acx - abx * acz;
  let nz = abx * acy - aby * acx;
  const len = Math.hypot(nx, ny, nz);
  if (len > 1e-12) {
    nx /= len;
    ny /= len;
    nz /= len;
  }
  n[0] = nx;
  n[1] = ny;
  n[2] = nz;
}

/**
 * 导出二进制 STL（紧凑、主流 CAD / 切片软件通用）。
 * @param {THREE.BufferGeometry} geometry
 * @param {object} [options]
 * @param {string} [options.name='basicCAD implicit surface'] 写入 80 字节头部的名称
 * @returns {ArrayBuffer}
 */
export function geometryToBinarySTL(geometry, options = {}) {
  const { name = 'basicCAD implicit surface' } = options;
  const posAttr = geometry.attributes.position;
  if (!posAttr) throw new Error('几何没有顶点数据，无法导出 STL');
  const triCount = geometryTriangleCount(geometry);
  if (triCount <= 0) throw new Error('几何没有三角形，无法导出 STL');

  const buf = new ArrayBuffer(84 + 50 * triCount);
  const dv = new DataView(buf);
  // 80 字节头部（ASCII 名称，不足补空格）
  const header = String(name).slice(0, 79);
  for (let i = 0; i < header.length; i++) dv.setUint8(i, header.charCodeAt(i));
  dv.setUint32(80, triCount, true);

  const tri = new Float64Array(9);
  const n = new Float64Array(3);
  let offset = 84;
  for (let t = 0; t < triCount; t++) {
    readTriangle(posAttr, geometry.index, t, tri);
    faceNormal(tri, n);
    dv.setFloat32(offset, n[0], true);
    dv.setFloat32(offset + 4, n[1], true);
    dv.setFloat32(offset + 8, n[2], true);
    for (let v = 0; v < 9; v++) dv.setFloat32(offset + 12 + v * 4, tri[v], true);
    dv.setUint16(offset + 48, 0, true); // 属性字节数
    offset += 50;
  }
  return buf;
}

/**
 * 导出 ASCII STL（可读、便于 diff / 调试，文件较大）。
 * @returns {string}
 */
export function geometryToASCIISTL(geometry, options = {}) {
  const { name = 'basicCAD implicit surface' } = options;
  const posAttr = geometry.attributes.position;
  if (!posAttr) throw new Error('几何没有顶点数据，无法导出 STL');
  const triCount = geometryTriangleCount(geometry);
  if (triCount <= 0) throw new Error('几何没有三角形，无法导出 STL');

  const lines = [`solid ${name}`];
  const tri = new Float64Array(9);
  const n = new Float64Array(3);
  for (let t = 0; t < triCount; t++) {
    readTriangle(posAttr, geometry.index, t, tri);
    faceNormal(tri, n);
    lines.push(`  facet normal ${n[0].toExponential(7)} ${n[1].toExponential(7)} ${n[2].toExponential(7)}`);
    lines.push('    outer loop');
    for (let v = 0; v < 3; v++) {
      lines.push(`      vertex ${tri[v * 3].toExponential(7)} ${tri[v * 3 + 1].toExponential(7)} ${tri[v * 3 + 2].toExponential(7)}`);
    }
    lines.push('    endloop');
    lines.push('  endfacet');
  }
  lines.push('endsolid ' + name);
  return lines.join('\n') + '\n';
}

/**
 * 导出 Wavefront OBJ（v / vn / f，顶点索引从 1 开始）。
 * 有顶点法线属性时一并写入，方便 CAD 软件平滑着色。
 * @param {THREE.BufferGeometry} geometry
 * @param {object} [options]
 * @param {string} [options.name='basicCAD implicit surface']
 * @returns {string}
 */
export function geometryToOBJ(geometry, options = {}) {
  const { name = 'basicCAD implicit surface' } = options;
  const posAttr = geometry.attributes.position;
  if (!posAttr) throw new Error('几何没有顶点数据，无法导出 OBJ');
  const nrmAttr = geometry.attributes.normal || null;
  const index = geometry.index;
  const vertCount = posAttr.count;
  const triCount = geometryTriangleCount(geometry);
  if (triCount <= 0) throw new Error('几何没有三角形，无法导出 OBJ');

  const lines = [
    `# ${name}`,
    `# vertices: ${vertCount}, triangles: ${triCount}`,
  ];
  for (let i = 0; i < vertCount; i++) {
    lines.push(`v ${posAttr.getX(i).toExponential(7)} ${posAttr.getY(i).toExponential(7)} ${posAttr.getZ(i).toExponential(7)}`);
  }
  if (nrmAttr) {
    for (let i = 0; i < vertCount; i++) {
      lines.push(`vn ${nrmAttr.getX(i).toExponential(7)} ${nrmAttr.getY(i).toExponential(7)} ${nrmAttr.getZ(i).toExponential(7)}`);
    }
  }
  const vref = (v) => (nrmAttr ? `${v}//${v}` : `${v}`);
  for (let t = 0; t < triCount; t++) {
    const a = index ? index.getX(t * 3) + 1 : t * 3 + 1;
    const b = index ? index.getX(t * 3 + 1) + 1 : t * 3 + 2;
    const c = index ? index.getX(t * 3 + 2) + 1 : t * 3 + 3;
    lines.push(`f ${vref(a)} ${vref(b)} ${vref(c)}`);
  }
  return lines.join('\n') + '\n';
}
