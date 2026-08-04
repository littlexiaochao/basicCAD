import * as THREE from 'three';

// =====================================================================
// 免网格体绘制：把 SDF 体数据上传为 3D 纹理，在 GPU 上用光线步进
// （ray marching）直接渲染等值面，不提取任何三角网格。
// ---------------------------------------------------------------------
// - 每像素从相机出发一条射线，与体包围盒求交后在着色器里逐步采样
//   SDF（三线性插值），找到符号翻转点后用二分法细化命中位置；
//   每个像素只渲染第一个命中的表面（被遮挡部分不渲染）。
// - 法线用 SDF 梯度（中心差分），平坦着色用屏幕空间导数（等价三角面法线）。
// - 支持与网格渲染一致的着色选项：Phong 光照 / 环境贴图反射（等距柱状
//   图采样 + 与场景相同的旋转）/ 高光 / 法线着色 / 反面单色。
// - 显示精度只改变步长（uniform），调节零重算。
// =====================================================================

const VERT = /* glsl */`
out vec3 vWorldPos;

void main() {
  vec4 wp = modelMatrix * vec4(position, 1.0);
  vWorldPos = wp.xyz;
  gl_Position = projectionMatrix * viewMatrix * wp;
}
`;

const FRAG = /* glsl */`
uniform sampler3D uSdfTex;
uniform vec3 uMin;
uniform vec3 uMax;
uniform float uStep;
uniform vec3 uBaseColor;
uniform float uSpecular;
uniform float uEnvStrength;
uniform bool uEnvEnabled;
uniform sampler2D uEnvMap;
uniform mat3 uEnvRotation;
uniform bool uNormalMode;
uniform bool uFlatMode;
uniform bool uBackfaceSolid;
uniform vec3 uBackfaceColor;
uniform mat4 uProjectionMatrix;
uniform mat4 uViewMatrix;

in vec3 vWorldPos;
out vec4 fragColor;

float sampleSdf(vec3 p) {
  vec3 uvw = (p - uMin) / (uMax - uMin);
  return texture(uSdfTex, uvw).r;
}

// three.js 同款等距柱状图 UV（用于环境贴图反射采样，方向与场景背景一致）
vec2 equirectUv(vec3 dir) {
  float u = atan(dir.z, dir.x) * 0.15915494309189535 + 0.5;
  float v = asin(clamp(dir.y, -1.0, 1.0)) * 0.3183098861837907 + 0.5;
  return vec2(u, v);
}

// 与 three.js 一致的 sRGB → 线性（EXR 环境贴图按 sRGB 色彩空间上传）
vec3 srgbToLinear(vec3 c) {
  return mix(
    pow(c * 0.9478672986 + 0.0521327014, vec3(2.4)),
    c * 0.0773993808,
    vec3(lessThanEqual(c, vec3(0.04045)))
  );
}

// 射线与体包围盒 [uMin, uMax] 的 slab 求交
bool slab(vec3 ro, vec3 rd, out float tEnter, out float tExit) {
  vec3 invR = 1.0 / rd;
  vec3 t0 = (uMin - ro) * invR;
  vec3 t1 = (uMax - ro) * invR;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  tEnter = max(max(tmin.x, tmin.y), tmin.z);
  tExit = min(min(tmax.x, tmax.y), tmax.z);
  return tExit >= 0.0 && tEnter <= tExit;
}

// 沿射线找等值面穿越：命中返回 true，tHit 为命中位置，
// entering = 从负（外）跨入正（内），即“正面”；反之为背面。
bool march(vec3 ro, vec3 rd, float tStart, float tEnd, out float tHit, out bool entering) {
  float t = tStart;
  float f = sampleSdf(ro + rd * t);
  bool neg = f < 0.0;
  bool hit = false;
  for (int i = 0; i < 1024; i++) {
    if (t >= tEnd) break;
    float tPrev = t;
    bool prevNeg = neg;
    t += uStep;
    if (t > tEnd) t = tEnd;
    f = sampleSdf(ro + rd * t);
    neg = f < 0.0;
    if (neg != prevNeg || f == 0.0) {
      float t0 = tPrev;
      float t1 = t;
      if (f != 0.0) {
        for (int b = 0; b < 6; b++) {
          float tm = (t0 + t1) * 0.5;
          if (sampleSdf(ro + rd * tm) < 0.0 == prevNeg) {
            t0 = tm;
          } else {
            t1 = tm;
          }
        }
      }
      tHit = (t0 + t1) * 0.5;
      entering = prevNeg && !neg;
      hit = true;
      break;
    }
  }
  return hit;
}

// SDF 梯度（中心差分），指向 SDF 增大方向（内部）
vec3 sdfGradient(vec3 p) {
  float h = uStep * 0.5;
  float gx = sampleSdf(p + vec3(h, 0.0, 0.0)) - sampleSdf(p - vec3(h, 0.0, 0.0));
  float gy = sampleSdf(p + vec3(0.0, h, 0.0)) - sampleSdf(p - vec3(0.0, h, 0.0));
  float gz = sampleSdf(p + vec3(0.0, 0.0, h)) - sampleSdf(p - vec3(0.0, 0.0, h));
  return vec3(gx, gy, gz);
}

// 与渲染器一致的 ACES 电影级色调映射（曝光 1.1）+ sRGB 输出
vec3 acesFilm(vec3 x) {
  return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}
vec3 linearToSrgb(vec3 c) {
  return mix(c * 12.92, pow(c, vec3(0.41666)) * 1.055 - 0.055, vec3(greaterThanEqual(c, vec3(0.0031308))));
}

void main() {
  vec3 ro = cameraPosition;
  vec3 rd = normalize(vWorldPos - cameraPosition);
  float tEnter;
  float tExit;
  if (!slab(ro, rd, tEnter, tExit)) discard;
  float tStart = max(tEnter, 0.0);

  float tHit;
  bool entering;
  if (!march(ro, rd, tStart, tExit, tHit, entering)) discard;
  vec3 hit = ro + rd * tHit;

  // 法线：默认 SDF 梯度（光顺着色），平坦着色用屏幕空间导数（面法线）
  vec3 n = sdfGradient(hit);
  if (length(n) < 1e-8) discard;
  n = normalize(n);
  if (uFlatMode) {
    vec3 fn = normalize(cross(dFdx(hit), dFdy(hit)));
    if (dot(fn, n) < 0.0) fn = -fn;
    n = fn;
  }
  if (dot(n, rd) > 0.0) n = -n; // 双面渲染：光照法线朝向相机

  vec3 color;
  if (uNormalMode) {
    // 法线着色：视空间法线映射为颜色（与 MeshNormalMaterial 一致）
    vec3 vn = mat3(viewMatrix) * n;
    color = vn * 0.5 + 0.5;
  } else {
    // Phong 光照，与场景灯光一致：半球光（白/暗蓝紫，0.6）+ 平行光（白，1.6）
    vec3 lightDir = normalize(vec3(5.0, 10.0, 7.0));
    vec3 hemiUp = vec3(1.0);
    vec3 hemiDown = vec3(0.2, 0.2, 0.267);
    float hemiMix = n.y * 0.5 + 0.5;
    vec3 ambient = mix(hemiDown, hemiUp, hemiMix) * 0.6;
    float diff = max(dot(n, lightDir), 0.0);
    vec3 h = normalize(lightDir - rd);
    float shininess = uSpecular * 1000.0;
    float spec = pow(max(dot(n, h), 0.0), shininess);
    color = uBaseColor * (ambient + vec3(1.6) * diff) + vec3(uSpecular) * spec * 1.6;

    // 环境贴图反射：uvEnvStrength = 0 无影响，1 = 镜子直接映射环境
    if (uEnvEnabled) {
      vec3 refl = reflect(-rd, n);
      vec3 dir = uEnvRotation * refl;
      vec3 env = srgbToLinear(texture(uEnvMap, equirectUv(dir)).rgb);
      color = mix(color, env, uEnvStrength);
    }

    // 反面单色：从背面穿越（相机在内部 / 开放表面的背侧）时用固定颜色
    if (uBackfaceSolid && !entering) {
      color = uBackfaceColor;
    }
  }

  // 半透明背面提示：封闭模型正面可见时，继续步进找到内部反面，
  // 以 10% 权重混入反面颜色（对应网格渲染透明度 0.9 的效果）
  if (uBackfaceSolid && !uNormalMode && entering) {
    float tHit2;
    bool entering2;
    if (march(ro, rd, tHit + uStep * 0.25, tExit, tHit2, entering2)) {
      color = mix(uBackfaceColor, color, 0.9);
    }
  }

  color = linearToSrgb(acesFilm(color * 1.1));
  fragColor = vec4(color, 1.0);
  // 写入真实命中深度，保证与其他几何的正确遮挡
  vec4 clip = uProjectionMatrix * uViewMatrix * vec4(hit, 1.0);
  gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
}
`;

/**
 * 把 SDF 数据（z 最快）重排为 three.js Data3DTexture 的 x 最快布局。
 * 三线性插值在 GPU 上完成，因此只传原始体数据。
 */
function createSdfTexture(sdf) {
  const N = sdf.resolution;
  const total = N * N * N;
  const src = sdf.data;
  const dst = new Float32Array(total);
  for (let x = 0; x < N; x++) {
    for (let y = 0; y < N; y++) {
      let s = (x * N + y) * N; // src: z 最快
      let d = y * N + x; // dst: x 最快（z 每步 +N²）
      for (let z = 0; z < N; z++) {
        dst[d] = src[s];
        s++;
        d += N * N;
      }
    }
  }
  const tex = new THREE.Data3DTexture(dst, N, N, N);
  tex.format = THREE.RedFormat;
  tex.type = THREE.FloatType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.wrapR = THREE.ClampToEdgeWrapping;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

/**
 * 创建免网格体绘制对象。
 * @param {object} sdf meshToSDF 的返回值
 * @param {object} [options]
 * @param {number} [options.precision=1] 显示精度（0.5~3，越小步长越细）
 * @returns {{ mesh: THREE.Mesh, setPrecision: (p:number)=>void,
 *            updateShading: (o:object)=>void, dispose: ()=>void }}
 */
export function createImplicitVolume(sdf, options = {}) {
  const { precision = 1 } = options;
  const cellMax = Math.max(sdf.cell.x, sdf.cell.y, sdf.cell.z);
  const size = new THREE.Vector3().subVectors(sdf.max, sdf.min);
  const center = new THREE.Vector3().addVectors(sdf.min, sdf.max).multiplyScalar(0.5);

  const texture = createSdfTexture(sdf);
  const uniforms = {
    uSdfTex: { value: texture },
    uMin: { value: sdf.min.clone() },
    uMax: { value: sdf.max.clone() },
    uStep: { value: cellMax },
    uBaseColor: { value: new THREE.Color(0xffffff) },
    uSpecular: { value: 0.15 },
    uEnvStrength: { value: 0.5 },
    uEnvEnabled: { value: false },
    uEnvMap: { value: null },
    uEnvRotation: { value: new THREE.Matrix3() },
    uNormalMode: { value: false },
    uFlatMode: { value: false },
    uBackfaceSolid: { value: true },
    uBackfaceColor: { value: new THREE.Color(0xd8d2aa) },
    uProjectionMatrix: { value: new THREE.Matrix4() },
    uViewMatrix: { value: new THREE.Matrix4() },
  };

  const material = new THREE.ShaderMaterial({
    vertexShader: VERT,
    fragmentShader: FRAG,
    uniforms,
    glslVersion: THREE.GLSL3,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: true,
  });

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
  mesh.position.copy(center);
  mesh.scale.copy(size);
  mesh.frustumCulled = false; // 体包围盒由 shader 内 slab 求交控制
  mesh.onBeforeRender = (renderer, scene, camera) => {
    uniforms.uProjectionMatrix.value.copy(camera.projectionMatrix);
    uniforms.uViewMatrix.value.copy(camera.matrixWorldInverse);
  };

  const volume = {
    mesh,
    /** 显示精度 → 步长（0.5~3 倍格距），只改 uniform，零重算 */
    setPrecision(p) {
      uniforms.uStep.value =
        cellMax * THREE.MathUtils.clamp(Number(p) || 1, 0.5, 3);
    },
    /**
     * 同步着色选项。
     * @param {object} o
     * @param {THREE.Color} [o.baseColor]
     * @param {number} [o.specular] 0~1
     * @param {number} [o.envStrength] 0~1
     * @param {boolean} [o.envEnabled]
     * @param {THREE.Texture|null} [o.envMap] 等距柱状环境贴图
     * @param {THREE.Matrix3} [o.envRotation]
     * @param {boolean} [o.normalMode]
     * @param {boolean} [o.flatMode]
     * @param {boolean} [o.backfaceSolid]
     * @param {THREE.Color} [o.backfaceColor]
     */
    updateShading(o = {}) {
      if (o.baseColor) uniforms.uBaseColor.value.copy(o.baseColor);
      if (o.specular !== undefined) uniforms.uSpecular.value = o.specular;
      if (o.envStrength !== undefined) uniforms.uEnvStrength.value = o.envStrength;
      if (o.envEnabled !== undefined) uniforms.uEnvEnabled.value = o.envEnabled;
      if (o.envMap !== undefined) uniforms.uEnvMap.value = o.envMap;
      if (o.envRotation) uniforms.uEnvRotation.value.copy(o.envRotation);
      if (o.normalMode !== undefined) uniforms.uNormalMode.value = o.normalMode;
      if (o.flatMode !== undefined) uniforms.uFlatMode.value = o.flatMode;
      if (o.backfaceSolid !== undefined) uniforms.uBackfaceSolid.value = o.backfaceSolid;
      if (o.backfaceColor) uniforms.uBackfaceColor.value.copy(o.backfaceColor);
    },
    dispose() {
      texture.dispose();
      mesh.geometry.dispose();
      material.dispose();
    },
  };
  volume.setPrecision(precision);
  return volume;
}
