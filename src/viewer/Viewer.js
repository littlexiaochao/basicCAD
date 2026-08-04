import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import WebGL from 'three/addons/capabilities/WebGL.js';
import { ViewCube } from './ViewCube.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
// OpenCASCADE 的 WASM 版本（用于解析 STEP / STP CAD 模型）
import occtWasmUrl from 'occt-import-js/dist/occt-import-js.wasm?url';
// Manifold 3D 的 WASM（用于 SDF 等值面提取与后续布尔运算）
import manifoldWasmUrl from 'manifold-3d/manifold.wasm?url';
import {
  setManifoldLocateFile,
  meshToSDF,
  sdfToThreeMesh,
  filterSmallComponents,
  analyzeRootClosedness,
  warmSplatWorkers,
} from '../sdf/implicit.js';
// Dual Contouring（Hermite 数据 + QEF）等值面提取：保留锐边、无需 WASM
import { sdfToDualContourMesh } from '../sdf/dualContour.js';
// 隐式表面网格导出（STL / OBJ）
import { geometryToBinarySTL, geometryToASCIISTL, geometryToOBJ } from '../sdf/meshExport.js';
// 免网格体绘制（GPU 光线步进直接渲染 SDF）
import { createImplicitVolume } from './implicitVolume.js';

// 告诉隐式表示模块去哪里加载 Manifold WASM（Vite 打包为资产 URL）
setManifoldLocateFile(() => manifoldWasmUrl);

/** occt-import-js 懒初始化（首次导入 STEP 时才加载 WASM） */
let _occtPromise = null;
async function getOcct() {
  if (!_occtPromise) {
    _occtPromise = (async () => {
      const mod = await import('occt-import-js');
      const occtimportjs = mod.default || mod;
      return occtimportjs({
        locateFile: (file) => (file.endsWith('.wasm') ? occtWasmUrl : file),
      });
    })().catch((err) => {
      _occtPromise = null;
      throw err;
    });
  }
  return _occtPromise;
}

/** 视角方向（相机看向目标点的方向向量） */
const VIEW_DIRECTIONS = {
  // CAD 坐标约定：Z 轴向上
  top: new THREE.Vector3(0, 0, 1),
  bottom: new THREE.Vector3(0, 0, -1),
  front: new THREE.Vector3(0, -1, 0),
  back: new THREE.Vector3(0, 1, 0),
  left: new THREE.Vector3(-1, 0, 0),
  right: new THREE.Vector3(1, 0, 0),
  'top-left': new THREE.Vector3(-1, 0, 1).normalize(),
  'top-right': new THREE.Vector3(1, 0, 1).normalize(),
  'bottom-left': new THREE.Vector3(-1, 0, -1).normalize(),
  'bottom-right': new THREE.Vector3(1, 0, -1).normalize(),
  'top-front': new THREE.Vector3(0, -1, 1).normalize(),
  'front-right': new THREE.Vector3(1, -1, 0).normalize(),
};

/** 环境贴图旋转：与 CAD 坐标一致（天空朝 +Z，和模型 Y→Z 旋转相同） */
const ENV_ROTATION = new THREE.Euler(THREE.MathUtils.degToRad(90), 0, 0);

/** 无环境背景时的默认场景底色 */
const DEFAULT_SCENE_BG = new THREE.Color(0x16181d);

/** 恒等矩阵（用于判断网格是否被操作轴变换过） */
const _IDENTITY_MATRIX = new THREE.Matrix4();

/**
 * 创建带彩色中心线的网格（CAD 坐标：Z 向上，格线在 XY 平面）：
 * - X 方向中心线（y = 0）为红色
 * - Y 方向中心线（x = 0）为绿色
 * - 其余格线为灰色
 */
function createColoredGrid(size = 12, divisions = 24) {
  const half = size / 2;
  const step = size / divisions;
  const gray = new THREE.Color(0x272d38);
  const red = new THREE.Color(0xff0000);
  const green = new THREE.Color(0x00ff00);
  const positions = [];
  const colors = [];

  // 平行于 X 轴的格线（y 恒定），y = 0 为 X 轴线（红色）
  for (let i = 0; i <= divisions; i++) {
    const y = -half + i * step;
    const color = Math.abs(y) < 1e-9 ? red : gray;
    positions.push(-half, y, 0, half, y, 0);
    colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
  }

  // 平行于 Y 轴的格线（x 恒定），x = 0 为 Y 轴线（绿色）
  for (let i = 0; i <= divisions; i++) {
    const x = -half + i * step;
    const color = Math.abs(x) < 1e-9 ? green : gray;
    positions.push(x, -half, 0, x, half, 0);
    colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({ vertexColors: true, toneMapped: false });
  const grid = new THREE.LineSegments(geometry, material);
  grid.name = 'basiccad-grid';
  return grid;
}

/** 生成户外环境贴图（天空渐变 + 太阳） */
function createOutdoorEnvironmentTexture(pmrem) {
  const w = 1024;
  const h = 512;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d');

  // 天空渐变
  const sky = ctx.createLinearGradient(0, 0, 0, h);
  sky.addColorStop(0, '#1d4fa8');
  sky.addColorStop(0.45, '#6fa8e8');
  sky.addColorStop(1, '#d8ecff');
  ctx.fillStyle = sky;
  ctx.fillRect(0, 0, w, h);

  // 太阳
  const sx = w * 0.76;
  const sy = h * 0.32;
  const sun = ctx.createRadialGradient(sx, sy, 0, sx, sy, 90);
  sun.addColorStop(0, 'rgba(255, 250, 225, 1)');
  sun.addColorStop(0.3, 'rgba(255, 236, 190, 0.85)');
  sun.addColorStop(1, 'rgba(255, 236, 190, 0)');
  ctx.fillStyle = sun;
  ctx.fillRect(sx - 90, sy - 90, 180, 180);

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return pmrem.fromEquirectangular(tex).texture;
}

/**
 * 将非索引几何焊接为索引几何（按位置合并顶点并平均法线），
 * 使顶点法线能在相邻三角形间线性插值，实现光顺着色。
 */
function weldGeometry(geometry) {
  if (geometry.index) return geometry;
  const pos = geometry.attributes.position;
  if (!pos) return geometry;
  const normal = geometry.attributes.normal;
  const EPS = 1e-4;
  const key = (x, y, z) => `${(x / EPS).toFixed(0)},${(y / EPS).toFixed(0)},${(z / EPS).toFixed(0)}`;
  const map = new Map();
  const positions = [];
  const indices = [];
  const normals = normal ? [] : null;
  const p = pos.array;
  const nr = normal ? normal.array : null;
  const n = pos.count;

  for (let i = 0; i < n; i++) {
    const x = p[i * 3];
    const y = p[i * 3 + 1];
    const z = p[i * 3 + 2];
    const k = key(x, y, z);
    let idx = map.get(k);
    if (idx === undefined) {
      idx = positions.length / 3;
      map.set(k, idx);
      positions.push(x, y, z);
      if (normals) normals.push(0, 0, 0);
    }
    indices.push(idx);
    if (normals && nr) {
      normals[idx * 3] += nr[i * 3];
      normals[idx * 3 + 1] += nr[i * 3 + 1];
      normals[idx * 3 + 2] += nr[i * 3 + 2];
    }
  }

  if (normals) {
    const vcount = positions.length / 3;
    for (let i = 0; i < vcount; i++) {
      const l = Math.hypot(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]) || 1;
      normals[i * 3] /= l;
      normals[i * 3 + 1] /= l;
      normals[i * 3 + 2] /= l;
    }
  }

  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setIndex(indices);
  if (normals) {
    g.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));
  } else {
    g.computeVertexNormals();
  }
  return g;
}

export class Viewer {
  /**
   * @param {HTMLElement} container 挂载 3D 视口的容器元素
   */
  constructor(container, options = {}) {
    this.container = container;
    /** 模型加载完成后的回调：onModelLoaded(info) */
    this.onModelLoaded = null;
    /** 模型被清除后的回调：onModelCleared() */
    this.onModelCleared = null;

    // ---------- 场景 ----------
    this.scene = new THREE.Scene();
    this.scene.background = DEFAULT_SCENE_BG.clone();

    // ---------- 相机 ----------
    this.camera = new THREE.PerspectiveCamera(50, this.aspect, 0.01, 10000);
    this.camera.position.set(3.2, 3.2, 3.2);
    // CAD 坐标约定：Z 轴向上（相机 up 对齐 Z），使 XY 平面格线在画面中平铺
    this.camera.up.set(0, 0, 1);
    this.perspCamera = this.camera;
    this.orthoCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 10000);
    this.orthoCamera.up.copy(this.camera.up); // 与透视相机保持一致的 Z-up 朝向
    this.projection = 'perspective';

    // ---------- 渲染器（含 WebGL2 检测与降级尝试） ----------
    if (!WebGL.isWebGL2Available()) {
      throw new Error('当前浏览器不支持 WebGL2，无法显示 3D 模型，请更新浏览器或开启硬件加速');
    }
    this.renderer = this._createRenderer();
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth || 1, container.clientHeight || 1);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.1;
    container.appendChild(this.renderer.domElement);

    // ---------- 环境与灯光 ----------
    this._pmrem = new THREE.PMREMGenerator(this.renderer);
    // 可用环境贴图列表（Blender 内置），随后异步加载 EXR 替换占位环境
    this._envDefs = [
      { id: 'studio', label: '工作室', url: 'env/studio.exr' },
      { id: 'forest', label: '户外森林', url: 'env/forest.exr' },
      { id: 'sunrise', label: '日出', url: 'env/sunrise.exr' },
      { id: 'sunset', label: '日落', url: 'env/sunset.exr' },
      { id: 'city', label: '城市', url: 'env/city.exr' },
      { id: 'courtyard', label: '庭院', url: 'env/courtyard.exr' },
      { id: 'night', label: '夜晚', url: 'env/night.exr' },
    ];
    const fallbackOutdoor = createOutdoorEnvironmentTexture(this._pmrem);
    this._envTextures = {};
    this._envBackgrounds = {};
    for (const def of this._envDefs) {
      this._envTextures[def.id] =
        def.id === 'studio'
          ? this._pmrem.fromScene(new RoomEnvironment(), 0.04).texture
          : fallbackOutdoor;
      this._envBackgrounds[def.id] = null;
    }
    this.envReady = false;
    /** 环境贴图全部加载完成后的回调：onEnvironmentsLoaded(list) */
    this.onEnvironmentsLoaded = null;
    this.scene.backgroundRotation = ENV_ROTATION.clone();
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 0.6));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(5, 10, 7);
    this.scene.add(keyLight);

    // ---------- 辅助元素：网格（中心线红/蓝标示 X/Z 轴方向，不另设坐标轴） ----------
    // 12×12 网格，每格 1 个单位
    this.grid = createColoredGrid(12, 12);
    this.scene.add(this.grid);

    // ---------- 轨道控制器：左键旋转 / 右键平移 / 滚轮缩放 ----------
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.05;
    this.controls.maxDistance = 500;
    this.controls.mouseButtons = {
      LEFT: THREE.MOUSE.ROTATE,
      MIDDLE: THREE.MOUSE.DOLLY,
      RIGHT: THREE.MOUSE.PAN,
    };
    this.controls.target.set(0, 0, 0);
    this.controls.update();

    // ---------- 视图立方体（左下角视角导航） ----------
    this.viewCube = null;
    try {
      this.viewCube = new ViewCube(
        options.viewCubeMount || container,
        (dir) => this.setViewDirection(dir),
        (dx, dy) => this.rotateViewByDrag(dx, dy)
      );
    } catch (err) {
      console.warn('视图立方体初始化失败：', err);
    }

    // ---------- 操作轴（TransformControls，Blender 风格） ----------
    this.transformControls = new TransformControls(this.camera, this.renderer.domElement);
    this.transformControls.space = 'world';
    this.transformControls.setSize(0.9);
    this.transformControls.enabled = false;
    this.transformControls.addEventListener('dragging-changed', (e) => {
      this.controls.enabled = !e.value;
      if (e.value) {
        // 拖拽开始：记录变换前状态
        this._dragStart = this._snapshotTransform();
      } else if (this._dragStart) {
        // 拖拽结束：若发生变换则压入撤销栈
        const cur = this._snapshotTransform();
        if (cur && !this._sameTransformState(this._dragStart, cur)) {
          this._pushUndo(this._dragStart);
        }
        this._dragStart = null;
      }
    });
    this.transformControls.addEventListener('objectChange', () => {
      this._updateModelBox();
    });
    this.transformControls.addEventListener('mouseDown', () => {
      const axis = this.transformControls.axis;
      if (axis) this._lastGizmoAxis = axis;
      if (this.onGizmoAxisPress) {
        this.onGizmoAxisPress(axis, this.transformControls.mode);
      }
    });
    this.scene.add(this.transformControls.getHelper());
    this.gizmoEnabled = false;
    this._lastGizmoAxis = null;
    this._dragStart = null;
    this._undoStack = [];
    this._redoStack = [];
    /** 点击操作轴手柄后的回调：onGizmoAxisPress(axis, mode) */
    this.onGizmoAxisPress = null;

    // 点击模型选中（显示操作轴）：指针按下/抬起位移小于阈值视为点击
    this._selectRaycaster = new THREE.Raycaster();
    this._pointerDown = null;
    this._onCanvasPointerDown = (e) => {
      this._pointerDown = { x: e.clientX, y: e.clientY };
    };
    this._onCanvasPointerUp = (e) => this._handleCanvasClick(e);
    this.renderer.domElement.addEventListener('pointerdown', this._onCanvasPointerDown);
    this.renderer.domElement.addEventListener('pointerup', this._onCanvasPointerUp);

    // ---------- 模型容器 ----------
    this.modelGroup = new THREE.Group();
    this.scene.add(this.modelGroup);
    this.modelName = '';
    this.modelBox = null;
    this.modelClosedness = null;

    // ---------- 隐式表示（SDF）容器 ----------
    this.implicitGroup = new THREE.Group();
    this.scene.add(this.implicitGroup);
    this.implicitGroup.visible = false;
    this.implicitModel = null;
    this._convertToken = 0;
    this._convertAbort = null;
    this._extractToken = 0;
    /**
     * 数据流：当前显示的数据阶段。
     * 'model'=原模型（导入网格）| 'implicit'=隐式表示（免网格体绘制）| 'mesh'=网格化结果
     */
    this.activeStage = 'model';
    this.hasImplicit = false;
    this.hasMesh = false;
    this._implicitVolume = null;
    /** 环境贴图反射方向旋转（与网格材质 envMapRotation 一致） */
    this._envRotationMatrix = new THREE.Matrix3().setFromMatrix4(
      new THREE.Matrix4().makeRotationFromEuler(ENV_ROTATION)
    );
    this._fallbackEnvTex = null;
    /** 网格化精度：提取密度倍数 = 该值 × SDF 格距（1~3，越小越精细；实时影响免网格预览步长，网格化时作为提取密度） */
    this.meshPrecision = 1;
    /** 等值面提取方式：'dc'=Dual Contouring（默认，保留锐边）/ 'manifold'=Manifold level set */
    this.extractMethod = 'dc';
    /** 隐式转换完成后的回调：onImplicitCreated(info) */
    this.onImplicitCreated = null;
    /** 网格化完成后的回调：onMeshCreated(info) */
    this.onMeshCreated = null;
    /** 数据流阶段切换回调：onDataStageChange(stage) */
    this.onDataStageChange = null;
    /** 着色叠加选项：edges / flat / normal / env（环境贴图着色）/ envType */
    this.shadingOptions = {
      edges: false,
      flat: false,
      normal: false,
      env: true,
      envType: 'studio',
      specular: 15,
      envStrength: 50,
      modelColor: '#ffffff',
      backfaceMode: 'solid',
      backfaceColor: '#d8d2aa',
    };
    // 环境背景默认不在世界中显示（可在着色模式下手动开启）
    this._envBgVisible = false;

    // ---------- 事件与动画 ----------
    this._onResize = this._onResize.bind(this);
    window.addEventListener('resize', this._onResize);
    this._tween = null;
    this._animate = this._animate.bind(this);
    this._animate();
    this._loadBlenderEnvs();
  }

  /** 异步加载 Blender 环境贴图（工作室 / 户外），加载完成后替换内置环境 */
  async _loadBlenderEnvs() {
    const loader = new EXRLoader();
    const base = import.meta.env.BASE_URL || './';
    const loaded = [];
    await Promise.all(
      this._envDefs.map(async (def) => {
        try {
          const tex = await loader.loadAsync(`${base}${def.url}`);
          tex.colorSpace = THREE.SRGBColorSpace;
          // 原始等距柱状图用作场景背景（环境盒包裹世界）
          tex.mapping = THREE.EquirectangularReflectionMapping;
          this._envBackgrounds[def.id] = tex;
          this._envTextures[def.id] = this._pmrem.fromEquirectangular(tex).texture;
          loaded.push({
            id: def.id,
            label: def.label,
            texture: this._envTextures[def.id],
            raw: tex,
          });
        } catch (err) {
          console.warn(`环境贴图加载失败：${def.url}`, err);
        }
      })
    );
    // 按定义顺序稳定排列
    const order = new Map(this._envDefs.map((def, i) => [def.id, i]));
    loaded.sort((a, b) => order.get(a.id) - order.get(b.id));
    this.envReady = true;
    // 若环境贴图着色处于开启状态，立即应用到模型
    this._applyEnvShading(this.shadingOptions.env);
    // 用当前环境贴图作为场景背景
    this.scene.background = this._envBgVisible
      ? this._envBackgrounds[this.shadingOptions.envType] || null
      : DEFAULT_SCENE_BG.clone();
    if (this.onEnvironmentsLoaded) this.onEnvironmentsLoaded(loaded);
  }

  /** 生成环境贴图材质球预览（返回 canvas）。使用独立的预览渲染器，需传原始等距柱状图 */
  createEnvPreview(rawTexture, size = 64) {
    if (!this._previewRenderer) {
      this._previewRenderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: true,
      });
      this._previewRenderer.outputColorSpace = THREE.SRGBColorSpace;
      this._previewRenderer.toneMapping = THREE.ACESFilmicToneMapping;
      this._previewPMREM = new THREE.PMREMGenerator(this._previewRenderer);
      this._previewScene = new THREE.Scene();
      this._previewCamera = new THREE.PerspectiveCamera(45, 1, 0.1, 10);
      this._previewCamera.position.set(0, 0, 2.6);
      this._previewMat = new THREE.MeshStandardMaterial({
        color: 0xffffff,
        metalness: 1,
        roughness: 0.12,
        envMapRotation: ENV_ROTATION.clone(),
      });
      this._previewSphere = new THREE.Mesh(
        new THREE.SphereGeometry(1, 48, 24),
        this._previewMat
      );
      this._previewScene.add(this._previewSphere);
    }
    this._previewRenderer.setPixelRatio(1);
    this._previewRenderer.setSize(size, size, false);
    // 用预览渲染器自己的 PMREM 转换原始等距柱状图，避免跨 WebGL 上下文贴图变黑
    this._previewMat.envMap = this._previewPMREM.fromEquirectangular(rawTexture).texture;
    this._previewMat.needsUpdate = true;
    this._previewRenderer.render(this._previewScene, this._previewCamera);
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    canvas.getContext('2d').drawImage(this._previewRenderer.domElement, 0, 0);
    return canvas;
  }

  // =================================================================
  // 尺寸与工具方法
  // =================================================================

  get aspect() {
    return this.container.clientWidth / Math.max(1, this.container.clientHeight);
  }

  /** 尝试创建 WebGL 渲染器（先开抗锯齿，失败则降级） */
  _createRenderer() {
    for (const options of [{ antialias: true }, { antialias: false }]) {
      try {
        return new THREE.WebGLRenderer(options);
      } catch (err) {
        // 继续尝试下一个配置
      }
    }
    throw new Error('无法创建 WebGL 渲染器，请更新浏览器或开启硬件加速');
  }

  _getModelBox() {
    if (this.modelGroup.children.length === 0) return null;
    const box = new THREE.Box3().setFromObject(this.modelGroup);
    if (box.isEmpty()) return null;
    return box;
  }

  /** 根据模型包围球计算一个合适的观察距离 */
  _viewDistance() {
    const box = this._getModelBox();
    if (!box) return 6;
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const dist = maxDim / (2 * Math.tan(THREE.MathUtils.degToRad(this.perspCamera.fov / 2)));
    return dist * 1.3;
  }

  // =================================================================
  // 动画循环与视图过渡
  // =================================================================

  _animate() {
    requestAnimationFrame(this._animate);
    if (this._tween) this._updateTween();
    this.controls.update();
    if (this.viewCube) {
      this.viewCube.sync(this.camera, this.controls.target);
      this.viewCube.render();
    }
    this.renderer.render(this.scene, this.camera);
  }

  _startTween(toPos, toTarget, duration = 450) {
    this._tween = {
      fromPos: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPos: toPos.clone(),
      toTarget: toTarget.clone(),
      start: performance.now(),
      duration,
    };
  }

  _updateTween() {
    const t = this._tween;
    let k = Math.min(1, (performance.now() - t.start) / t.duration);
    const ease = k < 0.5 ? 2 * k * k : 1 - Math.pow(-2 * k + 2, 2) / 2;
    this.camera.position.lerpVectors(t.fromPos, t.toPos, ease);
    this.controls.target.lerpVectors(t.fromTarget, t.toTarget, ease);
    if (k >= 1) this._tween = null;
  }

  /** 切换到指定视角（顶/底/前/后/左/右及顶左、顶正等组合视角） */
  setView(name) {
    const dir = VIEW_DIRECTIONS[name];
    if (dir) this.setViewDirection(dir);
  }

  /** 沿指定方向平滑切换视角 */
  setViewDirection(dir) {
    if (!dir) return;
    const target = this.controls.target.clone();
    const pos = dir.clone().multiplyScalar(this._viewDistance()).add(target);
    this._startTween(pos, target);
  }

  /** 重置为等轴测视角（模型居中） */
  resetView() {
    const box = this._getModelBox();
    const center = box ? box.getCenter(new THREE.Vector3()) : new THREE.Vector3();
    const pos = new THREE.Vector3(1, 1, 1)
      .normalize()
      .multiplyScalar(this._viewDistance())
      .add(center);
    this._startTween(pos, center);
  }

  /**
   * 切换投影模式
   * @param {'perspective'|'orthographic'} mode
   */
  setProjection(mode) {
    if (mode === this.projection) return;
    this.projection = mode;

    const pos = this.camera.position.clone();
    const target = this.controls.target.clone();
    const dist = pos.distanceTo(target);

    if (mode === 'orthographic') {
      // 根据当前透视视野计算正交视锥，保持视觉大小一致
      const h = 2 * Math.tan(THREE.MathUtils.degToRad(this.perspCamera.fov / 2)) * dist;
      const aspect = this.aspect;
      this.orthoCamera.left = (-h * aspect) / 2;
      this.orthoCamera.right = (h * aspect) / 2;
      this.orthoCamera.top = h / 2;
      this.orthoCamera.bottom = -h / 2;
      this.orthoCamera.updateProjectionMatrix();
      this.camera = this.orthoCamera;
    } else {
      this.camera = this.perspCamera;
      this.perspCamera.aspect = this.aspect;
      this.perspCamera.updateProjectionMatrix();
    }

    // 重新绑定控制器与操作轴相机并恢复位姿
    this.controls.object = this.camera;
    this.transformControls.camera = this.camera;
    this.camera.position.copy(pos);
    this.controls.target.copy(target);
    this.controls.update();
  }

  /** 开启 / 关闭操作轴 */
  setGizmoEnabled(enabled) {
    this.gizmoEnabled = enabled;
    this.transformControls.enabled = enabled;
    if (!enabled) this.transformControls.detach();
  }

  /** 切换操作轴模式：translate（移动）/ rotate（旋转） */
  setGizmoMode(mode) {
    const m = mode === 'rotate' ? 'rotate' : mode === 'scale' ? 'scale' : 'translate';
    this.transformControls.setMode(m);
  }

  /** 拖拽视图立方体旋转主视图（方向与左键拖拽旋转一致） */
  rotateViewByDrag(dx, dy) {
    const h = this.renderer.domElement.clientHeight || 1;
    // 临时关闭阻尼：程序化旋转一次到位，避免松开后残留旋转干扰后续吸附
    const damping = this.controls.enableDamping;
    this.controls.enableDamping = false;
    this.controls.rotateLeft((2 * Math.PI * dx) / h);
    this.controls.rotateUp((2 * Math.PI * dy) / h);
    this.controls.enableDamping = damping;
  }

  /** 显示 / 隐藏格线 */
  setGridVisible(visible) {
    this.grid.visible = visible;
  }

  /** 点击画布：命中模型则显示操作轴；空白处点击不改变操作轴状态 */
  _handleCanvasClick(e) {
    // 隐式表示（免网格体绘制）不可操作，不显示操作轴
    if (!this.gizmoEnabled || this.transformControls.dragging || this.activeStage === 'implicit') {
      this._pointerDown = null;
      return;
    }
    const start = this._pointerDown;
    this._pointerDown = null;
    if (!start || Math.hypot(e.clientX - start.x, e.clientY - start.y) > 6) return;

    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
    this._selectRaycaster.setFromCamera(ndc, this.camera);
    // 原模型 / 网格化结果都可挂操作轴（按当前数据流阶段选择目标）
    const target = this.activeStage === 'mesh' ? this.implicitGroup : this.modelGroup;
    const hits = this._selectRaycaster.intersectObjects(target.children, true);
    const root = target.children[0];
    if (hits.length && root) {
      this.transformControls.attach(root);
    }
  }

  /** 当前可操作的数据对象：网格化结果优先，否则原模型 */
  _activeTransformRoot() {
    if (this.activeStage === 'mesh' && this.implicitGroup.children.length) {
      return this.implicitGroup.children[0];
    }
    return this.modelGroup.children[0] || null;
  }

  /**
   * 应用数值变换（世界坐标）：
   * - 移动模式：沿当前轴平移 value 个单位
   * - 旋转模式：绕当前轴旋转 value 度
   */
  applyGizmoValue(value) {
    const obj = this.transformControls.object;
    const axis = this._lastGizmoAxis;
    if (!obj || !axis || !Number.isFinite(value)) return;
    const mode = this.transformControls.mode;
    const isUniformScale = mode === 'scale' && (axis === 'XYZ' || axis === 'XYZE');
    // 平移/旋转仅支持单轴；缩放支持单轴或中心 XYZ（等比例）
    if ((mode === 'translate' || mode === 'rotate' || !isUniformScale) && axis.length !== 1) return;
    const AXES = {
      X: new THREE.Vector3(1, 0, 0),
      Y: new THREE.Vector3(0, 1, 0),
      Z: new THREE.Vector3(0, 0, 1),
    };
    const vec = AXES[axis];
    if ((mode === 'translate' || mode === 'rotate') && !vec) return;
    // 无实际变化则不产生撤销记录
    if ((mode === 'translate' || mode === 'rotate') && value === 0) return;
    if (mode === 'scale' && value === 1) return;
    this._pushUndo(this._snapshotTransform());
    if (mode === 'translate') {
      obj.position[axis.toLowerCase()] += value;
    } else if (mode === 'rotate') {
      obj.rotateOnWorldAxis(vec, THREE.MathUtils.degToRad(value));
    } else if (mode === 'scale') {
      if (isUniformScale) {
        // 点击缩放中心：等比例缩放
        obj.scale.multiplyScalar(value);
      } else if (vec) {
        obj.scale[axis.toLowerCase()] *= value;
      }
    }
    this._updateModelBox();
  }

  /** 将模型包围盒中心移动到世界原点 (0,0,0) */
  centerModelToOrigin() {
    const root = this._activeTransformRoot();
    if (!root) return;
    const box = new THREE.Box3().setFromObject(root);
    if (box.isEmpty()) return;
    const center = box.getCenter(new THREE.Vector3());
    if (center.lengthSq() < 1e-12) return; // 已在原点，无需操作
    this._pushUndo(this._snapshotTransform());
    root.position.sub(center);
    this._updateModelBox();
  }

  // =================================================================
  // 撤销 / 重做（变换操作）
  // =================================================================

  /** 快照当前操作对象变换状态（操作轴附着对象 / 当前数据阶段对象） */
  _snapshotTransform() {
    const obj = this.transformControls.object || this._activeTransformRoot();
    if (!obj) return null;
    return {
      obj,
      position: obj.position.clone(),
      quaternion: obj.quaternion.clone(),
      scale: obj.scale.clone(),
    };
  }

  /** 判断两个变换状态是否相同 */
  _sameTransformState(a, b) {
    return (
      a.position.distanceToSquared(b.position) < 1e-10 &&
      a.quaternion.angleTo(b.quaternion) < 1e-10 &&
      a.scale.distanceToSquared(b.scale) < 1e-12
    );
  }

  /** 压入撤销栈（并清空重做栈），限制缓存长度 */
  _pushUndo(state) {
    if (!state) return;
    this._undoStack.push(state);
    if (this._undoStack.length > 100) this._undoStack.shift();
    this._redoStack.length = 0;
  }

  /** 恢复一个变换状态 */
  _applyState(state) {
    const obj = (state && state.obj) || this.modelGroup.children[0];
    if (!obj || !state) return;
    obj.position.copy(state.position);
    obj.quaternion.copy(state.quaternion);
    obj.scale.copy(state.scale);
    this._updateModelBox();
  }

  /** 撤销一步（Ctrl+Z） */
  undo() {
    if (!this._undoStack.length) return false;
    const cur = this._snapshotTransform();
    if (cur) this._redoStack.push(cur);
    this._applyState(this._undoStack.pop());
    return true;
  }

  /** 重做一步（Ctrl+Shift+Z / Ctrl+Y） */
  redo() {
    if (!this._redoStack.length) return false;
    const cur = this._snapshotTransform();
    if (cur) this._undoStack.push(cur);
    this._applyState(this._redoStack.pop());
    return true;
  }

  // =================================================================
  // 模型加载（OBJ / STL / STEP / STP）
  // =================================================================

  /**
   * 通过 File 对象加载模型。
   * OBJ 为 Y-up，旋转到 Z-up；STL / STEP 视为 CAD 的 Z-up 坐标，不旋转。
   */
  async loadFile(file) {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    let object;
    if (ext === 'obj') {
      const text = await file.text();
      object = new OBJLoader().parse(text);
      this._adoptObject(object, file.name, { format: 'obj', rotateToZUp: true });
    } else if (ext === 'stl' || ext === 'stla' || ext === 'stlb') {
      const buffer = await file.arrayBuffer();
      const geometry = new STLLoader().parse(buffer);
      object = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ color: 0xcccccc }));
      this._adoptObject(object, file.name, { format: 'stl' });
    } else if (ext === 'step' || ext === 'stp') {
      const occt = await getOcct();
      const buffer = new Uint8Array(await file.arrayBuffer());
      const result = occt.ReadStepFile(buffer, {
        linearUnit: 'millimeter',
        linearDeflectionType: 'bounding_box_ratio',
        linearDeflection: 0.001,
        angularDeflection: 0.5,
      });
      if (!result || !result.success) {
        throw new Error('STEP 文件解析失败，请确认是有效的 CAD 模型');
      }
      object = this._buildStepObject(result);
      this._adoptObject(object, file.name, { format: 'step', cadEdges: true });
    } else {
      throw new Error(`仅支持 OBJ / STL / STEP / STP 格式（当前文件：${ext || '未知'}）`);
    }
  }

  /**
   * 将 occt-import-js 的 STEP 解析结果转换为 three.js 网格组。
   * 每个网格附带由 CAD 原始边界提取的特征线（棱边）图层。
   */
  _buildStepObject(result) {
    const group = new THREE.Group();
    const edgeMaterial = new THREE.LineBasicMaterial({
      color: 0x111827,
      transparent: true,
      opacity: 0.85,
    });
    for (const rm of result.meshes) {
      if (!rm.attributes || !rm.attributes.position || !rm.index) continue;
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        'position',
        new THREE.Float32BufferAttribute(rm.attributes.position.array, 3)
      );
      if (rm.attributes.normal && rm.attributes.normal.array) {
        geometry.setAttribute(
          'normal',
          new THREE.Float32BufferAttribute(rm.attributes.normal.array, 3)
        );
      } else {
        geometry.computeVertexNormals();
      }
      geometry.setIndex(rm.index.array);

      const mesh = new THREE.Mesh(geometry, new THREE.MeshPhongMaterial({ color: 0xcccccc }));
      mesh.name = rm.name || 'STEP';
      // 原始 CAD 边界：基于三角网格的二面角提取特征棱边
      const edgeGeometry = this._computeFeatureEdges(geometry, 30);
      if (edgeGeometry) {
        const edges = new THREE.LineSegments(edgeGeometry, edgeMaterial);
        edges.name = 'basiccad-edges';
        mesh.add(edges);
      }
      group.add(mesh);
    }
    if (group.children.length === 0) {
      throw new Error('STEP 文件中没有可显示的几何体');
    }
    return group;
  }

  /**
   * 从三角网格提取 CAD 边界特征线：
   * 开放边界边始终保留；相邻两面夹角超过 angleDeg 的棱边保留（如立方体棱、圆柱端面圆）。
   * 位置按量化坐标合并，兼容面间重复顶点。
   */
  _computeFeatureEdges(geometry, angleDeg = 30) {
    const pos = geometry.attributes.position;
    const index = geometry.index;
    if (!pos || !index || index.count < 3) return null;
    const p = pos.array;
    const nVerts = pos.count;

    // 位置量化 -> 规范顶点 id
    const EPS = 1e-3;
    const vMap = new Map();
    const canonical = new Int32Array(nVerts);
    const firstVertex = new Map();
    for (let i = 0; i < nVerts; i++) {
      const key = `${Math.round(p[i * 3] / EPS)},${Math.round(p[i * 3 + 1] / EPS)},${Math.round(p[i * 3 + 2] / EPS)}`;
      let id = vMap.get(key);
      if (id === undefined) {
        id = vMap.size;
        vMap.set(key, id);
        firstVertex.set(id, i);
      }
      canonical[i] = id;
    }

    // 每个三角形的单位法线
    const triCount = index.count / 3;
    const triNormal = new Array(triCount);
    const triValid = new Uint8Array(triCount);
    const va = new THREE.Vector3();
    const vb = new THREE.Vector3();
    const vc = new THREE.Vector3();
    const e1 = new THREE.Vector3();
    const e2 = new THREE.Vector3();
    for (let t = 0; t < triCount; t++) {
      const ia = index.getX(t * 3);
      const ib = index.getX(t * 3 + 1);
      const ic = index.getX(t * 3 + 2);
      va.fromArray(p, ia * 3);
      vb.fromArray(p, ib * 3);
      vc.fromArray(p, ic * 3);
      const n = e1.subVectors(vb, va).cross(e2.subVectors(vc, va));
      const len = n.length();
      // 注意：n 复用临时向量，必须克隆后存入，否则后续迭代会覆盖已保存的法线
      triNormal[t] = len > 1e-12 ? n.clone().divideScalar(len) : null;
      triValid[t] = len > 1e-12 ? 1 : 0;
    }

    // 无向边邻接表
    const edgeMap = new Map();
    const edgeKey = (a, b) => {
      const ca = canonical[a];
      const cb = canonical[b];
      return ca < cb ? `${ca}:${cb}` : `${cb}:${ca}`;
    };
    const addEdge = (a, b, t) => {
      if (canonical[a] === canonical[b]) return; // 退化边
      const k = edgeKey(a, b);
      let e = edgeMap.get(k);
      if (!e) {
        e = { t1: -1, t2: -1 };
        edgeMap.set(k, e);
      }
      if (e.t1 === -1) e.t1 = t;
      else if (e.t2 === -1) e.t2 = t;
      else e.t2 = -2; // 非流形边（多于两个面），视为特征
    };
    for (let t = 0; t < triCount; t++) {
      if (!triValid[t]) continue; // 跳过退化三角形
      const ia = index.getX(t * 3);
      const ib = index.getX(t * 3 + 1);
      const ic = index.getX(t * 3 + 2);
      addEdge(ia, ib, t);
      addEdge(ib, ic, t);
      addEdge(ic, ia, t);
    }

    const thresholdCos = Math.cos(THREE.MathUtils.degToRad(angleDeg));
    const out = [];
    for (const [k, e] of edgeMap) {
      let draw = false;
      if (e.t2 === -2) {
        draw = true;
      } else if (e.t2 === -1) {
        draw = true; // 开放边界
      } else {
        const n1 = triNormal[e.t1];
        const n2 = triNormal[e.t2];
        draw = !n1 || !n2 || n1.dot(n2) < thresholdCos;
      }
      if (!draw) continue;
      const colon = k.indexOf(':');
      const ia = firstVertex.get(Number(k.slice(0, colon)));
      const ib = firstVertex.get(Number(k.slice(colon + 1)));
      out.push(p[ia * 3], p[ia * 3 + 1], p[ia * 3 + 2]);
      out.push(p[ib * 3], p[ib * 3 + 1], p[ib * 3 + 2]);
    }
    if (out.length === 0) return null;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
    return geo;
  }

  /**
   * 收纳模型：清空场景、统一归一化（缩放至最大边长 2 并居中）、适配视图。
   * @param {object} options { format, rotateToZUp, cadEdges }
   */
  _adoptObject(object, name, options = {}) {
    const { rotateToZUp = false, cadEdges = false } = options;
    if (this.transformControls.object) this.transformControls.detach();
    this._undoStack.length = 0;
    this._redoStack.length = 0;
    // 新模型载入前释放旧的隐式表示
    this._disposeImplicit();

    // CAD 坐标约定：OBJ 通常为 Y-up，绕 X 轴旋转 +90° 使旧 Y 变为新 Z（Z-up）；
    // STL / STEP 本身按 CAD 的 Z-up 约定，不旋转。
    if (rotateToZUp) object.rotation.x = Math.PI / 2;

    // 归一化：缩放到最大边长 2
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty()) {
      throw new Error('模型为空，无法加载');
    }
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const scale = 2 / maxDim;
    object.scale.multiplyScalar(scale);

    // 将包围盒中心偏移烘焙进几何体，使对象原点 = 包围盒中心
    // （这样操作轴始终位于模型包围盒中心，旋转也以中心为轴心）
    const box2 = new THREE.Box3().setFromObject(object);
    const center2 = box2.getCenter(new THREE.Vector3());
    const localOffset = center2
      .clone()
      .applyQuaternion(object.quaternion.clone().invert())
      .multiplyScalar(1 / scale)
      .negate();
    const translatedGeoms = new Set();
    object.traverse((o) => {
      if ((o.isMesh || o.isLineSegments) && o.geometry && !translatedGeoms.has(o.geometry)) {
        o.geometry.translate(localOffset.x, localOffset.y, localOffset.z);
        translatedGeoms.add(o.geometry);
      }
    });
    object.position.set(0, 0, 0);
    object.updateMatrixWorld(true);

    // 开启阴影与光照接收；保存 Phong 原材质并创建 Standard 变体（环境贴图着色用），
    // 并为每个网格叠加可切换的网格边图层
    if (cadEdges) this.shadingOptions.edges = true;
    object.traverse((o) => {
      if (o.isMesh) {
        o.castShadow = true;
        o.receiveShadow = true;
        // 保存原始几何（平坦着色用），并缓存焊接后的光顺几何
        o.userData.basiccadOriginalGeometry = o.geometry;
        o.userData.basiccadSmoothGeometry = null;
        // 保存原始 Phong 材质，并创建 Standard 变体（PBR 环境反射更可靠）
        if (!o.userData.basiccadPhongMaterial) {
          const mats = Array.isArray(o.material) ? o.material : [o.material];
          const stdMats = mats.map((m) => {
            if (!m || m.isMeshStandardMaterial) return m;
            return new THREE.MeshStandardMaterial({
              color: m.color ? m.color.clone() : 0xffffff,
              flatShading: m.flatShading || false,
              roughness: 0.6,
              metalness: 0,
            });
          });
          o.userData.basiccadPhongMaterial = Array.isArray(o.material) ? mats.slice() : mats[0];
          o.userData.basiccadStandardMaterial = Array.isArray(o.material) ? stdMats : stdMats[0];
        }
        // STEP 模型使用 CAD 原始边界特征线；其余格式使用三角网格的边
        let edges = o.getObjectByName('basiccad-edges');
        if (!edges) {
          edges = new THREE.LineSegments(
            new THREE.EdgesGeometry(o.geometry, 0),
            new THREE.LineBasicMaterial({
              color: 0x111827,
              transparent: true,
              opacity: 0.85,
            })
          );
          edges.name = 'basiccad-edges';
          o.add(edges);
        }
        edges.visible = this.shadingOptions.edges;
      }
    });

    this._disposeModel();
    this.modelGroup.clear();
    this.modelGroup.add(object);
    this.modelName = name;
    // 预热 Web Worker 池：首次隐式转换免去创建开销
    warmSplatWorkers();

    // 记录归一化后的实际尺寸（保持真实比例）
    this._updateModelBox();
    // 自动检测模型封闭性（水密性）：供信息面板显示，并决定隐式转换模式
    this.modelClosedness = analyzeRootClosedness(this.modelGroup);

    // 重置相机与视图
    this.controls.target.copy(this.modelBox.center);
    this.controls.update();
    this.resetView();
    this._applyEnvShading(this.shadingOptions.env);
    this._applyFlatShading(this.shadingOptions.flat);
    this._applyNormalShading(this.shadingOptions.normal);
    this._applyMaterialSettings();
    this._applyModelColor();
    this._applyBackfaceSettings();

    if (this.onModelLoaded) {
      this.onModelLoaded(this._buildInfo());
    }
  }

  /** 重新计算模型包围盒（加载或变换后调用） */
  _updateModelBox() {
    const box3 = new THREE.Box3().setFromObject(this.modelGroup);
    if (box3.isEmpty()) {
      this.modelBox = null;
      return;
    }
    this.modelBox = {
      size: box3.getSize(new THREE.Vector3()),
      center: box3.getCenter(new THREE.Vector3()),
    };
  }

  /**
   * 切换着色叠加选项
   * @param {'edges'|'flat'|'normal'|'env'} name
   * @param {boolean} enabled
   */
  setShadingOption(name, enabled) {
    this.shadingOptions[name] = enabled;
    if (name === 'flat') {
      this._applyFlatShading(enabled);
    } else if (name === 'edges') {
      for (const root of [this.modelGroup, this.implicitGroup]) {
        root.traverse((o) => {
          if (o.name === 'basiccad-edges') o.visible = enabled;
        });
      }
    } else if (name === 'normal') {
      this._applyNormalShading(enabled);
    } else if (name === 'env') {
      this._applyEnvShading(enabled);
    }
  }

  /** 设置镜面反射率（0=漫反射，100=镜面反射） */
  setSpecular(value) {
    this.shadingOptions.specular = value;
    this._applyMaterialSettings();
  }

  /** 设置环境贴图对模型的影响强度（0=无影响，100=金属全反射） */
  setEnvStrength(value) {
    this.shadingOptions.envStrength = value;
    this._applyMaterialSettings();
  }

  /**
   * 统一应用材质设置（镜面反射 + 环境反射）：
   * - 环境反射越高，漫反射颜色越黑、环境反射越强，100 时接近"镜子"直接映射环境景象
   */
  _applyMaterialSettings() {
    const sv = THREE.MathUtils.clamp(Number(this.shadingOptions.specular) || 0, 0, 100) / 100;
    const ev = THREE.MathUtils.clamp(Number(this.shadingOptions.envStrength) || 0, 0, 100) / 100;
    // Standard（环境贴图开启时）环境反射参与高光；Phong（默认）仅高光滑块生效
    const specGray = this.shadingOptions.env ? Math.max(sv, ev) : sv;
    this._modelMeshes((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (!m) continue;
        // 仅处理支持漫反射/环境反射的材质（跳过 MeshNormalMaterial 等）
        if (!('shininess' in m) && !m.isMeshStandardMaterial) continue;
        if ('shininess' in m) {
          // MeshPhongMaterial（默认着色）：高光滑块控制 shininess / 高光色
          m.shininess = sv * 1000;
          m.specular.setRGB(sv, sv, sv);
          m.envMapIntensity = ev * 2; // Phong 无环境贴图时无实际影响
        } else if (m.isMeshStandardMaterial) {
          // 环境反射超过 50 后才逐步变金属，100 → 全金属镜子
          m.metalness = Math.max(0, ev - 0.5) * 2;
          m.roughness = 1 - specGray * 0.9;
          m.envMapIntensity = ev * 2;
        }
        m.needsUpdate = true;
      }
    });
    this._syncRayMarchShading();
  }

  /** 切换环境贴图着色（开启后材质使用工作室环境反射） */
  _applyEnvShading(enabled) {
    const tex = enabled ? this._envTextures[this.shadingOptions.envType] : null;
    this._modelMeshes((o) => {
      if (o.material && o.material.isMeshNormalMaterial) return; // 法线着色时不切换
      // 环境贴图着色开启 → Standard 变体（PBR 环境反射）；关闭 → 默认 Phong 材质
      const std = o.userData.basiccadStandardMaterial;
      const phong = o.userData.basiccadPhongMaterial;
      o.material = enabled ? std || o.material : phong || o.material;
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (m && 'envMap' in m) {
          m.envMap = tex;
          if (m.envMap && 'envMapRotation' in m) {
            m.envMapRotation.copy(ENV_ROTATION);
          }
          m.needsUpdate = true;
        }
      }
    });
    // 切换材质后同步其他着色设置
    this._applyFlatShading(this.shadingOptions.flat);
    this._applyMaterialSettings();
    this._applyModelColor();
    this._syncRayMarchShading();
  }

  /** 切换环境贴图类型：studio（工作室）/ outdoor（户外） */
  setEnvType(type) {
    if (!this._envDefs.some((d) => d.id === type)) return;
    this.shadingOptions.envType = type;
    this._applyEnvShading(this.shadingOptions.env);
    this.scene.background = this._envBgVisible
      ? this._envBackgrounds[type] || null
      : DEFAULT_SCENE_BG.clone();
    this._syncRayMarchShading();
  }

  /** 显示 / 隐藏环境背景（不影响模型着色） */
  setEnvBackgroundVisible(visible) {
    this._envBgVisible = visible;
    this.scene.background = visible
      ? this._envBackgrounds[this.shadingOptions.envType] || null
      : DEFAULT_SCENE_BG.clone();
  }

  /** 设置模型固有色 */
  setModelColor(hex) {
    this.shadingOptions.modelColor = hex;
    this._applyModelColor();
  }

  /** 设置反面着色方式：solid（单色）/ front（正面着色） */
  setBackfaceMode(mode) {
    if (mode !== 'solid' && mode !== 'front') return;
    this.shadingOptions.backfaceMode = mode;
    this._applyBackfaceSettings();
  }

  /** 设置反面单色 */
  setBackfaceColor(hex) {
    this.shadingOptions.backfaceColor = hex;
    this._applyBackfaceSettings();
  }

  /**
   * 应用反面着色：双面渲染 + 背面颜色覆盖。
   * 单色模式：背面显示固定的浅灰黄；正面着色模式：背面与正面一样参与光照。
   */
  _applyBackfaceSettings() {
    const color = new THREE.Color(this.shadingOptions.backfaceColor || '#d8d2aa');
    const mix = this.shadingOptions.backfaceMode === 'solid' ? 1 : 0;
    this._modelMeshes((o) => {
      for (const variant of [o.userData.basiccadPhongMaterial, o.userData.basiccadStandardMaterial]) {
        const mats = Array.isArray(variant) ? variant : [variant];
        for (const m of mats) {
          if (!m) continue;
          if (!(m.isMeshStandardMaterial || 'shininess' in m)) continue;
          if (!m.userData.basiccadBackfaceReady) {
            m.userData.basiccadBackfaceReady = true;
            m.userData.basiccadBackfaceColor = { value: new THREE.Color() };
            m.userData.basiccadBackfaceMix = { value: 1 };
            m.side = THREE.DoubleSide;
            m.onBeforeCompile = (shader) => {
              shader.uniforms.basiccadBackfaceColor = m.userData.basiccadBackfaceColor;
              shader.uniforms.basiccadBackfaceMix = m.userData.basiccadBackfaceMix;
              // 手动声明自定义 uniform（否则着色器编译失败，three 回退旧程序）
              shader.fragmentShader =
                'uniform vec3 basiccadBackfaceColor;\nuniform float basiccadBackfaceMix;\n' +
                shader.fragmentShader;
              shader.fragmentShader = shader.fragmentShader.replace(
                '#include <opaque_fragment>',
                `if ( ! gl_FrontFacing && basiccadBackfaceMix > 0.5 ) { outgoingLight = basiccadBackfaceColor; }
                 #include <opaque_fragment>`
              );
            };
            m.needsUpdate = true;
          }
          m.userData.basiccadBackfaceColor.value.copy(color);
          m.userData.basiccadBackfaceMix.value = mix;
        }
      }
    });
    this._syncRayMarchShading();
  }

  /** 应用模型固有色到材质 */
  _applyModelColor() {
    const color = this.shadingOptions.modelColor
      ? new THREE.Color(this.shadingOptions.modelColor)
      : new THREE.Color(0xffffff);
    this._modelMeshes((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (m && 'color' in m) {
          m.color.copy(color);
          m.needsUpdate = true;
        }
      }
    });
    this._syncRayMarchShading();
  }

  /** 切换材质平面光照着色（flat shading） */
  _applyFlatShading(enabled) {
    this._modelMeshes((o) => {
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (m && 'flatShading' in m) {
          m.flatShading = enabled;
          m.needsUpdate = true;
        }
      }
      // 平坦着色用原始（非索引）几何；光顺着色用焊接后的索引几何（顶点法线插值）
      const original = o.userData.basiccadOriginalGeometry;
      if (original) {
        if (enabled) {
          if (o.geometry !== original) o.geometry = original;
        } else {
          if (!o.userData.basiccadSmoothGeometry) {
            o.userData.basiccadSmoothGeometry = weldGeometry(original);
          }
          if (o.geometry !== o.userData.basiccadSmoothGeometry) {
            o.geometry = o.userData.basiccadSmoothGeometry;
          }
        }
      }
    });
    // 刷新包围盒与模型统计（隐式表面显示期间不重复触发加载回调）
    if (this.modelGroup.children.length && this.activeStage === 'model') {
      this._updateModelBox();
      if (this.onModelLoaded) this.onModelLoaded(this._buildInfo());
    }
    this._syncRayMarchShading();
  }

  /** 切换法线着色（用 MeshNormalMaterial 将法线方向映射为颜色） */
  _applyNormalShading(enabled) {
    this._modelMeshes((o) => {
      if (enabled) {
        // 记住原始材质，便于关闭后恢复
        if (!o.userData.basiccadOriginalMaterial) {
          o.userData.basiccadOriginalMaterial = o.material;
        }
        o.material = new THREE.MeshNormalMaterial({
          flatShading: this.shadingOptions.flat,
        });
      } else if (o.userData.basiccadOriginalMaterial) {
        o.material = o.userData.basiccadOriginalMaterial;
        o.userData.basiccadOriginalMaterial = null;
      }
    });
    // 同步平面着色状态（MeshNormalMaterial 也支持 flatShading）
    this._applyFlatShading(this.shadingOptions.flat);
    // 恢复原始材质后重新应用环境贴图状态
    this._applyEnvShading(this.shadingOptions.env);
    // 恢复原始材质后重新应用材质设置（镜面 + 环境反射）
    this._applyMaterialSettings();
    // 恢复原始材质后重新应用固有色
    this._applyModelColor();
    this._syncRayMarchShading();
  }

  /** 清除当前模型并复位视图 */
  clearModel() {
    if (this.transformControls.object) this.transformControls.detach();
    this._undoStack.length = 0;
    this._redoStack.length = 0;
    this._disposeImplicit();
    this._disposeModel();
    this.modelGroup.clear();
    this.modelName = '';
    this.modelBox = null;
    this.modelClosedness = null;
    this.shadingOptions = {
      edges: false,
      flat: false,
      normal: false,
      env: true,
      envType: 'studio',
      specular: 15,
      envStrength: 50,
      modelColor: '#ffffff',
      backfaceMode: 'solid',
      backfaceColor: '#d8d2aa',
    };
    this.controls.target.set(0, 0, 0);
    this.controls.update();
    this.resetView();
    if (this.onModelCleared) this.onModelCleared();
  }

  /** 释放当前模型的几何与材质资源 */
  _disposeModel() {
    this.modelGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.userData.basiccadOriginalGeometry && o.userData.basiccadOriginalGeometry !== o.geometry) {
        o.userData.basiccadOriginalGeometry.dispose();
      }
      if (o.userData.basiccadSmoothGeometry && o.userData.basiccadSmoothGeometry !== o.geometry) {
        o.userData.basiccadSmoothGeometry.dispose();
      }
      const materials = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of materials) {
        if (m) m.dispose();
      }
      // 释放 Phong / Standard 变体（避免重复释放当前材质）
      for (const variant of [o.userData.basiccadPhongMaterial, o.userData.basiccadStandardMaterial]) {
        const arr = Array.isArray(variant) ? variant : [variant];
        for (const m of arr) {
          if (m && m !== o.material && !materials.includes(m)) m.dispose();
        }
      }
    });
  }

  /** 构建模型统计信息（顶点数 / 网格面数） */
  _buildInfo() {
    let vertices = 0;
    let triangles = 0;
    this.modelGroup.traverse((o) => {
      if (o.isMesh && o.geometry) {
        const pos = o.geometry.attributes.position;
        if (pos) vertices += pos.count;
        if (o.geometry.index) triangles += o.geometry.index.count / 3;
        else if (pos) triangles += pos.count / 3;
      }
    });
    return {
      name: this.modelName,
      vertices,
      triangles,
      closedness: this.modelClosedness,
    };
  }

  // =================================================================
  // 隐式表示（SDF）
  // =================================================================

  /**
   * 遍历当前应显示的模型网格（原始模型 + 隐式表面，按可见性切换）。
   * 用于让着色选项同时作用于隐式表面。
   */
  _modelMeshes(callback) {
    const roots = [this.modelGroup];
    // 网格化结果无论是否当前显示，着色选项都同步作用（切回时即正确）
    if (this.implicitGroup && this.implicitGroup.children.length) roots.push(this.implicitGroup);
    for (const root of roots) {
      root.traverse((o) => {
        if (o.isMesh) callback(o);
      });
    }
  }

  /**
   * 将当前模型转换为隐式表示（规则网格 SDF），并显示提取出的等值面。
   * @param {object} [options]
   * @param {number} [options.resolution=64] SDF 网格分辨率（N³）
   * @param {number} [options.thickness=0.04] 偏移距离（开放模型，壳体厚度 = 2 × 偏移距离）
   * @param {number} [options.precision=1] 网格化精度：等值面边长 = 该值 × SDF 格距（1~3）
   * @param {'auto'|boolean} [options.useWorkers] Web Worker 并行溅射（默认 auto）
   * @param {(progress:number, phase:string)=>void} [options.onProgress]
   * @returns {Promise<object|null>} { resolution, bytes, vertices, triangles, mode, thickness, parallel }
   * 转换后直接进入免网格体绘制显示（不提取网格）；「网格化」或导出时才提取。
   */
  async convertToImplicit(options = {}) {
    const {
      resolution = 64,
      thickness = 0.04,
      precision = 1,
      useWorkers = 'auto',
      extractMethod = 'dc',
      onProgress = null,
    } = options;
    this.meshPrecision = THREE.MathUtils.clamp(Number(precision) || 1, 1, 3);
    this.extractMethod = extractMethod === 'manifold' ? 'manifold' : 'dc';
    if (!this.modelGroup.children.length) {
      throw new Error('请先导入模型，再转换为隐式表示');
    }
    if (this.transformControls.object) this.transformControls.detach();

    this._disposeImplicit();
    const token = ++this._convertToken;
    const controller = new AbortController();
    this._convertAbort = controller;

    try {
      if (typeof onProgress === 'function') onProgress(0, 'sampling');
      const sdf = await meshToSDF(this.modelGroup, {
        resolution,
        thickness,
        useWorkers,
        // 始终保存最近面法线（Hermite 数据）：提取算法只影响「网格化」，
        // 与隐式化解耦——切到 DC 网格化时无需重新采样
        collectNormals: true,
        onProgress: (p) => {
          if (typeof onProgress === 'function') onProgress(p, 'sampling');
        },
        signal: controller.signal,
      });
      if (token !== this._convertToken) return null;

      this.implicitModel = sdf;
      this.hasImplicit = true;
      this.hasMesh = false;
      this.implicitModel.vertices = 0;
      this.implicitModel.triangles = 0;
      // 免网格体绘制显示（不提取网格，直到「网格化」/导出时才提取）
      this._ensureImplicitVolume();
      this._syncRayMarchShading();
      this.activeStage = 'implicit';
      this._applyStageVisibility();
      if (this.onDataStageChange) this.onDataStageChange('implicit');

      const info = {
        resolution,
        bytes: sdf.data.byteLength,
        vertices: this.implicitModel.vertices || 0,
        triangles: this.implicitModel.triangles || 0,
        mode: sdf.mode,
        thickness: sdf.thickness,
      };
      if (this.onImplicitCreated) this.onImplicitCreated(info);
      return info;
    } finally {
      this._convertAbort = null;
    }
  }

  /**
   * 数据流：切换到指定数据阶段显示。
   * @param {'model'|'implicit'|'mesh'} stage
   */
  setActiveStage(stage) {
    const valid =
      stage === 'model' ||
      (stage === 'implicit' && this.hasImplicit) ||
      (stage === 'mesh' && this.hasMesh);
    if (!valid) return;
    this.activeStage = stage;
    // 切换阶段时解除操作轴附着（避免作用于已隐藏的数据）
    if (this.transformControls.object) this.transformControls.detach();
    this._applyStageVisibility();
    if (this.onDataStageChange) this.onDataStageChange(stage);
  }

  /** 按当前数据流阶段更新各容器的可见性 */
  _applyStageVisibility() {
    this.modelGroup.visible = this.activeStage === 'model';
    this.implicitGroup.visible = this.activeStage === 'mesh';
    if (this._implicitVolume) {
      this._implicitVolume.mesh.visible = this.activeStage === 'implicit';
    }
  }

  /**
   * 网格化：把隐式表示提取为三角网格并显示（数据流进入「网格化」阶段）。
   * @param {object} [options]
   * @param {number} [options.precision=this.meshPrecision] 网格化精度（1~3）
   * @param {(progress:number)=>void} [options.onProgress]
   * @returns {Promise<object|null>} { resolution, bytes, vertices, triangles, mode, thickness }
   */
  async meshifyImplicit(options = {}) {
    const { precision = this.meshPrecision, onProgress = null } = options;
    if (!this.implicitModel) {
      throw new Error('请先转换为隐式表示，再网格化');
    }
    this.meshPrecision = THREE.MathUtils.clamp(Number(precision) || 1, 1, 3);
    if (this._implicitVolume) this._implicitVolume.setPrecision(this.meshPrecision);
    const token = ++this._extractToken;
    if (typeof onProgress === 'function') onProgress(0);
    try {
      const geometry = await this._extractImplicitGeometry(
        this.implicitModel,
        this.meshPrecision
      );
      if (token !== this._extractToken) {
        geometry.dispose();
        return null;
      }
      this._installImplicitMesh(geometry);
      this.hasMesh = true;
      this.activeStage = 'mesh';
      this._applyStageVisibility();
      if (this.onDataStageChange) this.onDataStageChange('mesh');
      const info = {
        resolution: this.implicitModel.resolution,
        bytes: this.implicitModel.data.byteLength,
        vertices: this.implicitModel.vertices || 0,
        triangles: this.implicitModel.triangles || 0,
        mode: this.implicitModel.mode,
        thickness: this.implicitModel.thickness,
      };
      if (this.onMeshCreated) this.onMeshCreated(info);
      return info;
    } catch (err) {
      console.warn('[basicCAD] 网格化失败：', err);
      throw err;
    }
  }

  /** 按当前精度 / 提取算法从 SDF 提取等值面几何（含碎屑过滤） */
  async _extractImplicitGeometry(sdf, precision) {
    let geometry;
    if (this.extractMethod === 'dc') {
      // DC：显示精度 → 粗网格步长（0.5~1.4→1 最精细，1.5~2.4→2，2.5~3→3）
      const step = THREE.MathUtils.clamp(Math.round(precision), 1, 3);
      geometry = await sdfToDualContourMesh(sdf, { step });
    } else {
      // Manifold：等值面目标边长 = 精度值 × SDF 格距（越小越精细）
      const cellMax = Math.max(sdf.cell.x, sdf.cell.y, sdf.cell.z);
      geometry = await sdfToThreeMesh(sdf, { edgeLength: precision * cellMax });
    }
    // 碎屑过滤：丢弃等值面中极小且孤立的连通分量（点状 / 杆状杂点）
    return filterSmallComponents(geometry, { minTriangles: 24, minRatio: 0.002 });
  }

  /** 把提取出的几何安装为半透明双面网格，并同步统计与着色 */
  _installImplicitMesh(geometry) {
    // 清理旧的隐式网格
    this.implicitGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) if (m) m.dispose();
    });
    this.implicitGroup.clear();

    const material = new THREE.MeshStandardMaterial({
      color: 0x6ea8ff,
      metalness: 0.15,
      roughness: 0.32,
      transparent: true,
      opacity: 0.9,
      // 闭合与开放统一半透明双面渲染，表现一致
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = 'basiccad-implicit-surface';
    // 网格边叠加层：与原始模型的「显示网格边」一致（三角网格边，可开关）
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geometry, 0),
      new THREE.LineBasicMaterial({
        color: 0x111827,
        transparent: true,
        opacity: 0.85,
      })
    );
    edges.name = 'basiccad-edges';
    edges.visible = this.shadingOptions.edges;
    mesh.add(edges);
    this.implicitGroup.add(mesh);
    this.implicitModel.triangles = geometry.index
      ? geometry.index.count / 3
      : geometry.attributes.position.count / 3;
    this.implicitModel.vertices = geometry.attributes.position.count;
    // 让当前着色选项同步作用于隐式表面
    this._applyEnvShading(this.shadingOptions.env);
    this._applyMaterialSettings();
    this._applyModelColor();
    this._applyFlatShading(this.shadingOptions.flat);
    this._applyNormalShading(this.shadingOptions.normal);
  }

  /** 确保免网格体绘制对象存在（从当前 SDF 上传体纹理） */
  _ensureImplicitVolume() {
    if (this._implicitVolume || !this.implicitModel) return;
    const volume = createImplicitVolume(this.implicitModel, {
      precision: this.meshPrecision,
    });
    this._implicitVolume = volume;
    this.scene.add(volume.mesh);
  }

  /** 释放免网格体绘制资源 */
  _disposeImplicitVolume() {
    if (!this._implicitVolume) return;
    this.scene.remove(this._implicitVolume.mesh);
    this._implicitVolume.dispose();
    this._implicitVolume = null;
  }

  /** 生成 1×1 白色等距柱状图，作为环境贴图未加载完成时的兜底 */
  _fallbackEnvTexture() {
    if (!this._fallbackEnvTex) {
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#c8ccd2';
      ctx.fillRect(0, 0, 1, 1);
      this._fallbackEnvTex = new THREE.CanvasTexture(canvas);
      this._fallbackEnvTex.mapping = THREE.EquirectangularReflectionMapping;
    }
    return this._fallbackEnvTex;
  }

  /** 把当前着色选项同步到免网格体绘制材质 */
  _syncRayMarchShading() {
    const volume = this._implicitVolume;
    if (!volume) return;
    const s = this.shadingOptions;
    volume.updateShading({
      baseColor: new THREE.Color(s.modelColor || '#ffffff'),
      specular: THREE.MathUtils.clamp(Number(s.specular) || 0, 0, 100) / 100,
      envStrength: THREE.MathUtils.clamp(Number(s.envStrength) || 0, 0, 100) / 100,
      envEnabled: !!s.env,
      envMap: this._envBackgrounds[s.envType] || this._fallbackEnvTexture(),
      envRotation: this._envRotationMatrix,
      normalMode: !!s.normal,
      flatMode: !!s.flat,
      backfaceSolid: s.backfaceMode !== 'front',
      backfaceColor: new THREE.Color(s.backfaceColor || '#d8d2aa'),
    });
  }

  /**
   * 导出当前隐式表面网格（所见即所得：跟随当前显示精度 / 提取算法）。
   * 免网格预览模式下还没有网格，导出时按当前精度补提取一次。
   * @param {'stl'|'obj'|'stl-ascii'} [format='stl'] 导出格式
   * @returns {Promise<{ blob: Blob, name: string, vertices: number, triangles: number }>}
   */
  async exportImplicitSurface(format = 'stl') {
    if (!this.implicitModel) {
      throw new Error('当前没有隐式表面可导出，请先转换为隐式表示');
    }
    if (!this.hasMesh) {
      // 导出时自动网格化（数据流进入「网格化」阶段）
      await this.meshifyImplicit({ precision: this.meshPrecision });
    }
    const mesh = this.implicitGroup.children[0];
    if (!mesh || !mesh.geometry) throw new Error('网格化失败，无法导出');
    // 若网格被操作轴移动/旋转/缩放，导出时把变换烘焙进顶点（所见即所得）
    mesh.updateWorldMatrix(true, false);
    let geometry = mesh.geometry;
    if (mesh.matrixWorld && !mesh.matrixWorld.equals(_IDENTITY_MATRIX)) {
      geometry = geometry.clone();
      geometry.applyMatrix4(mesh.matrixWorld);
      if (geometry.attributes.normal) geometry.computeVertexNormals();
    }
    const posAttr = geometry.attributes.position;
    const triangles = geometry.index
      ? geometry.index.count / 3
      : posAttr
        ? posAttr.count / 3
        : 0;
    const baseName = (this.modelName || 'implicit').replace(/\.[^.]+$/, '') || 'implicit';

    let blob;
    let name;
    if (format === 'obj') {
      blob = new Blob([geometryToOBJ(geometry, { name: 'basicCAD implicit surface' })], {
        type: 'text/plain;charset=utf-8',
      });
      name = `${baseName}-implicit.obj`;
    } else if (format === 'stl-ascii') {
      blob = new Blob([geometryToASCIISTL(geometry, { name: 'basicCAD implicit surface' })], {
        type: 'text/plain;charset=utf-8',
      });
      name = `${baseName}-implicit.stl`;
    } else {
      blob = new Blob([geometryToBinarySTL(geometry, { name: 'basicCAD implicit surface' })], {
        type: 'model/stl',
      });
      name = `${baseName}-implicit.stl`;
    }
    return { blob, name, vertices: posAttr ? posAttr.count : 0, triangles };
  }

  /**
   * 调节网格化精度：等值面提取边长 = 精度值 × SDF 格距（1~3，越小越精细）。
   * 免网格显示阶段只改光线步进步长（uniform，零重算）；「网格化」时按该精度提取。
   */
  setMeshPrecision(precision) {
    const p = THREE.MathUtils.clamp(Number(precision) || 1, 1, 3);
    this.meshPrecision = p;
    // 免网格显示：只改步长，零重算
    if (this._implicitVolume) {
      this._implicitVolume.setPrecision(p);
    }
  }

  /** 切换等值面提取方式：'dc'（Dual Contouring，保留锐边）或 'manifold'（Manifold level set） */
  async setExtractMethod(method) {
    const next = method === 'manifold' ? 'manifold' : 'dc';
    if (next === this.extractMethod) return;
    this.extractMethod = next;
    // 已有网格化结果时立即用新算法重新网格化（SDF 无需重算）
    if (this.hasMesh) {
      await this.meshifyImplicit({ precision: this.meshPrecision });
    }
  }

  /** 释放隐式表面资源，取消进行中的转换，并恢复原始网格显示 */
  _disposeImplicit() {
    this._convertToken++; // 使进行中的转换失效
    this._extractToken++; // 使进行中的重新提取失效
    this._disposeImplicitVolume();
    if (this._convertAbort) {
      try {
        this._convertAbort.abort();
      } catch {
        /* noop */
      }
      this._convertAbort = null;
    }
    this.implicitGroup.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      const mats = Array.isArray(o.material) ? o.material : [o.material];
      for (const m of mats) if (m) m.dispose();
    });
    this.implicitGroup.clear();
    this.implicitModel = null;
    this.hasImplicit = false;
    this.hasMesh = false;
    this.activeStage = 'model';
    this._applyStageVisibility();
    if (this.onDataStageChange) this.onDataStageChange('model');
  }

  // =================================================================
  // 生命周期
  // =================================================================

  _onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h);

    if (this.projection === 'orthographic') {
      // 保持正交视锥高度不变，仅更新宽高比
      const hh = this.orthoCamera.top - this.orthoCamera.bottom;
      const aspect = w / h;
      this.orthoCamera.left = (-hh * aspect) / 2;
      this.orthoCamera.right = (hh * aspect) / 2;
      this.orthoCamera.updateProjectionMatrix();
    } else {
      this.perspCamera.aspect = w / h;
      this.perspCamera.updateProjectionMatrix();
    }
  }

  dispose() {
    window.removeEventListener('resize', this._onResize);
    this.renderer.domElement.removeEventListener('pointerdown', this._onCanvasPointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this._onCanvasPointerUp);
    this._disposeImplicit();
    this._disposeModel();
    if (this.viewCube) this.viewCube.dispose();
    this.transformControls.dispose();
    if (this._previewRenderer) {
      this._previewMat.dispose();
      this._previewSphere.geometry.dispose();
      this._previewRenderer.dispose();
    }
    this.controls.dispose();
    this.renderer.dispose();
    if (this.renderer.domElement.parentNode === this.container) {
      this.container.removeChild(this.renderer.domElement);
    }
  }
}
