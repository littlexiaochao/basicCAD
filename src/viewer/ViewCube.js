import * as THREE from 'three';

/** 生成 26 个候选视角方向：6 个面 + 12 条边 + 8 个顶点 */
function buildCandidates() {
  const axes = [
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(-1, 0, 0),
    new THREE.Vector3(0, 1, 0),
    new THREE.Vector3(0, -1, 0),
    new THREE.Vector3(0, 0, 1),
    new THREE.Vector3(0, 0, -1),
  ];
  const list = [];
  for (const a of axes) list.push(a.clone());

  // 边方向：两个正交轴之和
  for (let i = 0; i < axes.length; i++) {
    for (let j = i + 1; j < axes.length; j++) {
      if (Math.abs(axes[i].dot(axes[j])) < 1e-6) {
        list.push(axes[i].clone().add(axes[j]).normalize());
      }
    }
  }

  // 顶点方向：三个轴符号组合
  for (const sx of [1, -1]) {
    for (const sy of [1, -1]) {
      for (const sz of [1, -1]) {
        list.push(new THREE.Vector3(sx, sy, sz).normalize());
      }
    }
  }
  return list;
}

/** 生成带文字的立方体面纹理 */
function makeLabelTexture(label) {
  const size = 256;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  // 面底色 0.2 灰（#333333），白色文字
  ctx.fillStyle = '#333333';
  ctx.fillRect(0, 0, size, size);
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 100px "PingFang SC", "Microsoft YaHei", sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, size / 2, size / 2 + 10);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export class ViewCube {
  /**
   * @param {HTMLElement} mount 挂载画布的容器
   * @param {(dir: THREE.Vector3) => void} onViewChange 点击后回调（方向向量）
   * @param {(dx: number, dy: number) => void} onViewRotate 拖拽旋转回调（像素增量）
   */
  constructor(mount, onViewChange, onViewRotate = null) {
    this.onViewChange = onViewChange;
    this.onViewRotate = onViewRotate;
    this.raycaster = new THREE.Raycaster();
    this.hovered = null;
    this._pressed = false;
    this._dragging = false;
    this._dragStart = null;
    this._lastPointer = null;

    // ---------- 渲染器 ----------
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setSize(108, 108, false);
    this.canvas = this.renderer.domElement;
    // 显示尺寸由 CSS 控制（.view-cube canvas 宽度 100%），与操作轴面板同宽
    this.canvas.style.width = '100%';
    this.canvas.style.height = 'auto';
    this.canvas.style.display = 'block';
    this.canvas.style.cursor = 'grab';
    this.canvas.style.touchAction = 'none';
    mount.appendChild(this.canvas);

    // ---------- 场景与相机（朝向由 sync() 与主相机同步） ----------
    this.scene = new THREE.Scene();
    // 小视场角 + 远距离 → 接近正交的轻微透视
    this.camera = new THREE.PerspectiveCamera(15, 1, 0.1, 20);
    this.camera.position.set(0, 0, 5.2);
    this.camera.lookAt(0, 0, 0);

    // ---------- 立方体（面顺序：+X -X +Y -Y +Z -Z） ----------
    this.faceDefs = [
      { label: '右', normal: new THREE.Vector3(1, 0, 0) },
      { label: '左', normal: new THREE.Vector3(-1, 0, 0) },
      { label: '后', normal: new THREE.Vector3(0, 1, 0) },
      { label: '前', normal: new THREE.Vector3(0, -1, 0) },
      { label: '顶', normal: new THREE.Vector3(0, 0, 1) },
      { label: '底', normal: new THREE.Vector3(0, 0, -1) },
    ];
    this.materials = this.faceDefs.map(
      (f) =>
        new THREE.MeshBasicMaterial({
          map: makeLabelTexture(f.label),
          transparent: false,
          opacity: 0.8,
        })
    );
    const geo = new THREE.BoxGeometry(1, 1, 1);
    this.cube = new THREE.Mesh(geo, this.materials);
    this.scene.add(this.cube);

    // 边线
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x999999, transparent: true, opacity: 0.9 })
    );
    edges.scale.setScalar(1.012);
    this.scene.add(edges);

    // 顶点小球
    const cornerGeo = new THREE.SphereGeometry(0.06, 12, 12);
    const cornerMat = new THREE.MeshBasicMaterial({ color: 0x999999 });
    for (const sx of [1, -1]) {
      for (const sy of [1, -1]) {
        for (const sz of [1, -1]) {
          const s = new THREE.Mesh(cornerGeo, cornerMat);
          s.position.set(sx / 2, sy / 2, sz / 2);
          this.scene.add(s);
        }
      }
    }

    this.candidates = buildCandidates();

    // ---------- 交互 ----------
    this._onPointerMove = this._onPointerMove.bind(this);
    this._onPointerLeave = this._onPointerLeave.bind(this);
    this._onPointerDown = this._onPointerDown.bind(this);
    this._onPointerUp = this._onPointerUp.bind(this);
    this.canvas.addEventListener('pointermove', this._onPointerMove);
    this.canvas.addEventListener('pointerleave', this._onPointerLeave);
    this.canvas.addEventListener('pointerdown', this._onPointerDown);
    this.canvas.addEventListener('pointerup', this._onPointerUp);
  }

  /** 将画布坐标转为 NDC */
  _ndcFromEvent(e) {
    const rect = this.canvas.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - rect.left) / rect.width) * 2 - 1,
      -((e.clientY - rect.top) / rect.height) * 2 + 1
    );
  }

  /** 射线命中立方体表面后，吸附到最近的面/边/顶点方向 */
  _snapDirection(ndc) {
    this.raycaster.setFromCamera(ndc, this.camera);
    const hits = this.raycaster.intersectObject(this.cube, false);
    if (!hits.length) return null;
    const d = hits[0].point.clone().normalize();
    let best = null;
    let bestDot = -Infinity;
    for (const c of this.candidates) {
      const dot = c.dot(d);
      if (dot > bestDot) {
        bestDot = dot;
        best = c;
      }
    }
    return best;
  }

  /** 悬停高亮：高亮与该方向相邻的面 */
  _setHover(dir) {
    if (this.hovered && dir && this.hovered.equals(dir)) return;
    this.hovered = dir;
    for (let i = 0; i < this.materials.length; i++) {
      const adjacent = dir && this.faceDefs[i].normal.dot(dir) > 0.5;
      this.materials[i].color.set(adjacent ? 0x6fb0ff : 0xffffff);
    }
  }

  _onPointerMove(e) {
    if (this._pressed) {
      // 按住且位移超过阈值后进入拖拽旋转模式
      if (!this._dragging) {
        const dist = Math.hypot(
          e.clientX - this._dragStart.x,
          e.clientY - this._dragStart.y
        );
        if (dist < 4) return; // 未超过阈值：保持待单击状态，不旋转
        this._dragging = true;
        this.canvas.style.cursor = 'grabbing';
      }
      const dx = e.clientX - this._lastPointer.x;
      const dy = e.clientY - this._lastPointer.y;
      this._lastPointer = { x: e.clientX, y: e.clientY };
      if (this.onViewRotate) this.onViewRotate(dx, dy);
      return;
    }
    this._setHover(this._snapDirection(this._ndcFromEvent(e)));
  }

  _onPointerLeave() {
    if (!this._pressed) this._setHover(null);
  }

  _onPointerDown(e) {
    this._pressed = true;
    this._dragging = false;
    this._dragStart = { x: e.clientX, y: e.clientY };
    this._lastPointer = { x: e.clientX, y: e.clientY };
    try {
      this.canvas.setPointerCapture(e.pointerId);
    } catch (err) {
      // 忽略指针捕获失败
    }
  }

  _onPointerUp(e) {
    const wasDragging = this._dragging;
    const moved =
      this._dragStart &&
      Math.hypot(e.clientX - this._dragStart.x, e.clientY - this._dragStart.y);
    this._pressed = false;
    this._dragging = false;
    this._dragStart = null;
    this._lastPointer = null;
    this.canvas.style.cursor = 'grab';
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch (err) {
      // 忽略指针捕获释放失败
    }
    // 从未进入拖拽模式（位移小于阈值）才视为单击：切换视角
    if (!wasDragging && moved !== null && moved < 4) {
      const dir = this._snapDirection(this._ndcFromEvent(e));
      if (dir && this.onViewChange) this.onViewChange(dir);
    }
  }

  /** 每帧将小相机朝向与主相机同步 */
  sync(camera, target) {
    const dir = camera.position.clone().sub(target).normalize();
    this.camera.position.copy(this.cube.position).addScaledVector(dir, 5.2);
    this.camera.quaternion.copy(camera.quaternion);
    this.camera.updateMatrixWorld();
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.canvas.removeEventListener('pointermove', this._onPointerMove);
    this.canvas.removeEventListener('pointerleave', this._onPointerLeave);
    this.canvas.removeEventListener('pointerdown', this._onPointerDown);
    this.canvas.removeEventListener('pointerup', this._onPointerUp);
    this.renderer.dispose();
    if (this.canvas.parentNode) this.canvas.parentNode.removeChild(this.canvas);
  }
}
