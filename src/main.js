import './style.css';
import { Viewer } from './viewer/Viewer.js';

// ===== DOM 引用 =====
const $ = (id) => document.getElementById(id);
const viewport = $('viewport');
const fileInput = $('file-input');
const loadingOverlay = $('loading-overlay');
const loadingText = $('loading-text');
const emptyHint = $('empty-hint');
const modelName = $('model-name');
const infoPanel = $('info-panel');

let viewer = null;

function formatNumber(n) {
  return Number(n).toLocaleString('zh-CN');
}

// ===== 导入模型（OBJ / STL / STEP / STP） =====
// 「导入模型」按钮是 <label for="file-input">，点击由浏览器原生弹出文件选择框，
// 不依赖 JS，因此即使脚本初始化失败也能正常打开。

async function handleFile(file) {
  if (!file) return;

  loadingOverlay.hidden = false;
  loadingText.textContent = `正在加载 ${file.name} …`;

  try {
    if (!viewer) throw new Error('3D 查看器未初始化，请检查浏览器 WebGL 支持');
    await viewer.loadFile(file);
  } catch (err) {
    console.error(err);
    showToast(`加载失败：${err.message}`);
  } finally {
    loadingOverlay.hidden = true;
    loadingText.textContent = '加载中…';
    fileInput.value = '';
  }
}

fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

// 键盘可访问性：Enter / 空格 也可打开文件选择框
$('btn-open').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// ===== 初始化查看器 =====
// 若 WebGL 不可用，不阻塞导入功能，仅在视口内显示原因
try {
  viewer = new Viewer(viewport, { viewCubeMount: $('view-cube-canvas') });
  viewer.onModelLoaded = (info) => {
    emptyHint.hidden = true;
    modelName.textContent = info.name;
    modelName.title = info.name;
    $('info-vertices').textContent = formatNumber(info.vertices);
    $('info-faces').textContent = formatNumber(info.triangles);
    infoPanel.hidden = false;
    // STEP 模型默认显示 CAD 原始边界线，同步勾选状态
    $('opt-edges').checked = viewer.shadingOptions.edges;
  };
  viewer.onModelCleared = () => {
    emptyHint.hidden = false;
    modelName.textContent = '';
    modelName.title = '';
    infoPanel.hidden = true;
    $('opt-edges').checked = false;
    $('opt-flat').checked = false;
    $('opt-normal').checked = false;
    $('opt-env').checked = true;
    envBox.hidden = false;
    setEnvActive('studio');
    $('opt-specular').value = 15;
    $('opt-specular-val').textContent = '15';
    $('opt-env-strength').value = 50;
    $('opt-env-strength-val').textContent = '50';
    $('opt-env-bg').checked = false;
    viewer?.setEnvBackgroundVisible(false);
    $('opt-color').value = '#ffffff';
    optBackfaceMode.value = 'solid';
    optBackfaceColor.value = '#d8d2aa';
    syncBackfaceUI();
    axisInput.hidden = true;
  };
  viewer.onGizmoAxisPress = (axis, mode) => {
    const uniformScale = mode === 'scale' && (axis === 'XYZ' || axis === 'XYZE');
    if (!axis || (!uniformScale && axis.length !== 1)) return; // 单轴 X/Y/Z，或缩放中心 XYZ
    const label =
      mode === 'rotate'
        ? `绕 ${axis} 旋转 (°)`
        : mode === 'scale'
          ? uniformScale
            ? '等比例缩放'
            : `${axis} 轴缩放`
          : `${axis} 轴平移`;
    $('axis-input-label').textContent = label;
    axisInput.hidden = false;
    axisInputValue.value = '';
    axisInputValue.focus();
  };
  // 开发模式下暴露实例，便于调试（生产构建不会包含）
  if (import.meta.env.DEV) window.__viewer = viewer;
} catch (err) {
  console.error(err);
  emptyHint.textContent = `无法启动 3D 查看器：${err.message}`;
}

// ===== 清除模型 =====
$('btn-clear').addEventListener('click', () => viewer?.clearModel());

// ===== 着色模式（可叠加） =====
const shadingDropdown = $('shading-dropdown');
const shadingMenu = $('shading-menu');

$('btn-shading').addEventListener('click', (e) => {
  e.stopPropagation();
  shadingDropdown.classList.toggle('open');
});

document.addEventListener('click', () => shadingDropdown.classList.remove('open'));
shadingMenu.addEventListener('click', (e) => e.stopPropagation());

$('opt-edges').addEventListener('change', (e) => viewer?.setShadingOption('edges', e.target.checked));
$('opt-flat').addEventListener('change', (e) => viewer?.setShadingOption('flat', e.target.checked));
$('opt-normal').addEventListener('change', (e) => {
  const on = e.target.checked;
  if (on) {
    // 法线着色与环境贴图着色互斥
    $('opt-env').checked = false;
    envBox.hidden = true;
    viewer?.setShadingOption('env', false);
  }
  viewer?.setShadingOption('normal', on);
});
$('opt-env').addEventListener('change', (e) => {
  const on = e.target.checked;
  envBox.hidden = !on;
  if (on) {
    // 环境贴图着色与法线着色互斥
    $('opt-normal').checked = false;
    viewer?.setShadingOption('normal', false);
  } else {
    // 关闭环境贴图着色时，同时关闭环境背景
    $('opt-env-bg').checked = false;
    viewer?.setEnvBackgroundVisible(false);
  }
  viewer?.setShadingOption('env', on);
});

// ===== 环境贴图下拉列表（材质球预览 + 名称） =====
const envList = $('env-list');
const envBox = $('env-box');
let envItems = [];

function setEnvActive(id) {
  envItems.forEach((item) => item.el.classList.toggle('active', item.id === id));
}

if (viewer) {
  viewer.onEnvironmentsLoaded = (list) => {
    envList.innerHTML = '';
    envItems = list.map((env) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = 'env-item';
      item.dataset.env = env.id;
      item.appendChild(viewer.createEnvPreview(env.raw, 64));
      const label = document.createElement('span');
      label.textContent = env.label;
      item.appendChild(label);
      item.addEventListener('click', () => {
        setEnvActive(env.id);
        viewer.setEnvType(env.id);
      });
      envList.appendChild(item);
      return { id: env.id, el: item };
    });
    setEnvActive(viewer.shadingOptions.envType);
    envBox.hidden = !$('opt-env').checked;
  };
}

// ===== 镜面反射率调节 =====
$('opt-specular').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  $('opt-specular-val').textContent = String(v);
  viewer?.setSpecular(v);
});

// ===== 环境反射强度调节 =====
$('opt-env-strength').addEventListener('input', (e) => {
  const v = Number(e.target.value);
  $('opt-env-strength-val').textContent = String(v);
  viewer?.setEnvStrength(v);
});

// ===== 显示环境背景开关 =====
$('opt-env-bg').addEventListener('change', (e) => viewer?.setEnvBackgroundVisible(e.target.checked));

// ===== 模型固有色 =====
$('opt-color').addEventListener('input', (e) => viewer?.setModelColor(e.target.value));

// ===== 反面着色 =====
const optBackfaceMode = $('opt-backface-mode');
const optBackfaceColor = $('opt-backface-color');

function syncBackfaceUI() {
  optBackfaceColor.disabled = optBackfaceMode.value !== 'solid';
}
optBackfaceMode.addEventListener('change', (e) => {
  syncBackfaceUI();
  viewer?.setBackfaceMode(e.target.value);
});
optBackfaceColor.addEventListener('input', (e) => viewer?.setBackfaceColor(e.target.value));
syncBackfaceUI();

// ===== 视角复位 =====
$('btn-view-reset').addEventListener('click', () => viewer?.resetView());

// ===== 透视 / 正交切换 =====
const btnProjection = $('btn-projection');
const setProjectionLabel = (mode) => {
  btnProjection.textContent = mode === 'orthographic' ? '透视' : '正交';
};
btnProjection.addEventListener('click', () => {
  const next = viewer?.projection === 'orthographic' ? 'perspective' : 'orthographic';
  viewer?.setProjection(next);
  setProjectionLabel(next);
});
setProjectionLabel('perspective');

// ===== 操作轴 =====
const gizmoModes = $('gizmo-modes');
const axisInput = $('axis-input');
const axisInputValue = $('axis-input-value');

// ===== 格线显示开关 =====
$('btn-grid').addEventListener('click', () => {
  const on = $('btn-grid').classList.toggle('active');
  viewer?.setGridVisible(on);
});

function hideAxisInput() {
  axisInput.hidden = true;
}

$('btn-gizmo').addEventListener('click', () => {
  const open = $('btn-gizmo').classList.toggle('open');
  gizmoModes.hidden = !open;
  if (open) {
    // 展开时启用操作轴（使用当前选中的模式）
    viewer?.setGizmoEnabled(true);
  } else {
    // 收起时关闭操作轴
    hideAxisInput();
    viewer?.setGizmoEnabled(false);
  }
});

gizmoModes.querySelectorAll('.gizmo-mode-btn').forEach((btn) => {
  btn.addEventListener('click', () => {
    gizmoModes.querySelectorAll('.gizmo-mode-btn').forEach((b) => b.classList.remove('active'));
    btn.classList.add('active');
    hideAxisInput();
    viewer?.setGizmoMode(btn.dataset.mode);
  });
});

// 将模型包围盒中心移动到世界原点
$('btn-center-origin').addEventListener('click', () => viewer?.centerModelToOrigin());

function applyAxisValue() {
  const value = parseFloat(axisInputValue.value);
  if (Number.isFinite(value)) viewer?.applyGizmoValue(value);
  axisInputValue.value = '';
}

axisInputValue.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    e.preventDefault();
    applyAxisValue();
  } else if (e.key === 'Escape') {
    hideAxisInput();
  }
});

// ===== 撤销 / 重做（Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y） =====
document.addEventListener('keydown', (e) => {
  const tag = e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || e.target.isContentEditable) return;
  if (!(e.ctrlKey || e.metaKey)) return;
  const key = e.key.toLowerCase();
  if (key === 'z') {
    e.preventDefault();
    if (e.shiftKey) viewer?.redo();
    else viewer?.undo();
  } else if (key === 'y') {
    e.preventDefault();
    viewer?.redo();
  }
});

// ===== 拖拽导入模型 =====
['dragenter', 'dragover'].forEach((type) => {
  window.addEventListener(type, (e) => e.preventDefault());
});

window.addEventListener('drop', (e) => {
  e.preventDefault();
  const files = e.dataTransfer ? [...e.dataTransfer.files] : [];
  const modelFile = files.find((f) => {
    const ext = (f.name.split('.').pop() || '').toLowerCase();
    return ['obj', 'stl', 'stla', 'stlb', 'step', 'stp'].includes(ext);
  });
  if (modelFile) {
    handleFile(modelFile);
  } else if (files.length > 0) {
    showToast('仅支持 OBJ / STL / STEP / STP 格式文件');
  }
});

// ===== 简易 Toast 提示 =====
function showToast(msg) {
  let toast = $('toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'toast';
    toast.style.cssText =
      'position:fixed;left:50%;bottom:48px;transform:translateX(-50%);' +
      'background:#ff5f57;color:#fff;padding:8px 16px;border-radius:6px;' +
      'font-size:12.5px;box-shadow:0 6px 20px rgba(0,0,0,.4);z-index:100;' +
      'max-width:80vw;text-align:center;';
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.style.opacity = '1';
  clearTimeout(toast._t);
  toast._t = setTimeout(() => {
    toast.style.transition = 'opacity .4s';
    toast.style.opacity = '0';
  }, 3000);
}
