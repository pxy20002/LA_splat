# 技术笔记

## MapAnything mask 过滤机制

### 入口
- `test/run_mapanything.py:86-92` — 调用 `model.infer()` 时传入：
  - `apply_mask=True`：启用 non-ambiguous mask
  - `mask_edges=True`：启用边缘检测 mask

### mask 生成位置
- **文件**：`map-anything/mapanything/utils/inference.py`
- **函数**：`postprocess_model_outputs_for_inference()` (第 288 行起)
- **mask 拼装逻辑**：第 411-483 行

### mask 的两层组成

| 层 | 来源 | 作用 |
|----|------|------|
| `non_ambiguous_mask` | MapAnything 模型学习输出（`model.infer()` 自动获得） | 过滤纹理模糊、重复图案等 MVS 匹配不可靠区域 |
| `edge_mask` | 后处理计算（第 447-483 行） | 过滤深度/法线突变边缘，因为这些地方深度值不稳定 |

**边缘检测细节**：
- 法线边缘：`normals_edge()`，用 `edge_normal_threshold`（默认 5.0）控制
- 深度边缘：`depth_edge()`，用 `edge_depth_threshold`（默认 0.03）控制
- 组合方式：`~(depth_edges & normal_edges)`，即深度和法线**同时**检测到边缘才排除

**最终 mask 公式**：
```
final_mask = non_ambiguous_mask & edge_mask
```

### mask 的应用方式
- 第 500-502 行：将 `pts3d`、`pts3d_cam`、`depth_along_ray`、`depth_z` 中 mask 为 False 的位置**置零**
- 第 504 行：mask 本身作为 `pred["mask"]` 输出供下游使用

### 调节方法
- 减少过滤：`apply_mask=False` 关掉全部 mask；`mask_edges=False` 只关边缘 mask
- 放宽边缘阈值：`edge_normal_threshold` 调大、`edge_depth_threshold` 调大
- 当前测试（4 张卡车图）通过率 6-9%，属于正常范围

---

## MapAnything 模型加载方式（重要）

### 错误方式
```python
model = init_model_from_config("mapanything", device=device)
```
**问题**：`init_model_from_config` 只加载了 DINOv2 编码器权重（从 torch hub），MapAnything 的 16 层 multi-view transformer、DPT 预测头、pose head 等核心组件**全为随机初始化**。导致所有场景输出均匀深度 ~1.0 + 相机位姿在原点 → 球壳状点云。

根因位置：`model.py:623` — `_load_pretrained_weights()` 检查 `self.pretrained_checkpoint_path is not None`，而 Hydra config 未指定该路径，权重加载被跳过。

### 正确方式
```python
from mapanything.models import MapAnything
model = MapAnything.from_pretrained("facebook/map-anything").to(device)
```
通过 `PyTorchModelHubMixin` 从 HuggingFace Hub 下载完整预训练权重。

### 历史教训
- 2026-05-07 发现：truck/train/playroom 三个场景可视化全部呈球壳状
- 诊断：`depth_along_ray` 全部 ~1.0（无变化），相机平移全部 < 3cm（无移动）
- 对比 VGGT：深度 0.3~6.0（正常变化），相机平移 > 1m（正常移动）
- 修复：改用 `from_pretrained()` 后点云形状正常
