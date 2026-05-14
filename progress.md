# AnchorSplat 复现进展

## 当前状态
- **阶段**: Phase 1 — 独立复现 AnchorSplat（不含 Gaussian Refiner）
- **开始日期**: 2026-05-06

---

## 已完成

### 2026-05-06 — 项目初始化
- **内容**: 分析三个子项目代码结构，创建 CLAUDE.md 项目索引，确立分阶段计划
- **涉及文件**: `CLAUDE.md`, `progress.md`, `notes.md`
- **备注**: `anchorsplat/` 目录当前仅含测试图片，无代码

### 2026-05-07 — Anchor Predictor 核心模块
- **内容**: 实现 `anchorsplat/anchor_predictor.py`：纯 PyTorch FPS 采样 + mask 过滤 + 3D 分位点裁剪 + `from_predictions()` 离线模式。支持 MapAnything 和 VGGT 两种 MVS 前端输出
- **涉及文件**: `anchorsplat/__init__.py`, `anchorsplat/anchor_predictor.py`, `test/test_anchor_predictor.py`
- **备注**: 
  - 无外部依赖（纯 PyTorch），FPS 手写实现
  - 自动处理有无 mask key、mask 全空等边缘情况
  - 大点云 (>50万) 用子集估算分位点 + 随机降采样，避免 OOM

### 2026-05-07 — MapAnything MVS 推理脚本 & 重大 Bug 修复
- **内容**: 
  - 编写 `test/run_mapanything.py`，MapAnything 离线推理 + 保存预测
  - 发现并修复 `init_model_from_config("mapanything")` **未加载预训练权重**的 Bug（只加载了 DINOv2 编码器，transformer/预测头随机初始化导致所有场景输出球壳）
  - 修复：改用 `MapAnything.from_pretrained("facebook/map-anything")`
- **涉及文件**: `test/run_mapanything.py`
- **备注**: 官方 demo 用的是 `from_pretrained()`，Hydra config 路径下未指定 checkpoint 导致 `pretrained_checkpoint_path=None`

### 2026-05-11 — 模块 2a: Plücker 射线嵌入
- **内容**: 实现 `anchorsplat/ray_embeddings.py`：从相机内参和外参计算 6 通道 Plücker 坐标（direction(3) + moment(3)）
- **涉及文件**: `anchorsplat/ray_embeddings.py`, `test/test_ray_embeddings.py`
- **备注**: 
  - 支持多种可视化模式（单张/slideshow/批量），支持多种 colormap（默认 jet 匹配论文）
  - 可选保存 `[10, H, W]` U-Net 输入张量供下游使用

### 2026-05-11 — 模块 2b: 2D U-Net 特征提取器
- **内容**: 实现 `anchorsplat/unet.py`：4 级编码器-解码器，输入 [B,10,H,W] → 输出 [B,64,H,W] 特征图
- **涉及文件**: `anchorsplat/unet.py`, `test/test_unet.py`
- **备注**: ~4.8M 参数，双线性上采样 + 跳跃连接，输出特征可视化通过（非全零/非全常数）

### 2026-05-11 — 模块 3: 特征投影模块
- **内容**: 实现 `anchorsplat/feature_projector.py`：3D锚点→相机投影→可见性检查（边界/前方/深度一致性）→bilinear 特征采样→多视角 average pooling
- **涉及文件**: `anchorsplat/feature_projector.py`, `test/test_feature_projector.py`
- **备注**: 
  - 深度一致性检查用 `||P_cam||`（射线距离）而非 `z_cam`（此前 bug 导致 97% 锚点被判不可见）
  - 合成场景测试 + 可见性过滤测试 + 真实数据集成测试全通过，63% 锚点可见

---

## 进行中

_(暂无)_

---

## 待办

1. ~~Anchor Predictor~~ ✅
2. ~~2D U-Net 特征提取器~~ ✅
### 2026-05-11 — 模块 4: Gaussian Decoder
- **内容**: 实现 `anchorsplat/gaussian_decoder.py`：16 层 Global Attention Transformer (640 dim, 10 heads) + 2 个单层 MLP + 5 个并行预测头，~79.7M 参数（论文 ~84M）
- **涉及文件**: `anchorsplat/gaussian_decoder.py`, `test/test_gaussian_decoder.py`
- **备注**: 
  - Attention 使用 PyTorch SDPA（自动调用 FlashAttention-2，H100 等效论文 Ascend Flash Attention）
  - 每锚点预测 4 个高斯球，5 个属性：δμ, α, s, r, sh(degree=0)
  - 约束：opacity(sigmoid), scale(exp), rotation(L2 normalize)
  - 5 项测试全通过（形状/值域/batch/梯度/端到端）

3. ~~特征投影模块~~ ✅
4. ~~Gaussian Decoder~~ ✅
5. 可微渲染 + 损失函数
6. 训练管道
