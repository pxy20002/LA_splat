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

---

## 进行中

_(暂无)_

---

## 待办

1. ~~Anchor Predictor~~ ✅ 已完成
2. 2D U-Net 特征提取器
3. 特征投影模块
4. Gaussian Decoder
5. 可微渲染 + 损失函数
6. 训练管道
