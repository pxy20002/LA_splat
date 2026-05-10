# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

本项目分两阶段推进：

**Phase 1（当前）**: 独立复现 AnchorSplat（不含 Gaussian Refiner），在 `anchorsplat/` 中实现完整的 AnchorSplat 前馈 3DGS 管道：MVS 前端 → 锚点生成 → 2D U-Net 特征提取 → 特征投影 → Gaussian Decoder → 可微渲染 → 损失训练。

**Phase 2（后续）**: 将复现好的 AnchorSplat 与 GWM 融合。用 AnchorSplat 前端替换 GWM 的 3D VAE，Anchor Features 直接作为 DiT Token，DiT 预测未来 Anchor Features，Gaussian Decoder 解码出未来高斯球。

目录结构：
- `anchorsplat/` — Phase 1 正式代码目标目录（当前几乎为空，仅含测试图片）
- `test/` — 临时测试/原型代码（正式的才放进 anchorsplat/）
- `gaussianwm/` — GWM 完整开源代码（Phase 2 才会用到）
- `map-anything/` — MapAnything MVS 模型（用作冻结的深度/位姿预测前端）

完整架构设计见 `papers.md`，详细复现参数见 `anchorsplat_read.md`。

## 文件维护规则

- **`CLAUDE.md`** — 架构理解、环境配置、关键参数有变化时及时更新
- **`progress.md`** — 每次有实质性代码改动后记录：改动摘要、涉及文件、所属阶段。格式参照现有条目

## 环境

- **本地（5060笔记本）**: conda 环境名 `mapanything`，仅用于轻量开发/测试，不跑训练
- **服务器**: 配有 H100 GPU，所有训练和资源密集型任务同步到服务器执行
- 代码在本地编写，随时同步到服务器

## 常用命令

以下命令标注了执行位置。

### MapAnything 推理（本地可运行）

```bash
conda activate mapanything
cd map-anything
# 图片文件夹 -> 3D 重建（深度图 + 相机位姿）
python scripts/demo_images_only_inference.py --input_dir <path> --output_dir <path>
```

### GWM 训练（仅服务器，gaussianwm/ 目录下执行）

```bash
# VAE 训练（单机多卡）
torchrun --nproc_per_node=8 gaussianwm/train_vae.py --config-name=train_vae

# DiT 世界模型训练
torchrun --nproc_per_node=8 gaussianwm/train_diffusion.py --config-name=train_gwm

# 或用脚本启动
bash scripts/pretrain/vae.sh
bash scripts/pretrain/dit.sh
```

### 安装依赖

GWM 使用 `uv` 管理依赖（`pyproject.toml` + `uv.lock`），MapAnything 使用 `pip` + `pyproject.toml`。两个子项目需分别安装。

## 核心架构

### Phase 1: AnchorSplat 独立管道（当前实现目标）

```
RGB图像 [B,V,3,H,W]  (V=多视角数)
  → [冻结] MapAnything MVS → 深度图 + 相机位姿 + 内参
  → 反投影 → 密集3D点云 → 3D裁剪(clipping) → FPS采样 → 稀疏3D锚点 {A_j}
  → 2D U-Net (RGB(3) + Depth(1) + Plücker Ray(6) = 10通道) → 2D特征图
  → 特征反投影 + Average Pooling聚合(含可见性/深度一致性检查) → Anchor Features
  → Gaussian Decoder (16层Global Attention + MLP Heads) → 高斯参数
  → 可微渲染 (3DGS Rasterizer) → 渲染图像 + 渲染深度
  → Loss = λ_I·l_I + λ_D·l_D + λ_α·l_α + λ_s·l_s
```

### Phase 2: GWM 融合（后续，暂不实现）

```
GWM数据: RGB序列 [B,T,3,H,W]
  → [Phase 1 的 AnchorSplat 前端] → Anchor Features [B*T, N, D]
  → [替换VAE] 直接作为Token输入GWM的DiT
  → DiT (Action条件, AdaLN-Zero注入) → 未来Anchor Features
  → [Phase 1 的 Gaussian Decoder] → 未来高斯参数 + 渲染
```

**Gaussian Decoder 精确参数**（来自 `anchorsplat_read.md`）:
- 16 个 Attention Blocks, 640 channels
- 2 个单层 MLP blocks
- 参数量 ~84M
- SH degree = 0（仅基色，无视角相关效果）
- 输出属性：中心偏移 δμ、不透明度 α、缩放 s、旋转 r、球谐系数 sh

**2D U-Net 输入**：RGB(3) + Depth(1) + Plücker 射线嵌入(6) = 10 通道
**特征聚合**：多视图特征投影到同一锚点时，经过可见性和深度一致性检查后，用 Average Pooling 聚合

**训练策略**（两阶段）：
- Stage 1: 训练 Gaussian Decoder, 5000 steps
  - Loss = λ_I·l_I + λ_D·l_D + λ_α·l_α + λ_s·l_s
  - l_I = L1 + 0.2·(1-SSIM) + 0.2·LPIPS
  - 权重: λ_I=200, λ_D=100, λ_α=0.1, λ_s=10000
- Stage 2: 冻结 Decoder，训练 Gaussian Refiner, 5000 steps（融合范围外）
- 优化器: AdamW, 精度: bfloat16
- 数据集: ScanNet++ V2, 输入分辨率 1168×1752, 渲染/监督分辨率 448×672

### 关键文件索引

**GWM 核心代码 (gaussianwm/gaussianwm/):**
- `gwm_predictor.py` — 顶层模型编排器 `GaussianPredictor`，包含训练/rollout逻辑
- `diffusion/models.py` — `DiT` 类，AdaLN-Zero 条件注入，`ActionEmbedder`，`DiTBlock`
- `diffusion/denoiser.py` — `Denoiser`，EDM 框架包装器（preconditioning + 噪声调度）
- `diffusion/diffusion_sampler.py` — `DiffusionSampler`，Heun/Euler ODE 求解器
- `encoder/models_ae.py` — `AutoEncoder`/`KLAutoEncoder`，3D VAE（PointEmbed + Cross-Attention + Self-Attention）
- `processor/regressor.py` — `Splatt3rRegressor`，封装 MASt3R 的 2D→3D 高斯预测
- `processor/datasets.py` — `DroidDataset`，RLDS 数据管道，10维 action，分段采样

**GWM 配置 (gaussianwm/configs/):**
- `train_gwm.yaml` — DiT 训练主配置
- `train_vae.yaml` — VAE 训练主配置
- `world_model/gwm.yaml` — DiT 架构参数（hidden=384, depth=12, heads=6）
- `vae/transformer.yaml` — VAE 架构参数（64 latents, dim=64）
- `dataset/droid.yaml` — 数据集参数（segment_length=10, context_length=2）

**MapAnything 核心代码 (map-anything/mapanything/):**
- `models/mapanything/model.py` — `MapAnything` 类，核心 Transformer（2178行）
- `models/mapanything/modular_dust3r.py` — 模块化 DUSt3R 封装
- `utils/image.py` — `load_images()` 图片加载/预处理/归一化
- `utils/geometry.py` — 几何运算工具（投影、四元数、光线，2189行）
- `utils/inference.py` — 推理管道：验证→预处理→前向→后处理

## 待实现模块 — Phase 1（anchorsplat/ 目录）

按依赖顺序排列：

1. **Anchor Predictor** — 调用 MapAnything 获取深度+位姿+内参，反投影到3D，3D裁剪，FPS采样锚点
2. **2D U-Net 特征提取器** — 轻量级U-Net，输入 10 通道（RGB 3 + Depth 1 + Plücker Ray 6），输出 2D 特征图
3. **特征投影模块** — 利用深度和位姿将 2D 特征反投影到 3D 锚点，Average Pooling 聚合（含可见性/深度一致性检查）
4. **Gaussian Decoder** — 16 层 Global Attention（640 channels）+ 2 个单层 MLP Heads（~84M 参数）。每个锚点→4 个高斯球，输出：中心偏移 δμ、不透明度 α、缩放 s、旋转 r、SH 系数（degree=0）
5. **可微渲染 + 损失** — 3DGS Rasterizer 渲染图像和深度，计算复合损失（L1+SSIM+LPIPS + 深度L1 + 正则化）
6. **训练管道** — Dataset（ScanNet++ V2）+ 训练循环 + 检查点，8000 steps

## Phase 2 待实现（后续，暂不碰）

- GWM DiT 集成层 — 用 Anchor Features 替换 VAE Latent
- 联合训练管道 — DiT + Gaussian Decoder 端到端

## 不需要实现

- Gaussian Refiner（Phase 2 也不包括）
- MVS 模型训练（MapAnything 保持冻结）

## 技术要点

- GWM 使用 **EDM (Elucidating Diffusion Models)** 框架，不是标准 DDPM。噪声调度用 log-normal sigma 采样，preconditioning 系数为 c_in/c_out/c_skip
- Action 通过 **AdaLN-Zero** 注入 DiT（全局条件调制），不是拼接到 token 序列
- MapAnything 输出使用 **OpenCV 约定**：+X右 +Y下 +Z前（cam2world）
- GWM 数据集来自 **DROID**（机器人操作），RLDS 格式，10维 action = pos_delta(3) + rot6d(6) + gripper(1)
- Splatt3R 输出的 14 维高斯特征 = means(3) + means_in_other_view(3) + scales(3) + rotations(4) + sh_dc(3) + opacities(1)
- AnchorSplat 中每个 3D 锚点固定衍生 **4 个**高斯球，中心坐标 μⱼ = Aⱼ + δμⱼ（偏移量限制在锚点周围小范围内）
- AnchorSplat SH degree = 0（仅 DC 分量，无视角依赖），精度 bfloat16，优化器 AdamW
- Plücker 射线嵌入 = 6 通道（由相机内参+外参计算得到的方向+力矩表示）
- 3D 裁剪 (clipping)：反投影后需裁剪到预定义空间边界，防止 MVS 离群点
- 训练分辨率：输入 1168×1752，渲染/监督 448×672（ScanNet++ V2）

