这是一份为您整理好的 AnchorSplat 核心复现指南。这份文档详细提取了论文中的模型架构、数学公式与实验参数，您可以直接将其作为 Prompt 提供给 Claude Code 等 AI 编程代理器进行代码复现。

***

# AnchorSplat 复现开发指南 (Reproduction Guide)

## 1. 核心思路与架构概览
[cite_start]AnchorSplat 是一个前馈 (feed-forward) 的 3D 高斯溅射 (3DGS) 框架，其核心创新点在于摒弃了像素对齐 (pixel-aligned) 的高斯生成方式，转而直接在 3D 空间中利用 3D 几何先验（如深度图或点云）生成锚点对齐 (anchor-aligned) 的高斯表示 [cite: 21, 24, 25, 26]。

[cite_start]整个模型 Pipeline 包含三个主要模块 [cite: 119]：
1. **Anchor Predictor (锚点预测器)**
2. **Gaussian Decoder (高斯解码器)**
3. **Gaussian Refiner (高斯细化器)**

---

## 2. Pipeline 模块详细分解

### 2.1 Anchor Predictor
该模块负责从无位姿输入图像中提取初始的 3D 几何先验。
* [cite_start]**Backbone**: 默认使用预训练的 MVS 模型 `MapAnything` 来预测输入图像的深度 ($D_i$) 和相机位姿 ($K_i, P_i$) [cite: 124, 130, 551]。
* [cite_start]**3D 投影与下采样**: 将像素对齐的 2D 深度点利用相机内参和外参反投影到 3D 空间 [cite: 125, 174, 175][cite_start]。为了减少计算冗余，使用最远点采样算法 (Farthest Point Sampling, FPS) 进行下采样，生成稀疏的 3D Anchors $A_j$ [cite: 129, 130]。
* [cite_start]**边界处理**: 为了防止 MVS 预测不准确导致离群点（如漂浮点），需要引入 3D 裁剪 (clipping) 操作，将所有反投影的点限制在预定义的空间边界内 [cite: 567, 568]。

### 2.2 Gaussian Decoder
该模块负责将多视角特征融合到 3D 锚点上，并预测高斯属性。
* [cite_start]**特征提取**: 使用轻量级的 2D U-Net 编码输入图像。输入通道包括：RGB 图像 ($I_i$, 3通道)、深度图 ($D_i$, 1通道) 以及通过相机内外参计算的 Plücker 射线嵌入 ($Ray_i$, 6通道) [cite: 179, 180]。
* [cite_start]**特征投影与聚合**: 将提取的 2D 特征反投影到 3D Anchors 上。当多个视角的特征映射到同一个 Anchor 时，经过可见性和深度一致性检查后，使用 **Average Pooling (平均池化)** 进行特征聚合 [cite: 181, 538, 542, 544]。
* [cite_start]**3D 交互与属性预测**: 聚合后的特征输入到一个基于 Transformer 的高斯预测器中，以建模全局锚点的 3D 空间交互 [cite: 182][cite_start]。随后，通过 MLP 预测高斯的属性：中心偏移量 $\delta\mu$、透明度 $\alpha$、缩放尺度 $s$、旋转参数 $r$ 和球谐系数 $sh$ [cite: 183, 187]。
* **关键超参数设定**:
  * [cite_start]**每个 Anchor 默认预测生成 4 个 Gaussians** [cite: 184, 506]。
  * [cite_start]高斯球谐系数 (Spherical Harmonics) 的阶数 (degree) 设定为 0 [cite: 224]。
  * [cite_start]高斯的绝对中心位置计算公式为 $\mu_j=A_j+\delta\mu_j$（偏移量通常限制在 $A_j$ 周围的很小范围内） [cite: 188]。
  * [cite_start]网络规模：包含 16 个 attention blocks (640 channels) 和 2 个单层 MLP 块，参数量约 84M [cite: 223, 224]。

### 2.3 Gaussian Refiner (即插即用模块)
[cite_start]此模块利用渲染误差在极少的前向传递中进一步修正生成的高斯体 [cite: 123, 190]。
* [cite_start]**误差特征提取**: 使用预训练的 ResNet-18 分别从渲染出的图像 $\hat{I}_i$ 和真实图像 $I_i$ 中提取多尺度特征 ($1/2$, $1/4$, $1/8$) [cite: 193][cite_start]。将这些特征 resize 并拼接为统一的 $1/4$ 分辨率特征，计算两者差异作为渲染误差 [cite: 193, 194]。
* [cite_start]**误差反投影与更新**: 将 2D 渲染误差可微地反投影到对应的 3D 高斯位置 [cite: 194]。
* [cite_start]**网络结构**: 使用一个 Transformer block (512 channels) 捕捉空间交互，随后利用 Point Transformer 结合当前的高斯属性、Anchor 特征和误差特征来输出属性的更新量 $\delta\mathcal{G}_j$ [cite: 198, 200, 201, 225]。
* [cite_start]**规模参数**: 包含 1 个 attention block、4 个 serialized attention blocks (均 512 channels) 和一个单层 MLP，参数量约 31M [cite: 223, 225, 226]。

---

## 3. 损失函数与训练策略

[cite_start]模型采用**两阶段训练**策略 [cite: 208]。

### Stage 1: 训练 Gaussian Decoder
* [cite_start]**目标**: 基于预测器提供的锚点训练 Decoder 生成高斯体 [cite: 208]。
* [cite_start]**损失函数**: 结合了渲染损失 $l_I$、深度损失 $l_D$ 以及针对高斯不透明度 $l_{\alpha}$ 和体积 $l_s$ 的正则化项 [cite: 209]。
  * [cite_start]$L_{GSdec}=\lambda_{I}\sum l_{I}(\hat{I}_{i},I_{i})+\lambda_{D}\sum l_{1}(\hat{D}_{i},D_{i})+\lambda_{\alpha}l_{\alpha}(\alpha_{j})+\lambda_{s}l_{s}(s_{j})$ [cite: 210, 212]
  * [cite_start]其中图像渲染损失结合了 L1、SSIM 和 LPIPS: $l_{I}=l_{1}+\gamma_{SSIM}(1-SSIM)+\gamma_{LPIPS}LPIPS$ [cite: 211]
  * [cite_start]正则化项用于防止高斯变得过度透明 ($l_{\alpha}$) 或过大 ($l_s$) [cite: 215]。
* [cite_start]**Loss 权重配置**: $\lambda_{I}=200$, $\gamma_{SSIM}=0.2$, $\gamma_{LPIPS}=0.2$, $\lambda_{D}=100$, $\lambda_{\alpha}=1e-1$, $\lambda_{s}=1e4$ [cite: 216, 217]。
* [cite_start]**训练步数**: 5,000 steps [cite: 223]。

### Stage 2: 训练 Gaussian Refiner
* [cite_start]**目标**: 冻结 Stage 1 训练好的 Decoder，仅训练 Refiner 网络 [cite: 217]。
* [cite_start]**损失函数**: 在此阶段，仅使用渲染损失 $l_I(\hat{I}_{i},I_{i})$ 对渲染图像进行真实图像的监督训练 [cite: 218]。
* [cite_start]**训练步数**: 5,000 steps [cite: 223]。

---

## 4. 实验环境与工程细节
在让代码 Agent 编写 `train.py` 和 `dataset.py` 时，请务必参考以下环境和超参数设置：

* [cite_start]**底层框架**: PyTorch，并利用 Ascend Flash Attention 提高 attention 计算效率 [cite: 221]。
* [cite_start]**精度设置**: 全程使用 `bfloat16` 精度进行计算 [cite: 222]。
* [cite_start]**优化器**: AdamW [cite: 222]。
* [cite_start]**算力配置参考**: 原论文在 64 张 Ascend 910B3 (64GB) NPU 上进行训练 [cite: 222]。
* **数据集设定 (以 ScanNet++ V2 为主)**:
  * [cite_start]输入图像原始分辨率采用 $1168 \times 1752$ [cite: 231]。
  * [cite_start]渲染 (Rendering) 与 损失计算监督 (Supervision) 时的分辨率固定为 $448 \times 672$ [cite: 231]。
  * [cite_start]批处理大小 (Batch size) 参考：在消融实验中使用了 `bs=32` [cite: 507]。