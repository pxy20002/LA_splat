# 项目核心架构与背景知识 (Context)

本项目旨在将 AnchorSplat 的锚点(Anchor)机制与 GWM 的世界模型相结合。为了更好地进行代码级复现与替换，以下是两篇论文核心 Pipeline 的详细定义：

## 1. AnchorSplat 核心 Pipeline (待复现与提取的模块)
AnchorSplat 是一种无需逐场景优化的前馈 3DGS 框架，其核心亮点是**在 3D 空间中基于稀疏锚点(Anchors)预测高斯属性**，而非传统的基于 2D 像素(Pixel-aligned)预测。它的核心流水线（红框部分）包含以下几个阶段：

* **阶段 1: Anchor Predictor (几何先验与锚点生成)**
    * **输入**: 未知位姿的多视角图像 (Unposed Multi-view Images)。
    * **处理**: 使用预训练的 MVS 模型（如项目中现有的 `MapAnything`）预测出深度图 (Depth) 和相机位姿 (Camera Poses)。将 2D 深度图反投影到 3D 空间形成密集的 3D 点云，然后通过**最远点采样 (FPS, Farthest Point Sampling)** 将其降采样为固定数量的稀疏 3D 锚点集合 $\{A_j\}_{j=1}^N$ ($A_j \in \mathbb{R}^3$)。
	    * **重要**: 原论文中 Pretrained MVS 模块是**冻结的**，不参与训练。我们直接使用 MapAnything 预训练权重进行推理，不做微调。
* **阶段 2: Feature Extraction & Projection (特征提取与 3D 投影)**
    * **处理**: 使用一个轻量级的 2D U-Net 处理输入图像、深度图以及相机光线特征 (Ray embeddings)，提取出 2D 多视角特征图。
    * **投影**: 根据深度和位姿，将 2D 特征投影并“绑定”到上述的 3D 锚点上，生成**带有高维特征的 3D 锚点 (Anchor Features, $\tilde{A}_j$)**。
* **阶段 3: Gaussian Decoder (高斯解码器)**
    * **全局注意力交互**: 将这些 Anchor Features 送入一个由 16 层全局注意力块 (Global Attention Blocks) 组成的 Transformer 中，捕捉所有锚点在 3D 空间中的全局交互。
    * **属性解码 (MLP Heads)**: 经过交互的特征通过多个并行的单层 MLP，预测出高斯球的各项属性：中心偏移量 ($\delta\mu$)、不透明度 ($\alpha$)、缩放 ($s$)、旋转 ($r$) 和 球谐系数 ($sh$)。
    * **输出**: **每个 3D 锚点会固定衍生出 4 个 3D 高斯球**（最终坐标为锚点坐标 + 中心偏移量）。

*(注：论文中还包含一个 Gaussian Refiner 用于计算渲染误差并微调高斯属性，但就我们的融合目标而言，Decoder 输出的高斯参数是最核心的对接点。)*

---

## 2. GWM (Gaussian World Model) 核心 Pipeline (待改造的底座)
GWM 是一个基于 3DGS 的动作条件视频预测世界模型（应用于机器人操作）。它通过预测未来的 3D 高斯场景来辅助策略学习。

* **阶段 1: World State Encoding (世界状态编码与 3D VAE)**
    * **初步重建**: 先用 Splatt3R (基于 Mast3R) 从当前观测图像中预测出全场景的 3D 高斯球 $G$。
    * **3D Gaussian VAE (关键痛点)**: 因为场景高斯球数量不定，GWM 使用 VAE 进行压缩。先用 FPS 采样固定数量的 $N$ 个高斯球作为 Query，通过 Cross-Attention 编码器，将全量高斯球信息聚合，提取出 $N$ 个固定长度的**隐变量 (Latent Embeddings, $x \in \mathbb{R}^{N \times D}$)**。
* **阶段 2: Diffusion-based Dynamics Modeling (基于 DiT 的动力学预测)**
    * **输入**: 对未来状态的 Latent 加噪后，拼接旋转位置编码 (RoPE) 作为 DiT 的主输入。
    * **条件注入**: 
        * 当前时间步 (Time $\tau$)：通过 AdaLN 注入。
        * **机器人动作 (Action $a_t$)**：作为 DiT 内部 Cross-Attention 的 Key 和 Value 注入。
        * 历史观测状态：作为拼接条件注入。
    * **输出**: 在 EDM 扩散框架下，DiT (Diffusion Transformer) 预测并去噪，输出**未来时刻的 Latent Embeddings**。
* **阶段 3: Decoding (状态解码)**
    * 使用 3D VAE 的 Decoder 将预测出的未来 Latent Embeddings 还原为未来的 3D 高斯球，用于计算渲染损失或供下游 RL/IL 策略使用。

---

## 3. 本项目的终极融合思路 (The "New" Pipeline)
基于上述分析，我们旨在用 AnchorSplat 替换掉 GWM 中的 3D VAE 机制：
1.  **废弃 GWM 的 3D VAE 编码器**：不再将生成的完整高斯球压缩为 Latent。
2.  **提取 Anchor 作为 Token**：使用 AnchorSplat 的前端（MVS + U-Net投影）直接生成**带特征的 3D 锚点 (Anchor Features)**，将这些 Anchor Features 直接作为 Token 输入给 GWM 的 DiT。
3.  **DiT 预测未来 Anchor**：GWM 的 DiT 在动作 (Action) 的调节下，预测出**未来的 Anchor Features**。
4.  **复用 Gaussian Decoder**：将 DiT 预测出的未来 Anchor Features 送入 AnchorSplat 的 **Gaussian Decoder** (Transformer + MLPs)，直接解码出未来的完整 3D 高斯球参数。