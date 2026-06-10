# LA-Splat

AnchorSplat 复现项目。前馈 3D Gaussian Splatting 框架。

## 环境

```bash
pip install -r requirements.txt
# MapAnything (MVS前端) 需单独安装
git clone https://github.com/facebookresearch/map-anything.git
cd map-anything && pip install -e .
```

## 训练

```bash
# 图片文件夹 → 自动 MapAnything 推理 + 缓存
python train/train.py --image_dir ./datasets/images_scene \
  --steps 5000 --num_anchors 256 --save_every 500

# 或从缓存的 .pt 文件
python train/train.py --pt_path ./datasets/scene.pt \
  --steps 5000 --num_anchors 256
```

## 评估

```bash
python train/eval.py --ckpt <checkpoint.pt> \
  --image_dir ./datasets/images_scene --num_anchors 256 \
  --save_path eval.png
```

## 测试

```bash
python test/test_anchor_predictor.py
python test/test_ray_embeddings.py
python test/test_unet.py
python test/test_feature_projector.py
python test/test_gaussian_decoder.py
python test/test_gradient_flow.py
python test/test_renderer.py
python test/test_e2e.py
```
