"""Quick check: diff-gaussian-rasterization return format."""
import inspect
import torch
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings

# Check signature
sig = inspect.signature(GaussianRasterizer.__call__)
print(f"__call__ parameters: {list(sig.parameters.keys())}")

# Test with minimal Gaussians
device = "cuda"
N, H, W = 10, 64, 64
means = torch.randn(N, 3, device=device) * 0.1
means[:, 2] = means[:, 2].abs() + 2.0
quats = torch.randn(N, 4, device=device)
quats = quats / quats.norm(dim=-1, keepdim=True)
scales = torch.rand(N, 3, device=device) * 0.01
opacities = torch.sigmoid(torch.randn(N, device=device)) * 0.5 + 0.25
colors = torch.sigmoid(torch.randn(N, 3, device=device))

viewmat = torch.eye(4, device=device)
viewmat[2, 3] = 3.0  # move camera back

tanfovx, tanfovy = 0.5, 0.5
proj = torch.tensor([
    [2.0, 0.0, 0.0, 0.0],
    [0.0, 2.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.5],
    [0.0, 0.0, 1.0, 0.0],
], device=device)

w2c = torch.linalg.inv(viewmat)
full_proj = proj @ w2c

settings = GaussianRasterizationSettings(
    image_height=H, image_width=W,
    tanfovx=tanfovx, tanfovy=tanfovy,
    bg=torch.ones(3, device=device),
    scale_modifier=1.0,
    viewmatrix=w2c.T.contiguous(),
    projmatrix=full_proj.T.contiguous(),
    sh_degree=0,
    campos=viewmat[:3, 3],
    prefiltered=False, debug=False,
)

rasterizer = GaussianRasterizer(settings)
result = rasterizer(
    means3D=means,
    means2D=torch.zeros(N, 2, device=device),
    shs=None,
    colors_precomp=colors,
    opacities=opacities,
    scales=scales,
    rotations=quats,
    cov3D_precomp=None,
)

print(f"\nReturn type: {type(result)}")
print(f"Return length: {len(result)}")
for i, r in enumerate(result):
    if isinstance(r, torch.Tensor):
        print(f"  [{i}] Tensor shape={list(r.shape)}")
    else:
        print(f"  [{i}] {type(r).__name__}: {r}")
