"""Read metrics.csv and show an interactive training dashboard.

Usage:
    python test/read_metrics.py metrics.csv             # interactive window
    python test/read_metrics.py metrics.csv --save      # save dashboard.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Interactive training dashboard from metrics.csv.")
    p.add_argument("csv_path", type=str, help="Path to metrics.csv.")
    p.add_argument("--save", action="store_true",
                   help="Save dashboard.png instead of showing interactive window.")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=str)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    steps = data[:, 0].astype(int)
    L = data[:, 1].astype(float)
    L1 = data[:, 2].astype(float)
    L2 = data[:, 3].astype(float)
    L3 = data[:, 4].astype(float)
    L4 = data[:, 5].astype(float)
    ssim_q = data[:, 6].astype(float)
    lpips_v = data[:, 7].astype(float)
    opacity = data[:, 8].astype(float)
    sx = data[:, 9].astype(float)
    sy = data[:, 10].astype(float)
    sz = data[:, 11].astype(float)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Total Loss
    ax = axes[0, 0]
    ax.plot(steps, L, linewidth=0.8, color="steelblue")
    ax.set_xlabel("Step"); ax.set_ylabel("Total Loss")
    ax.set_title("Total Loss"); ax.grid(True, alpha=0.3)

    # Top-right: Loss components
    ax = axes[0, 1]
    ax.plot(steps, L1, linewidth=0.8, label="L1 (rendering)")
    ax.plot(steps, L2, linewidth=0.8, label="L2 (depth)")
    ax.plot(steps, L3 * 100, linewidth=0.8, label="L3 (opacity) ×100")
    ax.plot(steps, L4 * 10, linewidth=0.8, label="L4 (scale) ×10")
    ax.set_xlabel("Step")
    ax.set_title("Loss Components (scaled)"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Bottom-left: SSIM + LPIPS
    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(steps, ssim_q, linewidth=0.8, color="tab:green", label="SSIM↑")
    ax2.plot(steps, lpips_v, linewidth=0.8, color="tab:orange", label="LPIPS↓")
    ax.set_xlabel("Step"); ax.set_ylabel("SSIM ↑", color="tab:green")
    ax2.set_ylabel("LPIPS ↓", color="tab:orange")
    ax.set_title("SSIM ↑ & LPIPS ↓")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7)
    ax.grid(True, alpha=0.3)

    # Bottom-right: alpha + scale per axis
    ax = axes[1, 1]
    ax.plot(steps, opacity, linewidth=0.8, color="tab:purple", label="α")
    ax.plot(steps, sx, linewidth=0.8, alpha=0.6, label="sX")
    ax.plot(steps, sy, linewidth=0.8, alpha=0.6, label="sY")
    ax.plot(steps, sz, linewidth=0.8, alpha=0.6, label="sZ")
    ax.set_xlabel("Step"); ax.set_title("Opacity & Scale per Axis")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    fig.suptitle(f"AnchorSplat Training — {csv_path.name}", fontsize=12)
    fig.tight_layout()

    if args.save:
        out = csv_path.parent / "dashboard.png"
        fig.savefig(out, dpi=150)
        print(f"Saved → {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
