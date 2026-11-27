#!/usr/bin/env python3

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def load_data(weights_path: str, dist_path: str, dir_path: str):
    weights = np.loadtxt(weights_path, delimiter=",", dtype=np.int64)
    dist = np.loadtxt(dist_path, delimiter=",", dtype=np.float64)
    direction = np.loadtxt(dir_path, delimiter=",", dtype=np.int8)
    return weights, dist, direction


def render_flood_field(
    weights: np.ndarray,
    dist: np.ndarray,
    direction: np.ndarray,
    output_path: str = "output/flood_field.png",
    show_arrows: bool = True,
    title_prefix: str = "Flood Field",
):
    n = weights.shape[0]
    fig, (ax_terrain, ax_flood) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=180)

    target_coords = np.argwhere(dist == 0)
    target_rc = target_coords[0] if len(target_coords) > 0 else None

    # --- Panel 1: Input Terrain Weights (Green -> Yellow -> Red, Obstacles Black) ---
    terrain_masked = np.ma.masked_where(weights == 0, weights)
    cmap_terrain = plt.cm.RdYlGn_r.copy()
    cmap_terrain.set_bad(color="#111111")

    im_terrain = ax_terrain.imshow(terrain_masked, cmap=cmap_terrain, origin="upper")
    ax_terrain.set_title(f"Terrain Traversal Costs ({n}x{n})")
    ax_terrain.set_xlabel("Column")
    ax_terrain.set_ylabel("Row")
    fig.colorbar(im_terrain, ax=ax_terrain, fraction=0.046, pad=0.04, label="Cell Entry Cost")

    # --- Panel 2: Output Flood Field ---
    unreachable_mask = (weights == 0) | (dist < 0)
    dist_masked = np.ma.masked_where(unreachable_mask, dist)

    cmap_flood = plt.cm.plasma_r.copy()
    cmap_flood.set_bad(color="#111111")

    im_flood = ax_flood.imshow(dist_masked, cmap=cmap_flood, origin="upper")
    ax_flood.set_title(f"{title_prefix} (Cost & Flow to Target)")
    ax_flood.set_xlabel("Column")
    ax_flood.set_ylabel("Row")
    fig.colorbar(im_flood, ax=ax_flood, fraction=0.046, pad=0.04, label="Cumulative Cost to Target")

    if show_arrows:
        step = max(1, n // 40)
        rows, cols = np.meshgrid(np.arange(0, n, step), np.arange(0, n, step), indexing="ij")
        sub_dir = direction[rows, cols]

        u = np.zeros_like(sub_dir, dtype=np.float32)
        v = np.zeros_like(sub_dir, dtype=np.float32)

        u[sub_dir == 1] = 1.0
        u[sub_dir == 3] = -1.0
        v[sub_dir == 0] = -1.0
        v[sub_dir == 2] = 1.0

        valid = sub_dir >= 0
        ax_flood.quiver(
            cols[valid],
            rows[valid],
            u[valid],
            v[valid],
            angles="xy",
            scale_units="xy",
            scale=1.35 / step,
            color="white",
            alpha=0.75,
            width=0.003,
            headwidth=3.5,
            headlength=4.0,
        )

    if target_rc is not None:
        tr, tc = target_rc
        for ax in (ax_terrain, ax_flood):
            ax.scatter(
                [tc],
                [tr],
                marker="*",
                s=190,
                color="#00e5ff",
                edgecolors="black",
                linewidths=1.1,
                zorder=5,
                label=f"Target ({tr}, {tc})",
            )
            ax.legend(loc="upper right", framealpha=0.85)

    out_parent = os.path.dirname(output_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved visualization to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render flood field heatmap from exported CSVs.")
    parser.add_argument("--weights", default="output/weights.csv", help="Path to weights.csv")
    parser.add_argument("--dist", default="output/flood_dist.csv", help="Path to flood_dist.csv")
    parser.add_argument("--dir", default="output/flood_dir.csv", help="Path to flood_dir.csv")
    parser.add_argument("--out", "--output", dest="out", default="output/flood_field.png", help="Output PNG path")
    parser.add_argument("--title", default="Flood Field", help="Title prefix for the flood field panel")
    parser.add_argument("--no-arrows", action="store_true", help="Disable flow direction arrows")
    args = parser.parse_args()

    w, d, dr = load_data(args.weights, args.dist, args.dir)
    render_flood_field(w, d, dr, output_path=args.out, show_arrows=not args.no_arrows, title_prefix=args.title)