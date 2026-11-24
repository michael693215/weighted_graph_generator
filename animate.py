from __future__ import annotations

import argparse
import os
from typing import Optional
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter


def format_time(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def build_frame_rgba(
    weights: np.ndarray,
    dist: np.ndarray,
    trace: np.ndarray,
    t_curr: float,
    t_prev: float,
    vmax: float,
    cmap,
) -> np.ndarray:
    """Builds an (N, N, 4) RGBA frame for timestamp t_curr without writing intermediate files."""
    n = weights.shape[0]
    rgba = np.zeros((n, n, 4), dtype=np.float32)

    # Default background for unsettled traversable cells: dark slate
    rgba[:, :] = [0.11, 0.14, 0.19, 1.0]

    # Obstacles: near-black
    obstacles = weights == 0
    rgba[obstacles] = [0.05, 0.05, 0.06, 1.0]

    # Cells settled up to t_curr
    settled = (~obstacles) & (trace >= 0) & (trace <= t_curr)
    if np.any(settled):
        norm_vals = np.clip(dist[settled] / vmax, 0.0, 1.0)
        rgba[settled] = cmap(norm_vals)

    # Highlight the active wavefront (cells settled in this frame's time slice) in bright cyan
    if t_curr > t_prev:
        frontier = (~obstacles) & (trace > t_prev) & (trace <= t_curr)
        rgba[frontier] = [0.0, 0.96, 0.83, 1.0]

    return rgba


def generate_animation(
    weights_path: str,
    dist_path: str,
    serial_trace_path: str,
    parallel_trace_path: Optional[str],
    output_path: str,
    frames: int,
    fps: int,
    hold_seconds: float,
):
    weights = np.loadtxt(weights_path, delimiter=",", dtype=np.int64, ndmin=2)
    dist = np.loadtxt(dist_path, delimiter=",", dtype=np.float64, ndmin=2)
    serial_trace = np.loadtxt(serial_trace_path, delimiter=",", dtype=np.int64, ndmin=2)

    traces = [("Serial Dijkstra", serial_trace)]
    if parallel_trace_path:
        parallel_trace = np.loadtxt(parallel_trace_path, delimiter=",", dtype=np.int64, ndmin=2)
        traces.append(("Parallel Delta-Stepping", parallel_trace))

    n = weights.shape[0]
    valid_dist = dist[(weights > 0) & (dist >= 0)]
    vmax = float(valid_dist.max()) if valid_dist.size > 0 else 1.0

    # Shared real-time ceiling across all provided solvers
    solver_max_ns = [float(max(1, tr.max())) for _, tr in traces]
    total_ns = max(solver_max_ns)

    # Total traversable & reachable cells for completion percentage
    total_reachable = int(np.sum((weights > 0) & (serial_trace >= 0)))
    total_reachable = max(1, total_reachable)

    # Locate target cell
    target_coords = np.argwhere(dist == 0)
    target_rc = target_coords[0] if len(target_coords) > 0 else None

    cmap = plt.cm.plasma_r
    num_panels = len(traces)
    fig, axes = plt.subplots(
        1,
        num_panels,
        figsize=(6.8 * num_panels, 6.8),
        dpi=130,
        squeeze=False,
    )
    axes = axes[0]

    # Timeline: `frames` active steps + hold frames at the end
    hold_frames = max(1, int(round(hold_seconds * fps)))
    time_steps = np.linspace(0.0, total_ns, frames)
    timeline = np.concatenate([time_steps, np.full(hold_frames, total_ns)])

    images = []
    for ax, (title, tr) in zip(axes, traces):
        init_rgba = build_frame_rgba(weights, dist, tr, 0.0, -1.0, vmax, cmap)
        im = ax.imshow(init_rgba, origin="upper", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if target_rc is not None:
            tr_r, tr_c = target_rc
            ax.scatter(
                [tr_c],
                [tr_r],
                marker="*",
                s=140,
                color="#00e5ff",
                edgecolors="black",
                linewidths=0.9,
                zorder=5,
            )
        images.append(im)

    suptitle = fig.suptitle("", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    def update(frame_idx: int):
        t_curr = timeline[frame_idx]
        t_prev = timeline[frame_idx - 1] if frame_idx > 0 else -1.0
        is_hold = frame_idx >= frames

        suptitle.set_text(
            f"Flood Field Propagation ({n}x{n})  |  Wall Clock: {format_time(t_curr)} / {format_time(total_ns)}"
        )

        for ax, im, (label, tr), finish_ns in zip(axes, images, traces, solver_max_ns):
            effective_prev = t_curr if is_hold else t_prev
            rgba = build_frame_rgba(weights, dist, tr, t_curr, effective_prev, vmax, cmap)
            im.set_data(rgba)

            settled_count = int(np.sum((weights > 0) & (tr >= 0) & (tr <= t_curr)))
            pct = min(100.0, 100.0 * settled_count / total_reachable)

            if t_curr >= finish_ns:
                ax.set_title(f"{label} — DONE ({format_time(finish_ns)})", color="#0a7e3a", fontweight="bold")
            else:
                ax.set_title(f"{label} — {pct:.0f}% settled", color="#222222")

        return images

    anim = FuncAnimation(fig, update, frames=len(timeline), interval=1000 / fps, blit=False)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".mp4":
        try:
            import imageio_ffmpeg
            mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
        writer = FFMpegWriter(fps=fps, bitrate=2400)
    else:
        writer = PillowWriter(fps=fps)

    print(f"Rendering {len(timeline)} frames to {output_path}...")
    anim.save(output_path, writer=writer)
    plt.close(fig)
    print(f"Saved animation to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate flood field propagation GIF or MP4 from CSV trace.")
    parser.add_argument("--weights", default="weights.csv", help="Path to weights.csv")
    parser.add_argument("--dist", default="flood_dist.csv", help="Path to flood_dist.csv")
    parser.add_argument("--serial", default="trace_serial.csv", help="Path to trace_serial.csv")
    parser.add_argument("--parallel", default=None, help="Optional path to trace_parallel.csv for side-by-side")
    parser.add_argument("--out", default="flood_animation.gif", help="Output filename (.gif or .mp4)")
    parser.add_argument("--frames", type=int, default=120, help="Number of active animation frames")
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--hold", type=float, default=1.5, help="Seconds to pause on the completed final frame")
    args = parser.parse_args()

    generate_animation(
        weights_path=args.weights,
        dist_path=args.dist,
        serial_trace_path=args.serial,
        parallel_trace_path=args.parallel,
        output_path=args.out,
        frames=args.frames,
        fps=args.