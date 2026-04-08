import os
import sys
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker
from dotenv import load_dotenv
from loguru import logger
import matplotlib.patches as mpatches

from src.read_datasets import load_dataset_numpy
from src.backends.mediapipe_backend import MediaPipeBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
load_dotenv()
YOLO_DATASET_PATH = os.getenv("YOLO_DATASET_PATH")

COCO_NAMES = [
    "Nose",
    "L.Eye",
    "R.Eye",
    "L.Ear",
    "R.Ear",
    "L.Shoulder",
    "R.Shoulder",
    "L.Elbow",
    "R.Elbow",
    "L.Wrist",
    "R.Wrist",
    "L.Hip",
    "R.Hip",
    "L.Knee",
    "R.Knee",
    "L.Ankle",
    "R.Ankle",
]

MP_TO_COCO = {
    0: 0,
    2: 1,
    5: 2,
    7: 3,
    8: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 8,
    15: 9,
    16: 10,
    23: 11,
    24: 12,
    25: 13,
    26: 14,
    27: 15,
    28: 16,
}

BODY_GROUPS = {
    "Head": [0, 1, 2, 3, 4],
    "Torso": [5, 6, 11, 12],
    "Arms": [7, 8, 9, 10],
    "Legs": [13, 14, 15, 16],
}
GROUP_COLORS = {
    "Head": "#5C85D6",
    "Torso": "#E07B4F",
    "Arms": "#5BAD72",
    "Legs": "#B07DC9",
}

# Torso keypoint indices used for torso-based normalisation
# Left-shoulder(5) → Right-hip(12),  Right-shoulder(6) → Left-hip(11)
TORSO_PAIRS = [(5, 12), (6, 11)]

# Per-keypoint OKS sigma (from COCO paper, Table 1)
COCO_KP_SIGMAS = np.array(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ]
)

_SEP = "=" * 80
_SEP_THIN = "-" * 80


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logger(log_dir: str) -> str:
    """
    Configure loguru:
      - Remove the default stderr sink.
      - Add a coloured console sink (INFO and above).
      - Add a plain-text file sink (DEBUG and above) saved to log_dir.

    Returns the path of the log file created.
    """
    logger.remove()  # drop default handler

    # ── Console sink (coloured, INFO+) ──────────────────────────────────────
    console_fmt = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, format=console_fmt, level="INFO", colorize=True)

    # ── File sink (plain text, DEBUG+) ───────────────────────────────────────
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"eval_{timestamp}.log")

    file_fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {function}:{line} - {message}"
    )
    logger.add(
        log_path,
        format=file_fmt,
        level="DEBUG",
        encoding="utf-8",
        enqueue=True,  # thread-safe writes
        backtrace=True,  # full tracebacks on exceptions
        diagnose=True,
    )

    logger.info(f"Log file: '{log_path}'")
    return log_path


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _kp_group_color(kp_idx):
    for grp, indices in BODY_GROUPS.items():
        if kp_idx in indices:
            return GROUP_COLORS[grp]
    return "#888888"


def _kp_group_name(kp_idx):
    for grp, indices in BODY_GROUPS.items():
        if kp_idx in indices:
            return grp
    return "Other"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def mp_to_coco(mp_keypoints):
    """Map MediaPipe landmark indices to COCO-17 layout."""
    coco_kp = np.zeros((17, 2))
    coco_vis = np.zeros(17)
    for mp_idx, coco_idx in MP_TO_COCO.items():
        kp = mp_keypoints[mp_idx]
        coco_kp[coco_idx] = [kp.x, kp.y]
        coco_vis[coco_idx] = kp.visibility
    return coco_kp, coco_vis


def denormalize(kp, w, h):
    """Convert [0,1]-normalised keypoints to pixel coordinates."""
    kp = kp.copy()
    kp[:, 0] *= w
    kp[:, 1] *= h
    return kp


# ---------------------------------------------------------------------------
# Core distance / scale computations
# ---------------------------------------------------------------------------


def compute_distances(gt, pred, vis_mask):
    dist = np.linalg.norm(gt - pred, axis=1)
    dist[vis_mask == 0] = np.nan
    return dist


def compute_bbox_scale(gt_kp_px, vis_mask):
    visible = gt_kp_px[vis_mask > 0]
    if len(visible) < 2:
        return 0.0
    x_min, y_min = visible.min(axis=0)
    x_max, y_max = visible.max(axis=0)
    scale = np.sqrt((x_max - x_min) * (y_max - y_min))
    return float(scale) if scale > 0 else 0.0


def compute_torso_scale(gt_kp_px, vis_mask):
    lengths = []
    for a, b in TORSO_PAIRS:
        if vis_mask[a] > 0 and vis_mask[b] > 0:
            lengths.append(np.linalg.norm(gt_kp_px[a] - gt_kp_px[b]))
    return float(np.mean(lengths)) if lengths else 0.0


def compute_scale(gt_kp_px, vis_mask):
    scale = compute_torso_scale(gt_kp_px, vis_mask)
    if scale <= 0:
        scale = compute_bbox_scale(gt_kp_px, vis_mask)
    return scale


def compute_normalized_distances(dist_px, scale):
    if scale <= 0:
        return np.full_like(dist_px, np.nan)
    return dist_px / scale


def compute_oks(dist_px, scale, vis_mask, sigmas=COCO_KP_SIGMAS):
    if scale <= 0:
        return np.full(17, np.nan)
    oks = np.full(17, np.nan)
    for i in range(17):
        if vis_mask[i] > 0 and not np.isnan(dist_px[i]):
            var = (scale * sigmas[i]) ** 2
            if var > 0:
                oks[i] = np.exp(-(dist_px[i] ** 2) / (2.0 * var))
    return oks


# ---------------------------------------------------------------------------
# Shared PCK helper
# ---------------------------------------------------------------------------


def _pck(distances_flat, threshold):
    valid = distances_flat[~np.isnan(distances_flat)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid <= threshold) * 100)


# ---------------------------------------------------------------------------
# OKS threshold metrics
# ---------------------------------------------------------------------------


def compute_oks_thresholds(all_oks, thresholds=(0.5, 0.75)):
    flat = all_oks.flatten()
    valid = flat[~np.isnan(flat)]

    global_oks_t = {}
    for t in thresholds:
        global_oks_t[t] = float(np.mean(valid > t) * 100) if len(valid) > 0 else 0.0

    per_kp_oks_t = {t: [] for t in thresholds}
    for i in range(17):
        col = all_oks[:, i]
        v = col[~np.isnan(col)]
        for t in thresholds:
            per_kp_oks_t[t].append(float(np.mean(v > t) * 100) if len(v) > 0 else 0.0)

    return global_oks_t, per_kp_oks_t


# ---------------------------------------------------------------------------
# Visibility-aware MAE
# ---------------------------------------------------------------------------


def compute_visibility_mae(all_distances_px, all_gt_vis):
    vis_mask = all_gt_vis > 0.5
    occ_mask = all_gt_vis == 0
    valid = ~np.isnan(all_distances_px)

    vis_dists = all_distances_px[vis_mask & valid]
    occ_dists = all_distances_px[occ_mask & valid]

    mae_visible = float(np.mean(vis_dists)) if len(vis_dists) > 0 else np.nan
    mae_occluded = float(np.mean(occ_dists)) if len(occ_dists) > 0 else np.nan

    return mae_visible, mae_occluded, len(vis_dists), len(occ_dists)


# ---------------------------------------------------------------------------
# Per-keypoint statistics
# ---------------------------------------------------------------------------


def _compute_per_kp_stats(all_distances, all_norm_distances=None):
    n_kp = 17
    n_samples = all_distances.shape[0]
    keys = ["mae", "median", "p25", "p75", "p95", "max", "valid_pct", "pck20px"]
    stats = {k: [] for k in keys}

    if all_norm_distances is not None:
        for k in ["norm_mae", "norm_median", "pck01", "pck02"]:
            stats[k] = []

    for i in range(n_kp):
        col = all_distances[:, i]
        valid = col[~np.isnan(col)]
        n_v = len(valid)
        stats["valid_pct"].append(n_v / n_samples * 100)
        if n_v > 0:
            stats["mae"].append(np.mean(valid))
            stats["median"].append(np.median(valid))
            stats["p25"].append(np.percentile(valid, 25))
            stats["p75"].append(np.percentile(valid, 75))
            stats["p95"].append(np.percentile(valid, 95))
            stats["max"].append(np.max(valid))
            stats["pck20px"].append(np.mean(valid <= 20) * 100)
        else:
            for k in ["mae", "median", "p25", "p75", "p95", "max", "pck20px"]:
                stats[k].append(0.0)

        if all_norm_distances is not None:
            ncol = all_norm_distances[:, i]
            nvalid = ncol[~np.isnan(ncol)]
            if len(nvalid) > 0:
                stats["norm_mae"].append(np.mean(nvalid))
                stats["norm_median"].append(np.median(nvalid))
                stats["pck01"].append(np.mean(nvalid <= 0.1) * 100)
                stats["pck02"].append(np.mean(nvalid <= 0.2) * 100)
            else:
                stats["norm_mae"].append(0.0)
                stats["norm_median"].append(0.0)
                stats["pck01"].append(0.0)
                stats["pck02"].append(0.0)

    return stats


# ---------------------------------------------------------------------------
# Plot helpers (shared style)
# ---------------------------------------------------------------------------

_STYLE = dict(facecolor="#F8F8F8")
_GRID = dict(linestyle="--", alpha=0.45)


def _save(fname, out_dir):
    """Save current figure to out_dir/fname and close."""
    path = os.path.join(out_dir, fname)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    logger.info(f"Saved plot: '{path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 01 — Global Metrics
# ---------------------------------------------------------------------------


def plot_global_metrics(
    all_distances,
    all_norm_distances,
    all_oks,
    valid_frames,
    missing_detections,
    oks_thresholds_global,
    mae_visible,
    mae_occluded,
    out_dir=".",
):
    mae = np.nanmean(all_distances)
    rmse = np.sqrt(np.nanmean(all_distances**2))
    norm_mae = np.nanmean(all_norm_distances)
    norm_rmse = np.sqrt(np.nanmean(all_norm_distances**2))
    mean_oks = np.nanmean(all_oks)

    n_evaluated = len(all_distances)
    det_rate = n_evaluated / valid_frames * 100 if valid_frames > 0 else 0.0
    miss_rate = missing_detections / valid_frames * 100 if valid_frames > 0 else 0.0

    flat_norm = all_norm_distances.flatten()
    pck01 = _pck(flat_norm, 0.1)
    pck02 = _pck(flat_norm, 0.2)

    oks50 = oks_thresholds_global.get(0.50, 0.0)
    oks75 = oks_thresholds_global.get(0.75, 0.0)

    vis_str = f"{mae_visible:.2f} px" if not np.isnan(mae_visible) else "N/A"
    occ_str = (
        f"{mae_occluded:.2f} px"
        if not np.isnan(mae_occluded)
        else "N/A (no occluded GT)"
    )

    fig, ax = plt.subplots(figsize=(5.5, 11), **_STYLE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Global Metrics", fontsize=13, fontweight="bold", pad=10)

    sections = [
        ("── Primary (Normalised) ──", "", "#888888", True),
        ("Normalised MAE", f"{norm_mae:.4f}", "#5C85D6", False),
        ("Normalised RMSE", f"{norm_rmse:.4f}", "#5C85D6", False),
        ("PCK@0.1  (norm)", f"{pck01:.1f}%", "#5BAD72", False),
        ("PCK@0.2  (norm)", f"{pck02:.1f}%", "#5BAD72", False),
        ("Mean OKS", f"{mean_oks:.4f}", "#E07B4F", False),
        ("OKS@0.5", f"{oks50:.1f}%", "#E07B4F", False),
        ("OKS@0.75", f"{oks75:.1f}%", "#E07B4F", False),
        ("Detection Rate", f"{det_rate:.1f}%", "#B07DC9", False),
        ("Miss Rate", f"{miss_rate:.1f}%", "#C97D7D", False),
        ("── Visibility Split (Pixel) ──", "", "#888888", True),
        ("MAE — Visible KPs", vis_str, "#5BAD72", False),
        ("MAE — Occluded KPs", occ_str, "#E07B4F", False),
        ("── Supplementary (Pixel) ──", "", "#888888", True),
        ("MAE", f"{mae:.2f} px", "#888888", False),
        ("RMSE", f"{rmse:.2f} px", "#888888", False),
    ]

    card_h = 0.060
    card_gap = 0.007
    y = 0.97
    for label, val, color, sep in sections:
        if sep:
            y -= 0.012
            ax.text(
                0.5,
                y,
                label,
                fontsize=7.5,
                color="#888888",
                ha="center",
                va="top",
                style="italic",
            )
            y -= 0.018
            continue
        y -= card_h + card_gap
        rect = FancyBboxPatch(
            (0.04, y),
            0.92,
            card_h,
            boxstyle="round,pad=0.01",
            linewidth=1.0,
            edgecolor=color,
            facecolor=color + "22",
        )
        ax.add_patch(rect)
        ax.text(0.10, y + card_h * 0.5, label, fontsize=8, color="#444444", va="center")
        ax.text(
            0.90,
            y + card_h * 0.5,
            val,
            fontsize=10,
            fontweight="bold",
            color=color,
            ha="right",
            va="center",
        )

    plt.savefig(
        os.path.join(out_dir, "01_global_metrics.png"), dpi=180, bbox_inches="tight"
    )
    logger.info(f"Saved plot: '{os.path.join(out_dir, '01_global_metrics.png')}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 02 — PCK (normalised thresholds — PRIMARY)
# ---------------------------------------------------------------------------


def plot_pck_normalized(
    all_norm_distances, thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3), out_dir="."
):
    flat = all_norm_distances.flatten()
    pck_vals = [_pck(flat, t) for t in thresholds]
    bar_colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(thresholds)))

    fig, ax = plt.subplots(figsize=(7, 4.5), **_STYLE)
    bars = ax.bar(
        [f"{t}" for t in thresholds],
        pck_vals,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.6,
    )
    for bar, val in zip(bars, pck_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_ylim(0, 115)
    ax.set_xlabel("Normalised threshold", fontsize=9)
    ax.set_ylabel("PCK (%)", fontsize=9)
    ax.set_title(
        "PCK (Normalised) — PRIMARY METRIC\n"
        "Correct iff  dist / scale  <  t   |   scale = torso or bbox",
        fontsize=11,
        fontweight="bold",
        pad=8,
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("02_pck_normalized.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 02b — PCK@20px (pixel, supplementary)
# ---------------------------------------------------------------------------


def plot_pck_pixel(all_distances, thresholds=(10, 20, 30), out_dir="."):
    flat = all_distances.flatten()
    pck_vals = [_pck(flat, t) for t in thresholds]
    bar_colors = ["#5C85D6", "#5BAD72", "#E07B4F"]

    fig, ax = plt.subplots(figsize=(5, 4), **_STYLE)
    bars = ax.bar(
        [f"PCK@{t}px" for t in thresholds],
        pck_vals,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.2,
        width=0.5,
    )
    for bar, val in zip(bars, pck_vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylim(0, 115)
    ax.set_ylabel("PCK (%)", fontsize=9)
    ax.set_title(
        "PCK (Pixel) — SUPPLEMENTARY ONLY\nNot scale-invariant; do not use as primary metric",
        fontsize=10,
        fontweight="bold",
        pad=8,
        color="#888888",
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("02b_pck_pixel_supplementary.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 03 — Error histogram (pixel + normalised side by side)
# ---------------------------------------------------------------------------


def plot_error_histogram_dual(all_distances, all_norm_distances, out_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), **_STYLE)

    for ax, data, xlabel, title, color in [
        (
            axes[0],
            all_distances,
            "Error (pixels)",
            "Pixel Error Distribution\n(supplementary)",
            "#888888",
        ),
        (
            axes[1],
            all_norm_distances,
            "Normalised error",
            "Normalised Error Distribution\n(PRIMARY)",
            "#5C85D6",
        ),
    ]:
        flat = data.flatten()
        flat = flat[~np.isnan(flat)]
        mae = np.mean(flat)
        median = np.median(flat)
        ax.hist(
            flat,
            bins=60,
            color=color,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.5,
            density=True,
        )
        ax.axvline(mae, color="#E07B4F", lw=1.8, ls="--", label=f"MAE {mae:.3f}")
        ax.axvline(
            median, color="#5BAD72", lw=1.8, ls=":", label=f"Median {median:.3f}"
        )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.legend(fontsize=8)
        ax.grid(axis="y", **_GRID)
        ax.set_facecolor("#FAFAFA")

    fig.suptitle(
        "Error Distribution Comparison", fontsize=13, fontweight="bold", y=1.01
    )
    _save("03_error_histogram_dual.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 04 — CDF (pixel + normalised)
# ---------------------------------------------------------------------------


def plot_error_cdf_dual(all_distances, all_norm_distances, out_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), **_STYLE)

    configs = [
        (
            axes[0],
            all_distances,
            "Error threshold (pixels)",
            "Error CDF — Pixel\n(supplementary)",
            (10, 20, 30),
            ["#5C85D6", "#5BAD72", "#E07B4F"],
            "#888888",
        ),
        (
            axes[1],
            all_norm_distances,
            "Normalised threshold",
            "Error CDF — Normalised (PRIMARY)",
            (0.1, 0.2, 0.3),
            ["#5C85D6", "#5BAD72", "#E07B4F"],
            "#5C85D6",
        ),
    ]

    for ax, data, xlabel, title, ths, clrs, lcolor in configs:
        flat = data.flatten()
        flat = flat[~np.isnan(flat)]
        sorted_e = np.sort(flat)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
        ax.plot(sorted_e, cdf * 100, color=lcolor, linewidth=2)
        for t, c in zip(ths, clrs):
            idx = np.searchsorted(sorted_e, t)
            pct = cdf[min(idx, len(cdf) - 1)] * 100
            ax.axvline(t, color=c, lw=1.2, ls="--", alpha=0.8)
            ax.axhline(pct, color=c, lw=0.8, ls=":", alpha=0.6)
            ax.text(
                t + sorted_e.max() * 0.01, pct - 3, f"{pct:.0f}%", color=c, fontsize=8
            )
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Cumulative (%)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylim(0, 103)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
        ax.grid(**_GRID)
        ax.set_facecolor("#FAFAFA")

    _save("04_error_cdf_dual.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 05 — Normalised MAE per keypoint (PRIMARY)
# ---------------------------------------------------------------------------


def plot_norm_mae_per_keypoint(all_norm_distances, out_dir="."):
    stats = _compute_per_kp_stats(all_norm_distances)
    colors = [_kp_group_color(i) for i in range(17)]
    y_pos = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), **_STYLE)
    bars = ax.barh(
        y_pos, stats["mae"], color=colors, edgecolor="white", linewidth=0.8, height=0.7
    )
    for bar, val in zip(bars, stats["mae"]):
        ax.text(
            val + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=9)
    ax.set_xlabel("Normalised MAE", fontsize=10)
    ax.set_title(
        "Normalised MAE per Keypoint (PRIMARY)\nDivided by √(bbox_w × bbox_h)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.invert_yaxis()
    ax.grid(axis="x", **_GRID)
    ax.set_facecolor("#FAFAFA")

    legend_handles = [
        mpatches.Patch(color=c, label=grp) for grp, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)

    _save("05_norm_mae_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 05b — Pixel MAE per keypoint (supplementary)
# ---------------------------------------------------------------------------


def plot_mae_per_keypoint(all_distances, out_dir="."):
    stats = _compute_per_kp_stats(all_distances)
    colors = [_kp_group_color(i) for i in range(17)]
    y_pos = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), **_STYLE)
    bars = ax.barh(
        y_pos, stats["mae"], color=colors, edgecolor="white", linewidth=0.8, height=0.7
    )
    for bar, val in zip(bars, stats["mae"]):
        ax.text(
            val + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=9)
    ax.set_xlabel("MAE (pixels) — supplementary", fontsize=10)
    ax.set_title(
        "Pixel MAE per Keypoint (supplementary)",
        fontsize=12,
        fontweight="bold",
        pad=8,
        color="#888888",
    )
    ax.invert_yaxis()
    ax.grid(axis="x", **_GRID)
    ax.set_facecolor("#FAFAFA")

    legend_handles = [
        mpatches.Patch(color=c, label=grp) for grp, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)

    _save("05b_pixel_mae_per_keypoint_supplementary.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 06 — Boxplot per keypoint
# ---------------------------------------------------------------------------


def plot_boxplot_per_keypoint(all_distances, out_dir="."):
    colors = [_kp_group_color(i) for i in range(17)]

    fig, ax = plt.subplots(figsize=(9, 7), **_STYLE)
    for i in range(17):
        valid = all_distances[:, i]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            continue
        ax.boxplot(
            valid,
            positions=[i],
            widths=0.6,
            vert=False,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker=".", markersize=2, alpha=0.4),
            boxprops=dict(facecolor=colors[i], linewidth=0),
        )
    ax.set_yticks(np.arange(17))
    ax.set_yticklabels(COCO_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Error (pixels)", fontsize=9)
    ax.set_title(
        "Error Distribution per Keypoint — Boxplot",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="x", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("06_boxplot_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 07 — PCK@0.2 per keypoint (normalised, PRIMARY)
# ---------------------------------------------------------------------------


def plot_pck02_per_keypoint(all_norm_distances, out_dir="."):
    pck02 = []
    for i in range(17):
        col = all_norm_distances[:, i]
        valid = col[~np.isnan(col)]
        pck02.append(np.mean(valid <= 0.2) * 100 if len(valid) > 0 else 0.0)

    pck_colors = plt.cm.RdYlGn(np.array(pck02) / 100)
    y_pos = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), **_STYLE)
    bars = ax.barh(
        y_pos, pck02, color=pck_colors, edgecolor="white", linewidth=0.6, height=0.7
    )
    for bar, val in zip(bars, pck02):
        ax.text(
            min(val + 1, 101),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f}%",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=8)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.set_xlabel("PCK@0.2 (normalised) [%]", fontsize=9)
    ax.set_title(
        "PCK@0.2 (Normalised) per Keypoint — PRIMARY\nred = worse, green = better",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.axvline(80, color="#555555", lw=1, ls="--", alpha=0.6)
    ax.grid(axis="x", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("07_pck02_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 08 — Extended stats table
# ---------------------------------------------------------------------------


def plot_stats_table(all_distances, all_norm_distances, out_dir="."):
    stats = _compute_per_kp_stats(all_distances, all_norm_distances)

    col_labels = [
        "Keypoint",
        "Group",
        "Norm.MAE ▲",
        "PCK@0.1 ▲",
        "PCK@0.2 ▲",
        "Pixel MAE",
        "Valid%",
    ]
    table_data = []
    for i in range(17):
        table_data.append(
            [
                COCO_NAMES[i],
                _kp_group_name(i),
                f"{stats['norm_mae'][i]:.4f}",
                f"{stats['pck01'][i]:.1f}%",
                f"{stats['pck02'][i]:.1f}%",
                f"{stats['mae'][i]:.2f}px",
                f"{stats['valid_pct'][i]:.1f}%",
            ]
        )

    fig, ax = plt.subplots(figsize=(13, 7), **_STYLE)
    ax.axis("off")
    ax.set_title(
        "Per-Keypoint Statistics (▲ = primary normalised metrics)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    tbl = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#3A3A5C")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    for i, row in enumerate(table_data):
        grp_color = GROUP_COLORS.get(row[1], "#888888")
        base_alpha = "18" if i % 2 == 0 else "30"
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            if j <= 1:
                cell.set_facecolor(grp_color + base_alpha)
                cell.set_text_props(color=grp_color, fontweight="bold")
            elif j <= 4:
                cell.set_facecolor("#EDF3FC" if i % 2 == 0 else "#DDE9F7")
            else:
                cell.set_facecolor("#F4F4F4" if i % 2 == 0 else "#ECECEC")

    _save("08_stats_table.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 09 — Group MAE P75
# ---------------------------------------------------------------------------


def plot_group_mae_p75(all_distances, all_norm_distances, out_dir="."):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), **_STYLE)

    for ax, data, ylabel, title in [
        (
            axes[0],
            all_distances,
            "Error (pixels)",
            "Body Group — Pixel (supplementary)",
        ),
        (
            axes[1],
            all_norm_distances,
            "Normalised error",
            "Body Group — Normalised (PRIMARY)",
        ),
    ]:
        group_data = {
            grp: data[:, idx].flatten()[~np.isnan(data[:, idx].flatten())]
            for grp, idx in BODY_GROUPS.items()
        }
        grp_names = list(BODY_GROUPS.keys())
        grp_clrs = [GROUP_COLORS[g] for g in grp_names]
        grp_mae = [np.mean(group_data[g]) for g in grp_names]
        grp_p75 = [np.percentile(group_data[g], 75) for g in grp_names]
        x = np.arange(len(grp_names))
        w = 0.35
        b1 = ax.bar(
            x - w / 2,
            grp_mae,
            width=w,
            color=grp_clrs,
            alpha=0.9,
            edgecolor="white",
            label="MAE",
        )
        b2 = ax.bar(
            x + w / 2,
            grp_p75,
            width=w,
            color=grp_clrs,
            alpha=0.5,
            edgecolor="white",
            hatch="//",
            label="P75",
        )
        for bars in [b1, b2]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + grp_mae[0] * 0.02,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(grp_names, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", **_GRID)
        ax.set_facecolor("#FAFAFA")

    _save("09_group_mae_p75.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 10 — Group violin
# ---------------------------------------------------------------------------


def plot_group_violin(all_distances, out_dir="."):
    group_data = {
        grp: all_distances[:, idx].flatten()[~np.isnan(all_distances[:, idx].flatten())]
        for grp, idx in BODY_GROUPS.items()
    }
    grp_names = list(BODY_GROUPS.keys())

    fig, ax = plt.subplots(figsize=(7, 5), **_STYLE)
    parts = ax.violinplot(
        [group_data[g] for g in grp_names],
        positions=range(len(grp_names)),
        showmedians=True,
        showextrema=False,
    )
    for pc, grp in zip(parts["bodies"], grp_names):
        pc.set_facecolor(GROUP_COLORS[grp])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(grp_names)))
    ax.set_xticklabels(grp_names, fontsize=11)
    ax.set_ylabel("Error (pixels)", fontsize=10)
    ax.set_title(
        "Error Distribution by Body Group (pixel, supplementary)",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.grid(axis="y", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("10_group_violin.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 11 — OKS distribution
# ---------------------------------------------------------------------------


def plot_oks_distribution(all_oks, out_dir="."):
    flat = all_oks.flatten()
    flat = flat[~np.isnan(flat)]
    mean_oks = np.mean(flat)

    fig, ax = plt.subplots(figsize=(7, 4.5), **_STYLE)
    ax.hist(
        flat,
        bins=50,
        color="#E07B4F",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        density=True,
    )
    ax.axvline(
        mean_oks, color="#5C85D6", lw=2, ls="--", label=f"Mean OKS = {mean_oks:.4f}"
    )
    ax.set_xlabel("OKS", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title(
        "Object Keypoint Similarity (OKS) Distribution",
        fontsize=12,
        fontweight="bold",
        pad=8,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", **_GRID)
    ax.set_facecolor("#FAFAFA")
    _save("11_oks_distribution.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 12 — Mean OKS per keypoint
# ---------------------------------------------------------------------------


def plot_oks_per_keypoint(all_oks, out_dir="."):
    oks_per_kp = [np.nanmean(all_oks[:, i]) for i in range(17)]
    colors = [_kp_group_color(i) for i in range(17)]
    y_pos = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), **_STYLE)
    bars = ax.barh(
        y_pos, oks_per_kp, color=colors, edgecolor="white", linewidth=0.8, height=0.7
    )
    for bar, val in zip(bars, oks_per_kp):
        ax.text(
            val + 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=8,
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=9)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Mean OKS", fontsize=10)
    ax.set_title("Mean OKS per Keypoint", fontsize=12, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.axvline(0.5, color="#555555", lw=1, ls="--", alpha=0.6)
    ax.grid(axis="x", **_GRID)
    ax.set_facecolor("#FAFAFA")

    legend_handles = [
        mpatches.Patch(color=c, label=grp) for grp, c in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9, framealpha=0.85)

    _save("12_oks_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 13 — OKS@0.5 and OKS@0.75 per keypoint
# ---------------------------------------------------------------------------


def plot_oks_thresholds_per_keypoint(per_kp_oks_t, out_dir="."):
    thresholds = sorted(per_kp_oks_t.keys())
    y_pos = np.arange(17)

    fig, axes = plt.subplots(
        1, len(thresholds), figsize=(9 * len(thresholds), 7), **_STYLE
    )
    if len(thresholds) == 1:
        axes = [axes]

    for ax, t in zip(axes, thresholds):
        vals = per_kp_oks_t[t]
        bar_colors = plt.cm.RdYlGn(np.array(vals) / 100)
        bars = ax.barh(
            y_pos, vals, color=bar_colors, edgecolor="white", linewidth=0.6, height=0.7
        )
        for bar, val in zip(bars, vals):
            ax.text(
                min(val + 1, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(y_pos)
        ax.set_yticklabels(COCO_NAMES, fontsize=8)
        ax.set_xlim(0, 115)
        ax.invert_yaxis()
        ax.set_xlabel(f"OKS@{t} (%)", fontsize=9)
        ax.set_title(
            f"OKS@{t} per Keypoint\nred = worse, green = better",
            fontsize=12,
            fontweight="bold",
            pad=8,
        )
        ax.axvline(80, color="#555555", lw=1, ls="--", alpha=0.6)
        ax.grid(axis="x", **_GRID)
        ax.set_facecolor("#FAFAFA")

    _save("13_oks_thresholds_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Data logging functions  (print_* → log_*)
# ---------------------------------------------------------------------------


def log_pck_normalized_data(
    all_norm_distances, thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3)
):
    flat = all_norm_distances.flatten()
    pck_vals = [_pck(flat, t) for t in thresholds]

    logger.info(_SEP)
    logger.info("PCK NORMALIZED DATA (PRIMARY METRIC)")
    logger.info(_SEP)
    logger.info("PCK@t: keypoint is *correct* iff dist_px / scale < t")
    logger.info("scale = torso scale (preferred) or bbox scale (fallback)")
    logger.info("")
    logger.info(f"{'Threshold':<12} {'PCK (%)'}")
    logger.info(_SEP_THIN[:30])
    for t, val in zip(thresholds, pck_vals):
        logger.info(f"{t:<12.3f} {val:.1f}%")
    logger.info("")
    logger.info(f"  PCK@0.1 : {pck_vals[1]:.1f}%  — Standard evaluation threshold")
    logger.info(f"  PCK@0.2 : {pck_vals[3]:.1f}%  — Common benchmark threshold")
    logger.info(_SEP)


def log_pck_pixel_data(all_distances, thresholds=(10, 20, 30)):
    flat = all_distances.flatten()
    pck_vals = [_pck(flat, t) for t in thresholds]

    logger.info(_SEP)
    logger.info("PCK PIXEL DATA (SUPPLEMENTARY ONLY — not scale-invariant)")
    logger.info(_SEP)
    logger.info(f"{'Threshold':<12} {'PCK (%)'}")
    logger.info(_SEP_THIN[:30])
    for t, val in zip(thresholds, pck_vals):
        logger.info(f"{t}px{'':8} {val:.1f}%")
    logger.info(_SEP)


def log_error_distribution_data(all_distances, all_norm_distances):
    logger.info(_SEP)
    logger.info("ERROR DISTRIBUTION STATISTICS")
    logger.info(_SEP)
    for name, data in [
        ("Pixel Error", all_distances),
        ("Normalized Error", all_norm_distances),
    ]:
        flat = data.flatten()
        flat = flat[~np.isnan(flat)]
        logger.info(f"{name}:")
        logger.info(f"  Mean (MAE)      : {np.mean(flat):.4f}")
        logger.info(f"  Median          : {np.median(flat):.4f}")
        logger.info(f"  Std Dev         : {np.std(flat):.4f}")
        logger.info(f"  25th Percentile : {np.percentile(flat, 25):.4f}")
        logger.info(f"  75th Percentile : {np.percentile(flat, 75):.4f}")
        logger.info(f"  95th Percentile : {np.percentile(flat, 95):.4f}")
        logger.info(f"  Max             : {np.max(flat):.4f}")
        logger.info(f"  Valid samples   : {len(flat)}")
    logger.info(_SEP)


def log_norm_mae_per_keypoint_data(all_norm_distances):
    stats = _compute_per_kp_stats(all_norm_distances)

    logger.info(_SEP)
    logger.info(
        "NORMALIZED MAE PER KEYPOINT (PRIMARY)  |  divided by √(bbox_w × bbox_h)"
    )
    logger.info(_SEP)
    logger.info(
        f"{'Keypoint':<15} {'Group':<8} {'Norm.MAE':<10} {'PCK@0.1':<10} {'PCK@0.2':<10}"
    )
    logger.info(_SEP_THIN[:70])

    pck01 = stats.get("pck01", [0.0] * 17)
    pck02 = stats.get("pck02", [0.0] * 17)
    for i in range(17):
        logger.info(
            f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} "
            f"{stats['mae'][i]:.4f}   {pck01[i]:.1f}%    {pck02[i]:.1f}%"
        )

    logger.info("")
    logger.info("Group averages:")
    for group, indices in BODY_GROUPS.items():
        logger.info(f"  {group}: {np.mean([stats['mae'][i] for i in indices]):.4f}")
    logger.info(_SEP)


def log_mae_per_keypoint_data(all_distances):
    stats = _compute_per_kp_stats(all_distances)

    logger.info(_SEP)
    logger.info("PIXEL MAE PER KEYPOINT (SUPPLEMENTARY)")
    logger.info(_SEP)
    logger.info(
        f"{'Keypoint':<15} {'Group':<8} {'MAE (px)':<10} "
        f"{'Median (px)':<12} {'PCK@20px':<10}"
    )
    logger.info(_SEP_THIN[:70])
    for i in range(17):
        logger.info(
            f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} "
            f"{stats['mae'][i]:.2f}     {stats['median'][i]:.2f}       "
            f"{stats['pck20px'][i]:.1f}%"
        )
    logger.info("")
    logger.info("Group averages (pixel MAE):")
    for group, indices in BODY_GROUPS.items():
        logger.info(f"  {group}: {np.mean([stats['mae'][i] for i in indices]):.2f}px")
    logger.info(_SEP)


def log_boxplot_statistics(all_distances):
    logger.info(_SEP)
    logger.info("BOXPLOT STATISTICS PER KEYPOINT (PIXEL ERROR)")
    logger.info(_SEP)
    logger.info(
        f"{'Keypoint':<15} {'Group':<8} {'Min':<8} {'Q1':<8} "
        f"{'Median':<8} {'Q3':<8} {'Max':<8} {'IQR':<8}"
    )
    logger.info(_SEP_THIN[:85])
    for i in range(17):
        col = all_distances[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            q1, q3 = np.percentile(valid, 25), np.percentile(valid, 75)
            logger.info(
                f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} "
                f"{np.min(valid):.2f}  {q1:.2f}  {np.median(valid):.2f}  "
                f"{q3:.2f}  {np.max(valid):.2f}  {q3 - q1:.2f}"
            )
        else:
            logger.warning(f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} No valid data")
    logger.info(_SEP)


def log_pck02_per_keypoint_data(all_norm_distances):
    stats = _compute_per_kp_stats(all_norm_distances)

    logger.info(_SEP)
    logger.info("PCK@0.2 (NORMALIZED) PER KEYPOINT (PRIMARY)  |  error < 0.2 = correct")
    logger.info(_SEP)
    logger.info(f"{'Keypoint':<15} {'Group':<8} {'PCK@0.2 (%)':<12} {'Norm.MAE':<10}")
    logger.info(_SEP_THIN[:60])

    group_pck02: dict[str, list] = {g: [] for g in BODY_GROUPS}
    for i in range(17):
        col = all_norm_distances[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            pck02 = np.mean(valid <= 0.2) * 100
            grp = _kp_group_name(i)
            if grp in group_pck02:
                group_pck02[grp].append(pck02)
            logger.info(
                f"{COCO_NAMES[i]:<15} {grp:<8} {pck02:.1f}%     {stats['mae'][i]:.4f}"
            )
        else:
            logger.warning(f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} No valid data")

    logger.info("")
    logger.info("Group averages (PCK@0.2):")
    for group, vals in group_pck02.items():
        if vals:
            logger.info(f"  {group}: {np.mean(vals):.1f}%")
    logger.info(_SEP)


def log_stats_table_data(all_distances, all_norm_distances):
    stats = _compute_per_kp_stats(all_distances, all_norm_distances)

    logger.info(_SEP)
    logger.info("COMPREHENSIVE STATISTICS TABLE  (▲ = primary normalized metrics)")
    logger.info(_SEP)
    header = (
        f"{'Keypoint':<15} {'Group':<8} {'Norm.MAE▲':<10} "
        f"{'PCK@0.1▲':<10} {'PCK@0.2▲':<10} {'Pixel MAE':<10} {'Valid%':<8}"
    )
    logger.info(header)
    logger.info(_SEP_THIN[: len(header)])
    for i in range(17):
        logger.info(
            f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} "
            f"{stats['norm_mae'][i]:.4f}    "
            f"{stats['pck01'][i]:.1f}%    "
            f"{stats['pck02'][i]:.1f}%    "
            f"{stats['mae'][i]:.2f}px  "
            f"{stats['valid_pct'][i]:.1f}%"
        )
    logger.info("")
    logger.info(f"  Avg Normalized MAE  : {np.mean(stats['norm_mae']):.4f}")
    logger.info(f"  Avg PCK@0.2 (norm)  : {np.mean(stats['pck02']):.1f}%")
    logger.info(f"  Avg Pixel MAE       : {np.mean(stats['mae']):.2f}px")
    logger.info(f"  Avg Valid %         : {np.mean(stats['valid_pct']):.1f}%")
    logger.info(_SEP)


def log_group_mae_p75_data(all_distances, all_norm_distances):
    logger.info(_SEP)
    logger.info("BODY GROUP STATISTICS — MAE AND P75")
    logger.info(_SEP)
    logger.info(f"{'Group':<10} {'Metric':<12} {'Pixel':<12} {'Normalized':<12}")
    logger.info(_SEP_THIN[:60])
    for group, indices in BODY_GROUPS.items():
        gp = all_distances[:, indices].flatten()
        gp = gp[~np.isnan(gp)]
        gn = all_norm_distances[:, indices].flatten()
        gn = gn[~np.isnan(gn)]
        if len(gp) > 0 and len(gn) > 0:
            logger.info(
                f"{group:<10} {'MAE':<12} {np.mean(gp):.2f}px   {np.mean(gn):.4f}"
            )
            logger.info(
                f"{group:<10} {'P75':<12} {np.percentile(gp, 75):.2f}px   "
                f"{np.percentile(gn, 75):.4f}"
            )
            logger.info(_SEP_THIN[:60])
    logger.info(_SEP)


def log_oks_distribution_data(all_oks):
    flat = all_oks.flatten()
    flat = flat[~np.isnan(flat)]

    logger.info(_SEP)
    logger.info("OKS (OBJECT KEYPOINT SIMILARITY) DISTRIBUTION")
    logger.info(_SEP)
    if len(flat) > 0:
        logger.info(f"Total valid OKS values : {len(flat)}")
        logger.info(f"Mean OKS               : {np.mean(flat):.4f}")
        logger.info(f"Median OKS             : {np.median(flat):.4f}")
        logger.info(f"Std Dev                : {np.std(flat):.4f}")
        logger.info(f"Min / Max              : {np.min(flat):.4f} / {np.max(flat):.4f}")
        logger.info("")
        logger.info("Percentiles:")
        for p in [10, 25, 50, 75, 90]:
            logger.info(f"  {p:>3}th : {np.percentile(flat, p):.4f}")
        logger.info("")
        logger.info("OKS Threshold Statistics:")
        for threshold in [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]:
            logger.info(
                f"  OKS > {threshold:.2f} : {np.mean(flat > threshold) * 100:.1f}%"
            )
    else:
        logger.warning("No valid OKS data available")
    logger.info(_SEP)


def log_oks_per_keypoint_data(all_oks):
    logger.info(_SEP)
    logger.info("MEAN OKS PER KEYPOINT")
    logger.info(_SEP)
    logger.info(
        f"{'Keypoint':<15} {'Group':<8} {'Mean OKS':<10} "
        f"{'OKS > 0.5 (%)':<15} {'OKS > 0.75 (%)':<15}"
    )
    logger.info(_SEP_THIN[:80])

    group_oks: dict[str, list] = {g: [] for g in BODY_GROUPS}
    for i in range(17):
        col = all_oks[:, i]
        valid = col[~np.isnan(col)]
        if len(valid) > 0:
            mean_v = np.mean(valid)
            grp = _kp_group_name(i)
            if grp in group_oks:
                group_oks[grp].append(mean_v)
            logger.info(
                f"{COCO_NAMES[i]:<15} {grp:<8} {mean_v:.4f}   "
                f"{np.mean(valid > 0.5) * 100:.1f}%       "
                f"{np.mean(valid > 0.75) * 100:.1f}%"
            )
        else:
            logger.warning(f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} No valid data")

    logger.info("")
    logger.info("Group averages (Mean OKS):")
    for group, vals in group_oks.items():
        if vals:
            logger.info(f"  {group}: {np.mean(vals):.4f}")
    logger.info(_SEP)


def log_oks_thresholds_per_keypoint_data(per_kp_oks_t):
    thresholds = sorted(per_kp_oks_t.keys())

    logger.info(_SEP)
    logger.info("OKS THRESHOLDS PER KEYPOINT  (OKS@t = % of valid KPs with OKS > t)")
    logger.info(_SEP)

    if thresholds:
        header = f"{'Keypoint':<15} {'Group':<8} "
        for t in thresholds:
            header += f"{'OKS@' + str(t):<12} "
        logger.info(header)
        logger.info(_SEP_THIN[: len(header)])
        for i in range(17):
            line = f"{COCO_NAMES[i]:<15} {_kp_group_name(i):<8} "
            for t in thresholds:
                line += f"{per_kp_oks_t[t][i]:.1f}%{'':6}"
            logger.info(line)
        logger.info("")
        logger.info("Group averages:")
        for group, indices in BODY_GROUPS.items():
            logger.info(f"  {group}:")
            for t in thresholds:
                avg = np.mean([per_kp_oks_t[t][i] for i in indices])
                logger.info(f"    OKS@{t}: {avg:.1f}%")
    logger.info(_SEP)


def log_cdf_key_points(all_distances, all_norm_distances):
    logger.info(_SEP)
    logger.info("CDF KEY POINTS (CUMULATIVE DISTRIBUTION FUNCTION)")
    logger.info(_SEP)

    for name, data, thresholds in [
        ("Pixel Error", all_distances, [10, 20, 30]),
        ("Normalized Error", all_norm_distances, [0.1, 0.2, 0.3]),
    ]:
        logger.info(f"{name}:")
        flat = data.flatten()
        flat = flat[~np.isnan(flat)]
        sorted_e = np.sort(flat)
        cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)

        logger.info(f"  {'Threshold':<12} {'Cumulative %':<15} Description")
        logger.info(f"  {_SEP_THIN[:50]}")
        for t in thresholds:
            idx = np.searchsorted(sorted_e, t)
            pct = cdf[min(idx, len(cdf) - 1)] * 100
            desc = (
                "Good accuracy"
                if t <= 0.1
                else "Standard threshold"
                if t <= 0.2
                else "Loose threshold"
            )
            logger.info(f"  {t:<12} {pct:.1f}%{'':9} {desc}")
        for pct_target in [50, 80, 90, 95]:
            idx = int(len(sorted_e) * pct_target / 100)
            if idx < len(sorted_e):
                logger.info(
                    f"  {sorted_e[idx]:<12.3f} {pct_target}%{'':9} "
                    f"{pct_target}th percentile"
                )
        logger.info("")
    logger.info(_SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Output directories ───────────────────────────────────────────────────
    eval_dir = os.path.join("eval_results", f"mediapipe_{timestamp}")
    log_dir = os.path.join("logs", f"mediapipe_{timestamp}")
    os.makedirs(eval_dir, exist_ok=True)

    # ── Logging — must be set up before any logger.* calls ───────────────────
    log_path = setup_logger(log_dir)

    logger.info(f"Eval output directory : '{eval_dir}'")
    logger.info(f"Log  output directory : '{log_dir}'")

    dataset = load_dataset_numpy(YOLO_DATASET_PATH)
    total_frames = len(dataset)
    logger.info(f"Dataset loaded: {total_frames} frames")

    mp_backend = MediaPipeBackend(static_image_mode=True)
    mp_backend.initialize()
    logger.info("MediaPipe backend initialised")

    all_distances: list = []
    all_norm_distances: list = []
    all_oks: list = []
    all_gt_vis: list = []

    invalid_frames = 0
    missing_detections = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        img = cv2.imread(sample["image_path"])
        if img is None:
            invalid_frames += 1
            logger.debug(f"Unreadable image skipped: {sample['image_path']}")
            continue

        h, w = img.shape[:2]
        result = mp_backend.process_frame(img)
        if result is None or not getattr(result, "keypoints", None):
            missing_detections += 1
            continue

        pred_kp_norm, _ = mp_to_coco(result.keypoints)
        pred_kp = denormalize(pred_kp_norm, w, h)
        gt_kp = denormalize(sample["keypoints"], w, h)
        gt_vis = sample["visible"]

        dist = compute_distances(gt_kp, pred_kp, gt_vis)
        scale = compute_scale(gt_kp, gt_vis)
        norm_dist = compute_normalized_distances(dist, scale)
        oks = compute_oks(dist, scale, gt_vis)

        all_distances.append(dist)
        all_norm_distances.append(norm_dist)
        all_oks.append(oks)
        all_gt_vis.append(gt_vis)

    if len(all_distances) == 0:
        logger.error("No samples evaluated — aborting.")
        return

    all_distances = np.array(all_distances)  # (N, 17)
    all_norm_distances = np.array(all_norm_distances)  # (N, 17)
    all_oks = np.array(all_oks)  # (N, 17)
    all_gt_vis = np.array(all_gt_vis)  # (N, 17)

    valid_frames = total_frames - invalid_frames
    n_evaluated = len(all_distances)
    det_rate = n_evaluated / valid_frames * 100 if valid_frames > 0 else 0.0
    miss_rate = missing_detections / valid_frames * 100 if valid_frames > 0 else 0.0

    # ── Primary normalised metrics ────────────────────────────────────────────
    global_nmae = np.nanmean(all_norm_distances)
    global_nrmse = np.sqrt(np.nanmean(all_norm_distances**2))
    flat_norm = all_norm_distances.flatten()
    pck01 = _pck(flat_norm, 0.1)
    pck02 = _pck(flat_norm, 0.2)

    # ── OKS metrics ───────────────────────────────────────────────────────────
    mean_oks = np.nanmean(all_oks)
    oks_global, oks_per_kp = compute_oks_thresholds(all_oks, thresholds=(0.50, 0.75))

    # ── Visibility-aware MAE ──────────────────────────────────────────────────
    mae_vis, mae_occ, n_vis, n_occ = compute_visibility_mae(all_distances, all_gt_vis)

    # ── Supplementary pixel metrics ───────────────────────────────────────────
    global_mae = np.nanmean(all_distances)
    global_rmse = np.sqrt(np.nanmean(all_distances**2))

    vis_str = f"{mae_vis:.2f} px  (n={n_vis})" if not np.isnan(mae_vis) else "N/A"
    occ_str = (
        f"{mae_occ:.2f} px  (n={n_occ})"
        if not np.isnan(mae_occ)
        else "N/A (no occluded GT)"
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  Total frames       : {total_frames}")
    logger.info(f"  Invalid (I/O err)  : {invalid_frames}")
    logger.info(f"  Valid frames       : {valid_frames}")
    logger.info(f"  Missing detections : {missing_detections}  ({miss_rate:.1f}%)")
    logger.info(f"  Detected           : {n_evaluated}  ({det_rate:.1f}%)")
    logger.info("  --- PRIMARY (Normalised) ---")
    logger.info(f"  Normalised MAE     : {global_nmae:.4f}")
    logger.info(f"  Normalised RMSE    : {global_nrmse:.4f}")
    logger.info(f"  PCK@0.1  (norm)    : {pck01:.1f}%")
    logger.info(f"  PCK@0.2  (norm)    : {pck02:.1f}%")
    logger.info(f"  Mean OKS           : {mean_oks:.4f}")
    logger.info(f"  OKS@0.50           : {oks_global[0.50]:.1f}%")
    logger.info(f"  OKS@0.75           : {oks_global[0.75]:.1f}%")
    logger.info("  --- Visibility Split (Pixel MAE) ---")
    logger.info(f"  MAE — Visible      : {vis_str}")
    logger.info(f"  MAE — Occluded     : {occ_str}")
    logger.info("  --- SUPPLEMENTARY (Pixel) ---")
    logger.info(f"  Pixel MAE          : {global_mae:.2f} px")
    logger.info(f"  Pixel RMSE         : {global_rmse:.2f} px")
    logger.info("=" * 60)
    logger.info(f"  Plots   → '{eval_dir}'")
    logger.info(f"  Log     → '{log_path}'")
    logger.info("=" * 60)

    # ── Detailed per-section logs + plots ────────────────────────────────────

    # Plot 01
    logger.info(_SEP)
    logger.info("GLOBAL METRICS DATA")
    logger.info(_SEP)
    logger.info(f"Normalised MAE     : {global_nmae:.4f}")
    logger.info(f"Normalised RMSE    : {global_nrmse:.4f}")
    logger.info(f"PCK@0.1 (norm)     : {pck01:.1f}%")
    logger.info(f"PCK@0.2 (norm)     : {pck02:.1f}%")
    logger.info(f"Mean OKS           : {mean_oks:.4f}")
    logger.info(f"OKS@0.5            : {oks_global[0.50]:.1f}%")
    logger.info(f"OKS@0.75           : {oks_global[0.75]:.1f}%")
    logger.info(f"Detection Rate     : {det_rate:.1f}%")
    logger.info(f"Miss Rate          : {miss_rate:.1f}%")
    logger.info(f"MAE — Visible KPs  : {vis_str}")
    logger.info(f"MAE — Occluded KPs : {occ_str}")
    logger.info(f"Pixel MAE          : {global_mae:.2f}px")
    logger.info(f"Pixel RMSE         : {global_rmse:.2f}px")
    logger.info(_SEP)

    plot_global_metrics(
        all_distances,
        all_norm_distances,
        all_oks,
        valid_frames,
        missing_detections,
        oks_global,
        mae_vis,
        mae_occ,
        eval_dir,
    )

    log_pck_normalized_data(all_norm_distances)
    plot_pck_normalized(all_norm_distances, out_dir=eval_dir)

    log_pck_pixel_data(all_distances)
    plot_pck_pixel(all_distances, out_dir=eval_dir)

    log_error_distribution_data(all_distances, all_norm_distances)
    plot_error_histogram_dual(all_distances, all_norm_distances, eval_dir)

    log_cdf_key_points(all_distances, all_norm_distances)
    plot_error_cdf_dual(all_distances, all_norm_distances, eval_dir)

    log_norm_mae_per_keypoint_data(all_norm_distances)
    plot_norm_mae_per_keypoint(all_norm_distances, eval_dir)

    log_mae_per_keypoint_data(all_distances)
    plot_mae_per_keypoint(all_distances, eval_dir)

    log_boxplot_statistics(all_distances)
    plot_boxplot_per_keypoint(all_distances, eval_dir)

    log_pck02_per_keypoint_data(all_norm_distances)
    plot_pck02_per_keypoint(all_norm_distances, eval_dir)

    log_stats_table_data(all_distances, all_norm_distances)
    plot_stats_table(all_distances, all_norm_distances, eval_dir)

    log_group_mae_p75_data(all_distances, all_norm_distances)
    plot_group_mae_p75(all_distances, all_norm_distances, eval_dir)

    plot_group_violin(all_distances, eval_dir)  # no dedicated log block needed

    log_oks_distribution_data(all_oks)
    plot_oks_distribution(all_oks, eval_dir)

    log_oks_per_keypoint_data(all_oks)
    plot_oks_per_keypoint(all_oks, eval_dir)

    log_oks_thresholds_per_keypoint_data(oks_per_kp)
    plot_oks_thresholds_per_keypoint(oks_per_kp, eval_dir)

    logger.success(f"Evaluation complete.  Plots → '{eval_dir}'  |  Log → '{log_path}'")


if __name__ == "__main__":
    main()
