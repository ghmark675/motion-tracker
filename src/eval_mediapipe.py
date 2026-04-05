import os
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from src.read_datasets import load_dataset_numpy
from src.backends.mediapipe_backend import MediaPipeBackend

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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
    """Per-keypoint Euclidean distance in pixels. NaN for invisible keypoints."""
    dist = np.linalg.norm(gt - pred, axis=1)
    dist[vis_mask == 0] = np.nan
    return dist


def compute_bbox_scale(gt_kp_px, vis_mask):
    """
    Bounding-box scale = sqrt(bbox_w * bbox_h) from visible GT keypoints.

    Returns 0.0 if fewer than 2 keypoints are visible (scale undefined).
    Used as fallback when torso landmarks are not available.
    """
    visible = gt_kp_px[vis_mask > 0]
    if len(visible) < 2:
        return 0.0
    x_min, y_min = visible.min(axis=0)
    x_max, y_max = visible.max(axis=0)
    scale = np.sqrt((x_max - x_min) * (y_max - y_min))
    return float(scale) if scale > 0 else 0.0


def compute_torso_scale(gt_kp_px, vis_mask):
    """
    Torso scale = mean length of shoulder→opposite-hip diagonals.

    Uses pairs (L.Shoulder→R.Hip) and (R.Shoulder→L.Hip).  At least one pair
    must have both landmarks visible; otherwise returns 0.0.

    Preferred over bbox scale because it is body-proportional regardless of
    which body parts are in frame.  A person cropped at the waist still gives a
    meaningful torso scale, whereas the bbox would shrink to the upper-body
    extent and inflate the normalised error.
    """
    lengths = []
    for a, b in TORSO_PAIRS:
        if vis_mask[a] > 0 and vis_mask[b] > 0:
            lengths.append(np.linalg.norm(gt_kp_px[a] - gt_kp_px[b]))
    return float(np.mean(lengths)) if lengths else 0.0


def compute_scale(gt_kp_px, vis_mask):
    """
    Dual-scale strategy (torso preferred, bbox as fallback).

    1. Try torso scale (shoulder↔opposite-hip diagonal).
    2. If torso keypoints are occluded, fall back to bbox scale.
    3. If neither is available, return 0.0 (frame will be excluded from
       normalised metrics).

    This prevents partial-visibility frames from inflating normalised error:
    when only the upper body is visible the GT bbox is small, which would
    incorrectly amplify every normalised distance by a large factor.
    """
    scale = compute_torso_scale(gt_kp_px, vis_mask)
    if scale <= 0:
        scale = compute_bbox_scale(gt_kp_px, vis_mask)
    return scale


def compute_normalized_distances(dist_px, scale):
    """
    Divide pixel distances by a body-proportional scale.

    PCK@0.2 definition: keypoint is *correct* when
        dist_px / scale  <  0.2
    i.e. the error is less than 20 % of sqrt(bbox_w × bbox_h) or the
    torso diagonal — whichever scale was used.

    Returns NaN for entries where dist_px is already NaN (invisible keypoints)
    or where scale == 0 (scale could not be computed for this frame).
    """
    if scale <= 0:
        return np.full_like(dist_px, np.nan)
    return dist_px / scale


def compute_oks(dist_px, scale, vis_mask, sigmas=COCO_KP_SIGMAS):
    """
    Per-keypoint Object Keypoint Similarity (simplified COCO formulation):

        OKS_i = exp( -d_i² / (2 · (scale · σ_i)²) )

    where σ_i are the per-keypoint constants from the COCO paper (Table 1),
    encoding how much natural variation exists for each landmark type.
    Harder-to-localise joints (wrists, ankles) have larger σ, so the same
    pixel error contributes a smaller OKS penalty.

    Returns NaN for invisible or invalid keypoints.
    Returns an all-NaN array if scale == 0.
    """
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
# Shared PCK helper  (eliminates duplicate threshold-loop logic)
# ---------------------------------------------------------------------------


def _pck(distances_flat, threshold):
    """
    Fraction (0–100) of valid keypoints whose distance ≤ threshold.

    Parameters
    ----------
    distances_flat : 1-D array, may contain NaN (invisible / no-scale frames)
    threshold      : scalar — pixel value OR normalised fraction depending on
                     which distance array is passed

    Returns 0.0 if no valid entries exist (numerical safety).
    """
    valid = distances_flat[~np.isnan(distances_flat)]
    if len(valid) == 0:
        return 0.0
    return float(np.mean(valid <= threshold) * 100)


# ---------------------------------------------------------------------------
# OKS threshold metrics  (OKS@0.5 and OKS@0.75)
# ---------------------------------------------------------------------------


def compute_oks_thresholds(all_oks, thresholds=(0.5, 0.75)):
    """
    OKS@t = percentage of valid keypoints with OKS > t.

    This mirrors AP@[.5:.95] used in COCO evaluation and provides a
    meaningful difficulty gradient: OKS@0.5 is a loose correctness criterion
    (roughly comparable to a 50 % match), while OKS@0.75 is strict.

    Parameters
    ----------
    all_oks     : (N, 17) array, NaN for invisible / no-scale keypoints
    thresholds  : iterable of OKS thresholds to evaluate

    Returns
    -------
    dict  {threshold: percentage}   e.g. {0.5: 84.3, 0.75: 61.7}
    Also returns per-keypoint breakdown as a second dict.
    """
    flat = all_oks.flatten()
    valid = flat[~np.isnan(flat)]

    global_oks_t = {}
    for t in thresholds:
        if len(valid) == 0:
            global_oks_t[t] = 0.0
        else:
            global_oks_t[t] = float(np.mean(valid > t) * 100)

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
    """
    Split MAE into visible vs occluded keypoints.

    Assumptions
    -----------
    * A keypoint is *visible*   if gt_vis > 0.5  (GT visibility flag set)
    * A keypoint is *occluded*  if gt_vis == 0    (GT visibility flag unset)
    * NaN entries in all_distances_px mean the keypoint was invisible and
      was already excluded from pixel-distance computation; they are skipped.

    Note: If the dataset only annotates fully-visible keypoints (gt_vis is
    always 1 for labelled joints), the occluded MAE will be NaN, which is
    reported explicitly rather than silently omitted.

    Parameters
    ----------
    all_distances_px : (N, 17) pixel-distance array (NaN = invisible)
    all_gt_vis       : (N, 17) visibility flags in {0, 1}  (or float 0/1)

    Returns
    -------
    mae_visible   : float or NaN
    mae_occluded  : float or NaN
    n_visible     : int — number of valid visible keypoint observations
    n_occluded    : int — number of valid occluded keypoint observations
    """
    vis_mask = all_gt_vis > 0.5  # (N,17) bool
    occ_mask = all_gt_vis == 0  # (N,17) bool
    valid = ~np.isnan(all_distances_px)  # (N,17) bool

    vis_dists = all_distances_px[vis_mask & valid]
    occ_dists = all_distances_px[occ_mask & valid]

    mae_visible = float(np.mean(vis_dists)) if len(vis_dists) > 0 else np.nan
    mae_occluded = float(np.mean(occ_dists)) if len(occ_dists) > 0 else np.nan

    return mae_visible, mae_occluded, len(vis_dists), len(occ_dists)


# ---------------------------------------------------------------------------
# Per-keypoint statistics (extended to handle normalised distances)
# ---------------------------------------------------------------------------


def _compute_per_kp_stats(all_distances, all_norm_distances=None):
    """
    Compute per-keypoint statistics for both pixel and (optionally) normalised
    distances.  Returns a dict with list-of-17 entries.
    """
    n_kp = 17
    n_samples = all_distances.shape[0]
    keys = ["mae", "median", "p25", "p75", "p95", "max", "valid_pct", "pck20px"]
    stats = {k: [] for k in keys}

    # Normalised stats (if provided)
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
    print(f"[INFO] Saved: '{path}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 01 — Global Metrics (extended with normalised + detection rate + OKS)
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
    """
    Summary card panel.  Normalised metrics are PRIMARY; pixel metrics secondary.

    Parameters
    ----------
    valid_frames          : frames where the image loaded successfully
    missing_detections    : frames where the model returned no pose
    oks_thresholds_global : dict {0.5: pct, 0.75: pct} from compute_oks_thresholds
    mae_visible / mae_occluded : from compute_visibility_mae
    """
    mae = np.nanmean(all_distances)
    rmse = np.sqrt(np.nanmean(all_distances**2))
    norm_mae = np.nanmean(all_norm_distances)
    norm_rmse = np.sqrt(np.nanmean(all_norm_distances**2))
    mean_oks = np.nanmean(all_oks)

    n_evaluated = len(all_distances)
    # Denominator is valid_frames (images that loaded OK), not total_frames.
    # Unreadable image files are neither detected nor missed — they are
    # excluded from the denominator to avoid penalising the model for I/O errors.
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
        # (label, value, colour, is_separator)
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
    print(f"[INFO] Saved: '{os.path.join(out_dir, '01_global_metrics.png')}'")
    plt.close()


# ---------------------------------------------------------------------------
# Plot 02 — PCK (normalised thresholds — PRIMARY)
# ---------------------------------------------------------------------------


def plot_pck_normalized(
    all_norm_distances, thresholds=(0.05, 0.1, 0.15, 0.2, 0.25, 0.3), out_dir="."
):
    """
    PCK bar chart using normalised distances.

    PCK@t definition: keypoint is *correct* iff
        dist_px / scale  <  t
    where scale = torso scale (preferred) or bbox scale (fallback).

    PCK@0.1 and PCK@0.2 are the headline metrics, matching Yang & Ramanan
    (2013) and the LSP / MPII benchmark protocols.
    """
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
# Plot 02b — PCK@20px (pixel, kept as supplementary reference)
# ---------------------------------------------------------------------------


def plot_pck_pixel(all_distances, thresholds=(10, 20, 30), out_dir="."):
    """Pixel-based PCK kept as supplementary only."""
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
        "PCK (Pixel) — SUPPLEMENTARY ONLY\n"
        "Not scale-invariant; do not use as primary metric",
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
    """
    Side-by-side histograms so reviewers can directly compare pixel vs
    normalised distributions.
    """
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
    stats = _compute_per_kp_stats(all_norm_distances)  # reuse generic helper
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
    for grp, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=grp)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    _save("05_norm_mae_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 05b — Pixel MAE per keypoint (supplementary, unchanged from v1)
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
    for grp, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=grp)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    _save("05b_pixel_mae_per_keypoint_supplementary.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 06 — Boxplot per keypoint (unchanged from v1)
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
# Plot 07 — PCK@0.2 per keypoint (normalised, PRIMARY; replaces PCK@20px)
# ---------------------------------------------------------------------------


def plot_pck02_per_keypoint(all_norm_distances, out_dir="."):
    """
    PCK@0.2 using normalised distances.  A keypoint is correct if its
    normalised error < 0.2.  This replaces the old PCK@20px plot.
    """
    stats = _compute_per_kp_stats(all_norm_distances)
    y_pos = np.arange(17)

    # Use pck "pck20px" slot which now holds pck@0.2 because we pass norm distances
    # Recompute explicitly for clarity
    pck02 = []
    for i in range(17):
        col = all_norm_distances[:, i]
        valid = col[~np.isnan(col)]
        pck02.append(np.mean(valid <= 0.2) * 100 if len(valid) > 0 else 0.0)

    pck_colors = plt.cm.RdYlGn(np.array(pck02) / 100)

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
# Plot 08 — Extended stats table (normalised + pixel)
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
            elif j <= 4:  # normalised primary columns — slight highlight
                cell.set_facecolor("#EDF3FC" if i % 2 == 0 else "#DDE9F7")
            else:
                cell.set_facecolor("#F4F4F4" if i % 2 == 0 else "#ECECEC")

    _save("08_stats_table.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 09 — Group MAE P75 (extended with normalised version)
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
# Plot 10 — Group violin (unchanged, pixel)
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
    for grp, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=grp)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    _save("12_oks_per_keypoint.png", out_dir)


# ---------------------------------------------------------------------------
# Plot 13 — OKS@0.5 and OKS@0.75 per keypoint
# ---------------------------------------------------------------------------


def plot_oks_thresholds_per_keypoint(per_kp_oks_t, out_dir="."):
    """
    Side-by-side bar charts showing OKS@0.5 and OKS@0.75 per keypoint.

    OKS@t = % of valid keypoints with OKS > t.
    OKS@0.5  is the standard "easy" threshold (COCO AP@.50).
    OKS@0.75 is the "strict" threshold (COCO AP@.75).
    Comparing the two reveals which keypoints degrade most under tight criteria.
    """
    thresholds = sorted(per_kp_oks_t.keys())
    y_pos = np.arange(17)
    colors = [_kp_group_color(i) for i in range(17)]

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
# Main
# ---------------------------------------------------------------------------


def main():
    base_dir = r"H:\\golf_data\\keyframes_yolo2"

    # ---- 创建带时间戳的评测结果文件夹 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("eval_results", f"mediapipe_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: '{out_dir}'")

    dataset = load_dataset_numpy(base_dir)
    total_frames = len(dataset)

    mp_backend = MediaPipeBackend(static_image_mode=True)
    mp_backend.initialize()

    all_distances = []
    all_norm_distances = []
    all_oks = []
    all_gt_vis = []  # needed for visibility-aware MAE

    invalid_frames = 0  # images that failed to load (I/O errors)
    missing_detections = 0  # images that loaded but yielded no pose

    for sample in tqdm(dataset, desc="Evaluating"):
        img = cv2.imread(sample["image_path"])
        if img is None:
            # Image unreadable — exclude from denominator entirely.
            # Counting these as misses would penalise the model for disk/path
            # errors, which is incorrect.
            invalid_frames += 1
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

        # --- pixel distances (NaN for invisible keypoints) ---
        dist = compute_distances(gt_kp, pred_kp, gt_vis)

        # --- dual-scale: torso preferred, bbox as fallback ---
        scale = compute_scale(gt_kp, gt_vis)

        # --- normalised distances and OKS (NaN when scale == 0) ---
        norm_dist = compute_normalized_distances(dist, scale)
        oks = compute_oks(dist, scale, gt_vis)

        all_distances.append(dist)
        all_norm_distances.append(norm_dist)
        all_oks.append(oks)
        all_gt_vis.append(gt_vis)

    if len(all_distances) == 0:
        print("No samples evaluated.")
        return

    all_distances = np.array(all_distances)  # (N, 17)
    all_norm_distances = np.array(all_norm_distances)  # (N, 17)
    all_oks = np.array(all_oks)  # (N, 17)
    all_gt_vis = np.array(all_gt_vis)  # (N, 17)

    # valid_frames = images that loaded OK (the correct denominator for
    # detection rate — excludes unreadable files which are an I/O problem,
    # not a model failure)
    valid_frames = total_frames - invalid_frames
    n_evaluated = len(all_distances)
    det_rate = n_evaluated / valid_frames * 100 if valid_frames > 0 else 0.0
    miss_rate = missing_detections / valid_frames * 100 if valid_frames > 0 else 0.0

    # --- primary normalised metrics ---
    global_nmae = np.nanmean(all_norm_distances)
    global_nrmse = np.sqrt(np.nanmean(all_norm_distances**2))
    flat_norm = all_norm_distances.flatten()
    pck01 = _pck(flat_norm, 0.1)
    pck02 = _pck(flat_norm, 0.2)

    # --- OKS metrics ---
    mean_oks = np.nanmean(all_oks)
    oks_global, oks_per_kp = compute_oks_thresholds(all_oks, thresholds=(0.50, 0.75))

    # --- visibility-aware MAE ---
    mae_vis, mae_occ, n_vis, n_occ = compute_visibility_mae(all_distances, all_gt_vis)

    # --- supplementary pixel metrics ---
    global_mae = np.nanmean(all_distances)
    global_rmse = np.sqrt(np.nanmean(all_distances**2))

    # ---- console summary ----
    print("\n" + "=" * 60)
    print(f"  Total frames       : {total_frames}")
    print(f"  Invalid (I/O err)  : {invalid_frames}")
    print(f"  Valid frames       : {valid_frames}")
    print(f"  Missing detections : {missing_detections}  ({miss_rate:.1f}%)")
    print(f"  Detected           : {n_evaluated}  ({det_rate:.1f}%)")
    print(f"  --- PRIMARY (Normalised) ---")
    print(f"  Normalised MAE     : {global_nmae:.4f}")
    print(f"  Normalised RMSE    : {global_nrmse:.4f}")
    print(f"  PCK@0.1  (norm)    : {pck01:.1f}%")
    print(f"  PCK@0.2  (norm)    : {pck02:.1f}%")
    print(f"  Mean OKS           : {mean_oks:.4f}")
    print(f"  OKS@0.50           : {oks_global[0.50]:.1f}%")
    print(f"  OKS@0.75           : {oks_global[0.75]:.1f}%")
    print(f"  --- Visibility Split (Pixel MAE) ---")
    vis_str = f"{mae_vis:.2f} px  (n={n_vis})" if not np.isnan(mae_vis) else "N/A"
    occ_str = (
        f"{mae_occ:.2f} px  (n={n_occ})"
        if not np.isnan(mae_occ)
        else "N/A (no occluded GT)"
    )
    print(f"  MAE — Visible      : {vis_str}")
    print(f"  MAE — Occluded     : {occ_str}")
    print(f"  --- SUPPLEMENTARY (Pixel) ---")
    print(f"  Pixel MAE          : {global_mae:.2f} px")
    print(f"  Pixel RMSE         : {global_rmse:.2f} px")
    print("=" * 60)
    print(f"  Results saved to   : '{out_dir}'")
    print("=" * 60)

    # ---------- plots ----------
    plot_global_metrics(
        all_distances,
        all_norm_distances,
        all_oks,
        valid_frames,
        missing_detections,
        oks_global,
        mae_vis,
        mae_occ,
        out_dir,
    )
    plot_pck_normalized(all_norm_distances, out_dir=out_dir)
    plot_pck_pixel(all_distances, out_dir=out_dir)
    plot_error_histogram_dual(all_distances, all_norm_distances, out_dir)
    plot_error_cdf_dual(all_distances, all_norm_distances, out_dir)
    plot_norm_mae_per_keypoint(all_norm_distances, out_dir)
    plot_mae_per_keypoint(all_distances, out_dir)
    plot_boxplot_per_keypoint(all_distances, out_dir)
    plot_pck02_per_keypoint(all_norm_distances, out_dir)
    plot_stats_table(all_distances, all_norm_distances, out_dir)
    plot_group_mae_p75(all_distances, all_norm_distances, out_dir)
    plot_group_violin(all_distances, out_dir)
    plot_oks_distribution(all_oks, out_dir)
    plot_oks_per_keypoint(all_oks, out_dir)
    plot_oks_thresholds_per_keypoint(oks_per_kp, out_dir)


if __name__ == "__main__":
    main()
