import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

from src.read_datasets import load_dataset_numpy
from src.backends.mediapipe_backend import MediaPipeBackend

COCO_NAMES = [
    "Nose", "L.Eye", "R.Eye", "L.Ear", "R.Ear",
    "L.Shoulder", "R.Shoulder", "L.Elbow", "R.Elbow",
    "L.Wrist", "R.Wrist", "L.Hip", "R.Hip",
    "L.Knee", "R.Knee", "L.Ankle", "R.Ankle"
]

MP_TO_COCO = {
    0: 0, 2: 1, 5: 2, 7: 3, 8: 4, 11: 5, 12: 6, 13: 7, 14: 8,
    15: 9, 16: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
}

BODY_GROUPS = {
    "Head":  [0, 1, 2, 3, 4],
    "Torso": [5, 6, 11, 12],
    "Arms":  [7, 8, 9, 10],
    "Legs":  [13, 14, 15, 16],
}
GROUP_COLORS = {
    "Head":  "#5C85D6",
    "Torso": "#E07B4F",
    "Arms":  "#5BAD72",
    "Legs":  "#B07DC9",
}


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


def mp_to_coco(mp_keypoints):
    coco_kp = np.zeros((17, 2))
    coco_vis = np.zeros(17)
    for mp_idx, coco_idx in MP_TO_COCO.items():
        kp = mp_keypoints[mp_idx]
        coco_kp[coco_idx] = [kp.x, kp.y]
        coco_vis[coco_idx] = kp.visibility
    return coco_kp, coco_vis


def denormalize(kp, w, h):
    kp = kp.copy()
    kp[:, 0] *= w
    kp[:, 1] *= h
    return kp


def compute_distances(gt, pred, vis_mask):
    diff = gt - pred
    dist = np.linalg.norm(diff, axis=1)
    dist[vis_mask == 0] = np.nan
    return dist


def _compute_per_kp_stats(all_distances):
    n_kp = 17
    n_samples = all_distances.shape[0]
    stats = {k: [] for k in ["mae", "median", "p25", "p75", "p95", "max", "valid_pct", "pck20"]}
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
            stats["pck20"].append(np.mean(valid <= 20) * 100)
        else:
            for k in ["mae", "median", "p25", "p75", "p95", "max", "pck20"]:
                stats[k].append(0)
    return stats


def plot_global_metrics(all_distances, thresholds=(10, 20, 30)):
    mae     = np.nanmean(all_distances)
    rmse    = np.sqrt(np.nanmean(all_distances ** 2))
    median  = np.nanmedian(all_distances)
    n_valid = int(np.sum(~np.isnan(all_distances)))

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="#F8F8F8")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Global Metrics", fontsize=13, fontweight="bold", pad=10)

    card_data = [
        ("MAE",             f"{mae:.2f} px",   "#5C85D6"),
        ("RMSE",            f"{rmse:.2f} px",  "#E07B4F"),
        ("Median Error",    f"{median:.2f} px","#5BAD72"),
        ("Valid Keypoints", f"{n_valid:,}",    "#B07DC9"),
    ]
    card_h = 0.17; card_gap = 0.04; y0 = 0.88
    for i, (label, val, color) in enumerate(card_data):
        y = y0 - i * (card_h + card_gap)
        rect = FancyBboxPatch((0.04, y - card_h), 0.92, card_h,
                               boxstyle="round,pad=0.01", linewidth=1.2,
                               edgecolor=color, facecolor=color + "22")
        ax.add_patch(rect)
        ax.text(0.12, y - card_h * 0.35, label, fontsize=9, color="#555555", va="center")
        ax.text(0.88, y - card_h * 0.6, val, fontsize=13, fontweight="bold",
                color=color, ha="right", va="center")

    plt.savefig("01_global_metrics.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '01_global_metrics.png'")
    plt.close()


def plot_pck(all_distances, thresholds=(10, 20, 30)):
    n_valid   = int(np.sum(~np.isnan(all_distances)))
    bar_colors = ["#5C85D6", "#5BAD72", "#E07B4F"]
    pck_vals  = [
        np.sum(np.nan_to_num(all_distances, nan=np.inf) <= t) / n_valid * 100
        for t in thresholds
    ]

    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#F8F8F8")
    bars = ax.bar([f"PCK@{t}px" for t in thresholds], pck_vals,
                   color=bar_colors, edgecolor="white", linewidth=1.2, width=0.5)
    for bar, val in zip(bars, pck_vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Percentage (%)", fontsize=9)
    ax.set_title("PCK — Correct Keypoint Ratio", fontsize=12, fontweight="bold", pad=8)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("02_pck.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '02_pck.png'")
    plt.close()


def plot_error_histogram(all_distances):
    flat_valid = all_distances.flatten()
    flat_valid = flat_valid[~np.isnan(flat_valid)]
    mae    = np.nanmean(all_distances)
    median = np.nanmedian(all_distances)

    fig, ax = plt.subplots(figsize=(8, 4), facecolor="#F8F8F8")
    ax.hist(flat_valid, bins=60, color="#5C85D6", alpha=0.75,
            edgecolor="white", linewidth=0.6, density=True, label="Density")
    ax.axvline(mae,    color="#E07B4F", linewidth=1.8, linestyle="--", label=f"MAE {mae:.1f}px")
    ax.axvline(median, color="#5BAD72", linewidth=1.8, linestyle=":",  label=f"Median {median:.1f}px")
    ax.set_xlabel("Error (pixels)", fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.set_title("Global Error Distribution", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("03_error_histogram.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '03_error_histogram.png'")
    plt.close()


def plot_error_cdf(all_distances, thresholds=(10, 20, 30)):
    flat_valid = all_distances.flatten()
    flat_valid = flat_valid[~np.isnan(flat_valid)]
    bar_colors = ["#5C85D6", "#5BAD72", "#E07B4F"]
    sorted_e   = np.sort(flat_valid)
    cdf        = np.arange(1, len(sorted_e) + 1) / len(sorted_e)

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#F8F8F8")
    ax.plot(sorted_e, cdf * 100, color="#5C85D6", linewidth=2)
    for t, c in zip(thresholds, bar_colors):
        idx = np.searchsorted(sorted_e, t)
        pct = cdf[min(idx, len(cdf) - 1)] * 100
        ax.axvline(t,   color=c, linewidth=1.2, linestyle="--", alpha=0.8)
        ax.axhline(pct, color=c, linewidth=0.8, linestyle=":",  alpha=0.6)
        ax.text(t + 0.5, pct - 3, f"{pct:.0f}%", color=c, fontsize=8)
    ax.set_xlabel("Error threshold (pixels)", fontsize=9)
    ax.set_ylabel("Cumulative percentage (%)", fontsize=9)
    ax.set_title("Error CDF", fontsize=12, fontweight="bold", pad=8)
    ax.set_ylim(0, 103)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("04_error_cdf.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '04_error_cdf.png'")
    plt.close()


def plot_mae_per_keypoint(all_distances):
    stats  = _compute_per_kp_stats(all_distances)
    colors = [_kp_group_color(i) for i in range(17)]
    y_pos  = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F8F8")
    bars = ax.barh(y_pos, stats["mae"], color=colors, edgecolor="white", linewidth=0.8, height=0.7)
    for bar, val in zip(bars, stats["mae"]):
        ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=9)
    ax.set_xlabel("MAE (pixels)", fontsize=10)
    ax.set_title("Mean Absolute Error per Keypoint", fontsize=12, fontweight="bold", pad=8)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")
    for grp, c in GROUP_COLORS.items():
        ax.barh([], [], color=c, label=grp)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    plt.tight_layout()
    plt.savefig("05_mae_per_keypoint.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '05_mae_per_keypoint.png'")
    plt.close()


def plot_boxplot_per_keypoint(all_distances):
    colors = [_kp_group_color(i) for i in range(17)]
    y_pos  = np.arange(17)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F8F8")
    for i in range(17):
        valid = all_distances[:, i]
        valid = valid[~np.isnan(valid)]
        if len(valid) == 0:
            continue
        ax.boxplot(valid, positions=[i], widths=0.6, vert=False,
                   patch_artist=True,
                   medianprops=dict(color="white", linewidth=1.5),
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4),
                   boxprops=dict(facecolor=colors[i], linewidth=0))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Error (pixels)", fontsize=9)
    ax.set_title("Error Distribution per Keypoint (box plot)", fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("06_boxplot_per_keypoint.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '06_boxplot_per_keypoint.png'")
    plt.close()


def plot_pck20_per_keypoint(all_distances):
    stats     = _compute_per_kp_stats(all_distances)
    y_pos     = np.arange(17)
    pck_colors = plt.cm.RdYlGn(np.array(stats["pck20"]) / 100)

    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#F8F8F8")
    bars = ax.barh(y_pos, stats["pck20"], color=pck_colors,
                   edgecolor="white", linewidth=0.6, height=0.7)
    for bar, val in zip(bars, stats["pck20"]):
        ax.text(min(val + 1, 101), bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}%", va="center", fontsize=8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COCO_NAMES, fontsize=8)
    ax.set_xlim(0, 115)
    ax.invert_yaxis()
    ax.set_xlabel("PCK@20px (%)", fontsize=9)
    ax.set_title("PCK@20px per Keypoint  (red = worse, green = better)",
                 fontsize=12, fontweight="bold", pad=8)
    ax.axvline(80, color="#555555", linewidth=1, linestyle="--", alpha=0.6)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("07_pck20_per_keypoint.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '07_pck20_per_keypoint.png'")
    plt.close()


def plot_stats_table(all_distances):
    stats = _compute_per_kp_stats(all_distances)

    col_labels = ["Keypoint", "Group", "MAE", "Median", "P25", "P75", "P95", "Max", "Valid%", "PCK@20px"]
    table_data = []
    for i in range(17):
        table_data.append([
            COCO_NAMES[i],
            _kp_group_name(i),
            f"{stats['mae'][i]:.2f}",
            f"{stats['median'][i]:.2f}",
            f"{stats['p25'][i]:.2f}",
            f"{stats['p75'][i]:.2f}",
            f"{stats['p95'][i]:.2f}",
            f"{stats['max'][i]:.1f}",
            f"{stats['valid_pct'][i]:.1f}%",
            f"{stats['pck20'][i]:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(14, 7), facecolor="#F8F8F8")
    ax.axis("off")
    ax.set_title("Detailed Per-Keypoint Statistics Table",
                 fontsize=13, fontweight="bold", pad=12)

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
        grp_color  = GROUP_COLORS.get(row[1], "#888888")
        base_alpha = "18" if i % 2 == 0 else "30"
        for j in range(len(col_labels)):
            cell = tbl[i + 1, j]
            if j <= 1:
                cell.set_facecolor(grp_color + base_alpha)
                cell.set_text_props(color=grp_color, fontweight="bold")
            else:
                cell.set_facecolor("#F4F4F4" if i % 2 == 0 else "#ECECEC")

    plt.savefig("08_stats_table.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '08_stats_table.png'")
    plt.close()


def plot_group_mae_p75(all_distances):
    group_data = {
        grp: all_distances[:, idx].flatten()[~np.isnan(all_distances[:, idx].flatten())]
        for grp, idx in BODY_GROUPS.items()
    }
    grp_names = list(BODY_GROUPS.keys())
    grp_clrs  = [GROUP_COLORS[g] for g in grp_names]
    grp_mae   = [np.mean(group_data[g]) for g in grp_names]
    grp_p75   = [np.percentile(group_data[g], 75) for g in grp_names]
    x = np.arange(len(grp_names))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#F8F8F8")
    b1 = ax.bar(x - w / 2, grp_mae, width=w, color=grp_clrs, alpha=0.9, edgecolor="white", label="MAE")
    b2 = ax.bar(x + w / 2, grp_p75, width=w, color=grp_clrs, alpha=0.5, edgecolor="white", hatch="//", label="P75")
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, f"{bar.get_height():.1f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(grp_names, fontsize=11)
    ax.set_ylabel("Error (pixels)", fontsize=10)
    ax.set_title("MAE vs P75 by Body Group", fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("09_group_mae_p75.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '09_group_mae_p75.png'")
    plt.close()


def plot_group_violin(all_distances):
    group_data = {
        grp: all_distances[:, idx].flatten()[~np.isnan(all_distances[:, idx].flatten())]
        for grp, idx in BODY_GROUPS.items()
    }
    grp_names = list(BODY_GROUPS.keys())

    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#F8F8F8")
    parts = ax.violinplot(
        [group_data[g] for g in grp_names],
        positions=range(len(grp_names)),
        showmedians=True, showextrema=False
    )
    for pc, grp in zip(parts["bodies"], grp_names):
        pc.set_facecolor(GROUP_COLORS[grp])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("white")
    parts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(grp_names)))
    ax.set_xticklabels(grp_names, fontsize=11)
    ax.set_ylabel("Error (pixels)", fontsize=10)
    ax.set_title("Error Distribution by Body Group", fontsize=12, fontweight="bold", pad=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_facecolor("#FAFAFA")

    plt.tight_layout()
    plt.savefig("10_group_violin.png", dpi=180, bbox_inches="tight")
    print("[INFO] Saved: '10_group_violin.png'")
    plt.close()


def main():
    base_dir = r"H:\\golf_data\\keyframes_yolo2"
    dataset  = load_dataset_numpy(base_dir)

    mp_backend = MediaPipeBackend(static_image_mode=True)
    mp_backend.initialize()

    all_distances      = []
    missing_detections = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        img = cv2.imread(sample["image_path"])
        if img is None:
            continue

        h, w   = img.shape[:2]
        result = mp_backend.process_frame(img)
        if result is None or not getattr(result, "keypoints", None):
            missing_detections += 1
            continue

        pred_kp_norm, _ = mp_to_coco(result.keypoints)
        pred_kp = denormalize(pred_kp_norm, w, h)
        gt_kp   = denormalize(sample["keypoints"], w, h)
        gt_vis  = sample["visible"]

        dist = compute_distances(gt_kp, pred_kp, gt_vis)
        all_distances.append(dist)

    all_distances = np.array(all_distances)

    if len(all_distances) == 0:
        print("No samples evaluated.")
        return

    print("\n" + "=" * 50)
    print(f"Total samples : {len(dataset)}")
    print(f"Missed frames : {missing_detections}")
    print(f"Evaluated     : {len(all_distances)}")
    print(f"Global MAE    : {np.nanmean(all_distances):.2f} px")
    print(f"Global RMSE   : {np.sqrt(np.nanmean(all_distances**2)):.2f} px")
    print("=" * 50)

    plot_global_metrics(all_distances)
    plot_pck(all_distances)
    plot_error_histogram(all_distances)
    plot_error_cdf(all_distances)
    plot_mae_per_keypoint(all_distances)
    plot_boxplot_per_keypoint(all_distances)
    plot_pck20_per_keypoint(all_distances)
    plot_stats_table(all_distances)
    plot_group_mae_p75(all_distances)
    plot_group_violin(all_distances)


if __name__ == "__main__":
    main()