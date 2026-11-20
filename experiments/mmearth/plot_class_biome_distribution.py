#!/usr/bin/env python3
"""
Publication Figure: Biome and Class Distribution Comparison
===========================================================
Compares the 2 available MMEarth-20k dataset variants:
  1. Original MMEarth-20K (WACV Fixed / Natural Distribution, N=20,000)
  2. Balanced MMEarth-20K (Curated 14-Biome Equal Quota, N=19,992)

Outputs:
  - assets/mmearth_20k_class_biome_distribution.png
  - assets/mmearth_20k_class_biome_distribution.pdf
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

REPO_DIR = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_DIR / "temp/scratch/class_biome_3datasets.json"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

biome_names = [
    "Tropical Moist Broadleaf",
    "Tropical Dry Broadleaf",
    "Tropical Coniferous",
    "Temperate Broadleaf/Mixed",
    "Temperate Coniferous",
    "Boreal Forests / Taiga",
    "Tropical Savannas",
    "Temperate Grasslands",
    "Flooded Grasslands",
    "Montane Grasslands",
    "Tundra",
    "Mediterranean Scrub",
    "Deserts & Xeric",
    "Mangroves",
]

classes = [
    "Water",
    "Trees",
    "Grass",
    "Flooded Veg",
    "Crops",
    "Shrub/Scrub",
    "Built",
    "Bare",
    "Snow/Ice",
]

# Official Dynamic World Color Palette
dw_colors = [
    "#419BDF", # Water
    "#397D49", # Trees
    "#88B053", # Grass
    "#7A87C6", # Flooded Veg
    "#E49635", # Crops
    "#DFC35A", # Shrub/Scrub
    "#C4281B", # Built
    "#A59B8F", # Bare
    "#B39FE1", # Snow/Ice
]

# The 2 available datasets
dataset_keys = ["Fixed (WACV 20k)", "Balanced 20k"]
titles = [
    "Original MMEarth-20k (WACV Fixed)",
    "Curated Balanced MMEarth-20k (Equal Quota)",
]

fig = plt.figure(figsize=(17, 13), dpi=300)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 2.3], hspace=0.28, wspace=0.20)

x = np.arange(len(biome_names))

for col_idx, (d_key, title) in enumerate(zip(dataset_keys, titles)):
    d = data[d_key]
    tile_counts = np.array(d["tile_counts"])
    pixel_hist = np.array(d["pixel_hist"]) # (14, 9)

    # ── Row 0: Tile Count Histogram ───────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, col_idx])
    bars = ax_top.bar(x, tile_counts, color="#2C3E50" if col_idx == 0 else "#16A085",
                      edgecolor="#1A252F", width=0.72, alpha=0.9)
    ax_top.set_title(f"{title}\nTotal Sampled Tiles: {sum(tile_counts):,}", fontsize=13, fontweight="bold", pad=10)
    ax_top.set_ylabel("Number of Tiles", fontsize=11, fontweight="semibold")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels([f"B{i}" for i in range(14)], fontsize=10)
    ax_top.grid(axis="y", linestyle="--", alpha=0.5)

    min_c, max_c = min(tile_counts), max(tile_counts)
    if min_c == max_c:
        ax_top.set_ylim(0, max_c * 1.35)
        ax_top.axhline(max_c, color="#27AE60", linestyle=":", linewidth=2, label=f"Uniform Quota: {max_c:,} tiles / biome")
        ax_top.legend(loc="upper right", fontsize=10, framealpha=0.9)
    else:
        ax_top.set_ylim(0, max_c * 1.30)
        ax_top.text(
            0.03, 0.85,
            f"Sampling Spread: {min_c:,} – {max_c:,} tiles\nGeographic Skew Ratio: {max_c/min_c:.1f}x",
            transform=ax_top.transAxes,
            fontsize=10,
            fontweight="medium",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FDEDEC", ec="#E74C3C", alpha=0.95),
        )

    # ── Row 1: Normalized Class Distribution Stacked Bar Chart per Biome ──────
    ax_bottom = fig.add_subplot(gs[1, col_idx])

    row_sums = pixel_hist.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    pct_hist = (pixel_hist / row_sums) * 100.0 # (14, 9)

    y_pos = np.arange(len(biome_names))[::-1] # 0 at top

    left = np.zeros(len(biome_names))
    for c_idx in range(9):
        c_vals = pct_hist[:, c_idx]
        ax_bottom.barh(
            y_pos,
            c_vals,
            left=left,
            color=dw_colors[c_idx],
            edgecolor="none",
            height=0.75,
            label=classes[c_idx] if col_idx == 0 else "",
        )
        left += c_vals

    ax_bottom.set_xlim(0, 100)
    ax_bottom.set_xlabel("Pixel Class Share (%)", fontsize=11, fontweight="semibold")
    ax_bottom.set_yticks(y_pos)
    if col_idx == 0:
        ax_bottom.set_yticklabels(
            [f"B{i}: {name}" for i, name in enumerate(biome_names)],
            fontsize=10,
            fontweight="medium",
        )
    else:
        ax_bottom.set_yticklabels([f"B{i}" for i in range(14)], fontsize=10)

    ax_bottom.set_title(f"Class-Wise Pixel Composition (%) Across 14 Biomes", fontsize=11, fontweight="semibold", pad=8)
    ax_bottom.grid(axis="x", linestyle="--", alpha=0.5)

# Shared Legend for classes
handles, labels = fig.axes[2].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=9,
    fontsize=11,
    frameon=True,
    bbox_to_anchor=(0.5, 0.02),
    title="Dynamic World Semantic Land-Cover Classes",
    title_fontsize=12,
)

out_png = REPO_DIR / "assets/mmearth_20k_class_biome_distribution.png"
out_pdf = REPO_DIR / "assets/mmearth_20k_class_biome_distribution.pdf"
out_png.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(out_png, bbox_inches="tight", dpi=300)
plt.savefig(out_pdf, bbox_inches="tight")
plt.close()

print(f"2-Dataset Distribution figure successfully created:")
print(f"  PNG: {out_png}")
print(f"  PDF: {out_pdf}")

