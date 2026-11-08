# scripts/dataset_audit.py
import os
import hashlib
import csv
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/raw")
CLASSES    = ["AiArtData", "RealArt"]
OUTPUT_CSV = Path("scripts/audit_results.csv")
FIGURES    = Path("scripts/figures")
FIGURES.mkdir(parents=True, exist_ok=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def audit_image(path: Path, label: str) -> dict:
    record = {
        "class":      label,
        "filename":   path.name,
        "filepath":   str(path),
        "file_size":  path.stat().st_size,
        "width":      None,
        "height":     None,
        "color_mode": None,
        "format":     None,
        "hash":       None,
        "corrupt":    False,
    }
    try:
        with Image.open(path) as img:
            record["width"]      = img.width
            record["height"]     = img.height
            record["color_mode"] = img.mode
            record["format"]     = img.format
            record["hash"]       = sha256(path)
    except Exception:
        record["corrupt"] = True
    return record

# ── Walk dataset ──────────────────────────────────────────────────────────────
print("Auditing dataset ...")
records = []

for label in CLASSES:
    class_dir = DATA_DIR / label
    files = list(class_dir.rglob("*"))
    image_files = [
        f for f in files
        if f.is_file() and f.suffix.lower() in
           {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
    ]
    print(f"  {label}: {len(image_files)} images found")
    for path in image_files:
        records.append(audit_image(path, label))

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)
print(f"\nAudit CSV saved to {OUTPUT_CSV}")

# ── Summary statistics ────────────────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

corrupt   = [r for r in records if r["corrupt"]]
clean     = [r for r in records if not r["corrupt"]]

# Count per class
class_counts = defaultdict(int)
for r in records:
    class_counts[r["class"]] += 1

print(f"\nTotal images   : {len(records)}")
print(f"Corrupt files  : {len(corrupt)}")
for label, count in class_counts.items():
    print(f"  {label:15s}: {count}")

# Duplicate detection
hashes = [r["hash"] for r in clean if r["hash"]]
unique_hashes   = set(hashes)
duplicate_count = len(hashes) - len(unique_hashes)
print(f"\nExact duplicates (by SHA-256): {duplicate_count}")

# Per-class stats for width, height, file size
for label in CLASSES:
    subset = [r for r in clean if r["class"] == label]
    widths  = [r["width"]     for r in subset if r["width"]]
    heights = [r["height"]    for r in subset if r["height"]]
    sizes   = [r["file_size"] for r in subset]

    print(f"\n--- {label} ---")
    print(f"  Width  : min={min(widths):>6}  median={int(np.median(widths)):>6}  max={max(widths):>6}")
    print(f"  Height : min={min(heights):>6}  median={int(np.median(heights)):>6}  max={max(heights):>6}")
    print(f"  Size(B): min={min(sizes):>8}  median={int(np.median(sizes)):>8}  max={max(sizes):>8}")

# ── Task 1.3 EDA Figures ──────────────────────────────────────────────────────
print("\nGenerating EDA figures ...")

# 1. Sample image grid per class
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle("Sample Images per Class", fontsize=14)

import random
random.seed(42)

for row, label in enumerate(CLASSES):
    subset = [r for r in clean if r["class"] == label]
    samples = random.sample(subset, min(5, len(subset)))
    for col, record in enumerate(samples):
        img = Image.open(record["filepath"]).convert("RGB")
        axes[row, col].imshow(img)
        axes[row, col].axis("off")
        axes[row, col].set_title(label, fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES / "sample_grid.png", dpi=150)
plt.close()
print("  Saved sample_grid.png")

# 2. Width histogram per class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label in zip(axes, CLASSES):
    subset = [r for r in clean if r["class"] == label]
    widths = [r["width"] for r in subset if r["width"]]
    ax.hist(widths, bins=40, color="#4C8BF5", edgecolor="white")
    ax.set_title(f"Width Distribution -- {label}")
    ax.set_xlabel("Width (px)")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES / "width_hist.png", dpi=150)
plt.close()
print("  Saved width_hist.png")

# 3. Height histogram per class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label in zip(axes, CLASSES):
    subset = [r for r in clean if r["class"] == label]
    heights = [r["height"] for r in subset if r["height"]]
    ax.hist(heights, bins=40, color="#F4A261", edgecolor="white")
    ax.set_title(f"Height Distribution -- {label}")
    ax.set_xlabel("Height (px)")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES / "height_hist.png", dpi=150)
plt.close()
print("  Saved height_hist.png")

# 4. File size histogram per class
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, label in zip(axes, CLASSES):
    subset = [r for r in clean if r["class"] == label]
    sizes = [r["file_size"] / 1024 for r in subset]  # KB
    ax.hist(sizes, bins=40, color="#2A9D8F", edgecolor="white")
    ax.set_title(f"File Size Distribution -- {label}")
    ax.set_xlabel("File Size (KB)")
    ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(FIGURES / "filesize_hist.png", dpi=150)
plt.close()
print("  Saved filesize_hist.png")

print("\nDone. All figures saved to scripts/figures/")
