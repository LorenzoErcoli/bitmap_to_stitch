#!/usr/bin/env python3
"""Segment an image into k color regions and export masks + metadata.

Lightweight k-means color quantization (no scikit deps). Saves one mask per cluster
(white foreground on black) and a JSON summary with bbox, area and mean color.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def load_image(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)


def initialize_centroids(pixels: np.ndarray, k: int, seed: int | None = None) -> np.ndarray:
    rng = random.Random(seed)
    idx = rng.sample(range(len(pixels)), k)
    return pixels[idx]


def kmeans(
    pixels: np.ndarray,
    k: int,
    max_iter: int = 15,
    tol: float = 1e-4,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (labels, centroids)."""
    centroids = initialize_centroids(pixels, k, seed=seed)
    labels = np.zeros(len(pixels), dtype=np.int32)
    for _ in range(max_iter):
        # assign
        dists = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        # recompute
        new_centroids = []
        for i in range(k):
            members = pixels[new_labels == i]
            if len(members) == 0:
                new_centroids.append(centroids[i])
            else:
                new_centroids.append(members.mean(axis=0))
        new_centroids = np.stack(new_centroids)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        labels = new_labels
        if shift < tol:
            break
    return labels, centroids


def mask_stats(mask: np.ndarray) -> Dict:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return {
            "area_px": 0,
            "area_pct": 0.0,
            "bbox": None,
        }
    bbox = {"minx": int(xs.min()), "miny": int(ys.min()), "maxx": int(xs.max()), "maxy": int(ys.max())}
    return {
        "area_px": int(mask.sum()),
        "bbox": bbox,
    }


def save_mask(mask: np.ndarray, path: Path) -> None:
    img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment image into k color clusters and export masks + JSON.")
    parser.add_argument("--image", required=True, help="Input image.")
    parser.add_argument("--k", type=int, default=4, help="Number of clusters (default: 4).")
    parser.add_argument(
        "--min-area-pct",
        type=float,
        default=0.5,
        help="Discard clusters smaller than this percentage of pixels (default 0.5%%).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=15,
        help="Max iterations for k-means (default: 15).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--output-dir",
        default="embroidery_dst_lab/output/segments",
        help="Output folder for masks and metadata.",
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")
    output_dir = Path(args.output_dir)
    masks_dir = output_dir / "masks"

    rgb = load_image(img_path)
    h, w, _ = rgb.shape
    pixels = rgb.reshape(-1, 3)
    labels, centroids = kmeans(pixels, k=args.k, max_iter=args.max_iter, seed=args.seed)

    total_px = h * w
    min_area_px = total_px * (args.min_area_pct / 100.0)

    metadata = {
        "image": str(img_path),
        "k": args.k,
        "min_area_pct": args.min_area_pct,
        "width": w,
        "height": h,
        "regions": [],
    }

    for idx in range(args.k):
        mask = (labels.reshape(h, w) == idx).astype(np.uint8)
        stats = mask_stats(mask)
        area_px = stats["area_px"]
        area_pct = (area_px / total_px * 100.0) if total_px else 0.0
        if area_px < min_area_px:
            continue
        mean_color = centroids[idx]
        mask_path = masks_dir / f"region_{idx}.png"
        save_mask(mask, mask_path)
        region_entry = {
            "id": idx,
            "area_px": area_px,
            "area_pct": area_pct,
            "bbox": stats["bbox"],
            "mean_color_rgb": [float(x) for x in mean_color],
            "mask": str(mask_path),
        }
        metadata["regions"].append(region_entry)

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "segments.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved {len(metadata['regions'])} region mask(s) to {masks_dir}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
