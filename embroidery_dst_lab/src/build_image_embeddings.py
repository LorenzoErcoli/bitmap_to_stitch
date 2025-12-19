#!/usr/bin/env python3
"""Generate image embeddings for artwork/stitched photos in the embroidery library.

This keeps a lightweight, dependency-free embedding (resize+histogram) so it works offline.
If you later install a vision model (e.g., open_clip), you can swap the embedding backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - guard for missing pillow
    raise SystemExit("Pillow is required: pip install pillow") from exc


def list_items(library_root: Path) -> Dict[str, Path]:
    """Return mapping item_id -> manifest path."""
    manifests = {}
    for manifest in (library_root / "items").glob("*/manifest.json"):
        item_id = manifest.parent.name
        manifests[item_id] = manifest
    return manifests


def load_manifest(manifest_path: Path) -> Dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def _compute_simple_embedding(image_path: Path, size: int = 64) -> List[float]:
    """Coarse embedding: resize, per-channel histogram, grayscale downsample."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize((size, size))
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H, W, 3)

    # RGB histograms with 8 bins per channel (24 dims)
    hist = []
    bins = np.linspace(0, 1, 9)
    for c in range(3):
        h, _ = np.histogram(arr[..., c], bins=bins, density=True)
        hist.append(h)
    hist_vec = np.concatenate(hist)

    # Mean/std per channel (6 dims)
    mean_std = np.concatenate([arr.mean(axis=(0, 1)), arr.std(axis=(0, 1))])

    # Grayscale downsample to 16x16 (256 dims)
    gray = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    small = np.array(
        Image.fromarray((gray * 255).astype(np.uint8)).resize((16, 16), resample=Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    small_vec = small.flatten()

    vec = np.concatenate([hist_vec, mean_std, small_vec])
    return _l2_normalize(vec).astype(np.float32).tolist()


def compute_embedding(image_path: Path, model: str = "simple") -> Tuple[str, List[float]]:
    model = model.lower()
    if model != "simple":
        raise SystemExit(f"Unsupported model '{model}'. Only 'simple' is available offline.")
    return "simple_v1", _compute_simple_embedding(image_path)


def build_embeddings(
    library_root: Path, model: str = "simple", overwrite: bool = False
) -> Dict[str, Dict[str, Dict]]:
    items = list_items(library_root)
    if not items:
        raise SystemExit(f"No items found under {library_root}/items")

    embeddings: Dict[str, Dict[str, Dict]] = {}
    for item_id, manifest_path in items.items():
        manifest = load_manifest(manifest_path)
        paths = manifest.get("paths", {})
        entry: Dict[str, Dict] = {}

        for key in ("artwork", "stitched_photo"):
            rel_path = paths.get(key)
            if not rel_path:
                continue
            img_path = (library_root / rel_path).resolve()
            if not img_path.exists():
                continue
            model_name, vec = compute_embedding(img_path, model=model)
            entry[key] = {
                "path": rel_path,
                "model": model_name,
                "embedding": vec,
            }

        if entry:
            embeddings[item_id] = entry

    return embeddings


def save_embeddings(library_root: Path, payload: Dict[str, Dict[str, Dict]]) -> Path:
    out_path = library_root / "image_embeddings.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate embeddings for artwork/stitched photos in the embroidery library."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--model",
        default="simple",
        help="Embedding model; 'simple' is offline-friendly.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignored for now (placeholder for future per-item updates).",
    )
    args = parser.parse_args()

    library_root = Path(args.library_root)
    embeddings = build_embeddings(library_root, model=args.model, overwrite=args.overwrite)
    if not embeddings:
        print("No artwork/stitched_photo assets found; nothing to embed.")
        return
    out_path = save_embeddings(library_root, embeddings)
    print(f"Saved embeddings for {len(embeddings)} item(s) to {out_path}")


if __name__ == "__main__":
    main()
