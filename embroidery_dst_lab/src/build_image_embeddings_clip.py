#!/usr/bin/env python3
"""Generate CLIP embeddings for artwork/stitched photos in the embroidery library.

Requires open_clip_torch + torch + pillow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

try:
    import torch
    import open_clip
    from PIL import Image
except ImportError as exc:  # pragma: no cover - guard for missing deps
    raise SystemExit(
        "Dependencies missing. Install with: pip install open_clip_torch torch pillow"
    ) from exc


def list_items(library_root: Path) -> Dict[str, Path]:
    manifests = {}
    for manifest in (library_root / "items").glob("*/manifest.json"):
        manifests[manifest.parent.name] = manifest
    return manifests


def load_manifest(manifest_path: Path) -> Dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_model(model_name: str, pretrained: str, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    return model, preprocess


def compute_clip_embedding(
    image_path: Path, model, preprocess, device: str
) -> list[float]:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        emb = model.encode_image(img_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().float().tolist()


def build_embeddings(
    library_root: Path,
    embeddings_file: Path,
    model_name: str,
    pretrained: str,
) -> Dict[str, Dict[str, Dict]]:
    items = list_items(library_root)
    if not items:
        raise SystemExit(f"No items found under {library_root}/items")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_model(model_name, pretrained, device)
    model_tag = f"clip:{model_name}:{pretrained}"

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
            vec = compute_clip_embedding(img_path, model, preprocess, device)
            entry[key] = {"path": rel_path, "model": model_tag, "embedding": vec}
        if entry:
            embeddings[item_id] = entry

    embeddings_file.write_text(json.dumps(embeddings, indent=2), encoding="utf-8")
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CLIP embeddings for artwork/stitched photos."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--model",
        default="ViT-B-32",
        help="CLIP model name (default: ViT-B-32).",
    )
    parser.add_argument(
        "--pretrained",
        default="openai",
        help="Pretrained weights identifier (default: openai).",
    )
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Output JSON file (default: <library_root>/image_embeddings_clip.json).",
    )
    args = parser.parse_args()

    library_root = Path(args.library_root)
    out_file = (
        Path(args.embeddings_file)
        if args.embeddings_file
        else library_root / "image_embeddings_clip.json"
    )
    embeddings = build_embeddings(library_root, out_file, args.model, args.pretrained)
    print(f"Saved embeddings for {len(embeddings)} item(s) to {out_file}")


if __name__ == "__main__":
    main()
