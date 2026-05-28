#!/usr/bin/env python3
"""Per-region retrieval: use segment masks to query the library via image embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from build_image_embeddings import compute_embedding as compute_simple_embedding


def load_embeddings(path: Path) -> Dict[str, Dict[str, Dict]]:
    if not path.exists():
        raise SystemExit(f"Embeddings file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_clip_model(model_str: str) -> Tuple[str, str] | None:
    if not model_str.startswith("clip:"):
        return None
    parts = model_str.split(":")
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


def compute_clip_embedding(image: Image.Image, model_str: str) -> List[float]:
    parsed = parse_clip_model(model_str)
    if not parsed:
        raise SystemExit(f"Invalid CLIP model string: {model_str}")
    model_name, pretrained = parsed
    try:
        import torch
        import open_clip
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install CLIP deps: pip install open_clip_torch torch pillow") from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        emb = model.encode_image(img_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().float().tolist()


def compute_query_embedding(pil_image: Image.Image, model_str: str) -> List[float]:
    if model_str.startswith("simple"):
        _, vec = compute_simple_embedding(pil_image, model="simple")
        return vec
    if model_str.startswith("clip:"):
        return compute_clip_embedding(pil_image, model_str)
    raise SystemExit(f"Unsupported model '{model_str}' for query embedding.")


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


def prepare_region_image(base_image: Image.Image, mask_path: Path, bbox: Dict | None) -> Image.Image:
    mask_img = Image.open(mask_path).convert("L")
    if bbox:
        mask_img = mask_img.crop((bbox["minx"], bbox["miny"], bbox["maxx"] + 1, bbox["maxy"] + 1))
        img = base_image.crop((bbox["minx"], bbox["miny"], bbox["maxx"] + 1, bbox["maxy"] + 1))
    else:
        img = base_image
    # apply mask as alpha
    img_rgba = img.convert("RGBA")
    alpha = mask_img.point(lambda p: 255 if p >= 128 else 0)
    img_rgba.putalpha(alpha)
    return img_rgba.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run image-embedding retrieval per region mask produced by segment_image.py."
    )
    parser.add_argument("--segments", required=True, help="Path to segments.json produced by segment_image.py.")
    parser.add_argument("--image", required=True, help="Original image used for segmentation.")
    parser.add_argument(
        "--embeddings-file",
        default="embroidery_library/image_embeddings_clip.json",
        help="Embeddings JSON to use (default: embroidery_library/image_embeddings_clip.json).",
    )
    parser.add_argument(
        "--model",
        default="clip:ViT-B-32:openai",
        help="Embedding model tag matching the embeddings file (default: clip:ViT-B-32:openai).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k results per region (default: 3).",
    )
    parser.add_argument(
        "--min-area-pct",
        type=float,
        default=0.5,
        help="Skip regions smaller than this percentage of the image (default: 0.5%%).",
    )
    parser.add_argument(
        "--output",
        default="embroidery_dst_lab/output/region_retrieval.json",
        help="Where to save the retrieval results.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    args = parser.parse_args()

    segments = json.loads(Path(args.segments).read_text(encoding="utf-8"))
    regions = segments.get("regions", [])
    base_img = Image.open(args.image).convert("RGB")

    embeddings = load_embeddings(Path(args.embeddings_file))
    corpus_items: List[str] = []
    corpus_assets: List[str] = []
    vectors: List[List[float]] = []
    models: List[str] = []
    for item_id, assets in embeddings.items():
        for asset_type, entry in assets.items():
            corpus_items.append(item_id)
            corpus_assets.append(asset_type)
            vectors.append(entry["embedding"])
            models.append(entry.get("model", ""))
    matrix = np.asarray(vectors, dtype=np.float32)

    unique_models = {m for m in models if m}
    if args.model not in unique_models:
        if len(unique_models) > 1:
            raise SystemExit(f"Embeddings file has models {unique_models}; requested {args.model} not found.")
        # fallback to the single model present
        args.model = next(iter(unique_models))

    results = []
    total_px = segments.get("width", 0) * segments.get("height", 0)
    min_area_px = total_px * (args.min_area_pct / 100.0) if total_px else 0

    for region in regions:
        if region.get("area_px", 0) < min_area_px:
            continue
        mask_path = Path(region["mask"])
        region_img = prepare_region_image(base_img, mask_path, region.get("bbox"))
        query_vec = compute_query_embedding(region_img, args.model)
        sims = cosine_similarity(np.asarray(query_vec, dtype=np.float32), matrix)
        ranked_idx = np.argsort(-sims)
        top = []
        for idx in ranked_idx[: args.top_k]:
            top.append(
                {
                    "item_id": corpus_items[idx],
                    "asset": corpus_assets[idx],
                    "score": float(sims[idx]),
                    "model": models[idx],
                }
            )
        results.append(
            {
                "region_id": region.get("id"),
                "bbox": region.get("bbox"),
                "area_px": region.get("area_px"),
                "area_pct": region.get("area_pct"),
                "mean_color_rgb": region.get("mean_color_rgb"),
                "mask": region.get("mask"),
                "matches": top,
            }
        )

    payload = {"segments": str(args.segments), "image": str(args.image), "model": args.model, "results": results}
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Saved region retrieval results to {out_path} (regions: {len(results)})")


if __name__ == "__main__":
    main()
