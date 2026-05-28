#!/usr/bin/env python3
"""Nearest-neighbor retrieval on image embeddings (artwork/stitched_photo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from build_image_embeddings import compute_embedding as compute_simple_embedding


def load_embeddings(path: Path) -> Dict[str, Dict[str, Dict]]:
    if not path.exists():
        raise SystemExit(
            f"Embeddings file not found: {path}. Run a build_image_embeddings script first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def select_item_embedding(
    embeddings: Dict[str, Dict[str, Dict]],
    item_id: str,
    prefer: str = "artwork",
) -> Tuple[str, List[float]]:
    item = embeddings.get(item_id)
    if not item:
        raise SystemExit(f"Item '{item_id}' not found in embeddings.")
    # prefer requested asset, fallback to any available
    if prefer in item:
        key = prefer
    elif "artwork" in item:
        key = "artwork"
    elif "stitched_photo" in item:
        key = "stitched_photo"
    else:
        raise SystemExit(f"No image embeddings found for item '{item_id}'.")
    entry = item[key]
    return entry["model"], entry["embedding"]


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


def parse_clip_model(model_str: str) -> Tuple[str, str] | None:
    if not model_str.startswith("clip:"):
        return None
    parts = model_str.split(":")
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


def compute_clip_embedding(image_path: Path, model_str: str) -> List[float]:
    parsed = parse_clip_model(model_str)
    if not parsed:
        raise SystemExit(f"Invalid CLIP model string: {model_str}")
    model_name, pretrained = parsed
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise SystemExit(
            "CLIP dependencies missing. Install with: pip install open_clip_torch torch pillow"
        ) from exc
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    with Image.open(image_path) as img:
        img = img.convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        emb = model.encode_image(img_tensor)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze(0).cpu().float().tolist()


def compute_query_embedding(image_path: Path, model_str: str) -> List[float]:
    if model_str.startswith("simple"):
        _, vec = compute_simple_embedding(image_path, model="simple")
        return vec
    if model_str.startswith("clip:"):
        return compute_clip_embedding(image_path, model_str)
    raise SystemExit(f"Unsupported model '{model_str}' for query embedding.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find nearest neighbors using image embeddings (artwork/stitched_photo)."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--query-image",
        default=None,
        help="Path to an image to query. If set, computes embedding on the fly.",
    )
    parser.add_argument(
        "--query-item",
        default=None,
        help="Use an existing item-id as query (requires embeddings already built).",
    )
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Embeddings JSON to use (default: <library_root>/image_embeddings.json).",
    )
    parser.add_argument(
        "--prefer-asset",
        default="artwork",
        choices=["artwork", "stitched_photo"],
        help="When using --query-item, prefer this asset type if available.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model to use. If omitted and embeddings contain multiple models, specify one.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of nearest neighbors to show (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    if not args.query_image and not args.query_item:
        raise SystemExit("Provide --query-image or --query-item.")

    library_root = Path(args.library_root)
    embeddings_path = (
        Path(args.embeddings_file)
        if args.embeddings_file
        else library_root / "image_embeddings.json"
    )
    embeddings = load_embeddings(embeddings_path)

    # Flatten corpus vectors
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

    unique_models = {m for m in models if m}
    if args.model:
        filtered = [
            (i, a, v, m)
            for i, a, v, m in zip(corpus_items, corpus_assets, vectors, models)
            if m == args.model
        ]
        if not filtered:
            raise SystemExit(f"No embeddings with model '{args.model}' found in {embeddings_path}")
        corpus_items, corpus_assets, vectors, models = map(list, zip(*filtered))
        model_used = args.model
    else:
        if len(unique_models) > 1:
            raise SystemExit(
                f"Multiple models found in embeddings ({unique_models}). Specify --model."
            )
        model_used = unique_models.pop() if unique_models else "simple_v1"

    matrix = np.asarray(vectors, dtype=np.float32)

    # Build query embedding
    if args.query_image:
        img_path = Path(args.query_image)
        if not img_path.exists():
            raise SystemExit(f"Query image not found: {img_path}")
        query_vec = compute_query_embedding(img_path, model_used)
        model_name = model_used
    else:
        model_name, query_vec = select_item_embedding(
            embeddings, args.query_item, prefer=args.prefer_asset
        )
        if model_name != model_used:
            raise SystemExit(
                f"Query item embedding model '{model_name}' != selected corpus model '{model_used}'"
            )

    query_arr = np.asarray(query_vec, dtype=np.float32)
    sims = cosine_similarity(query_arr, matrix)

    # Rank
    ranked_idx = np.argsort(-sims)
    results = []
    for idx in ranked_idx[: args.top_k]:
        results.append(
            {
                "item_id": corpus_items[idx],
                "asset": corpus_assets[idx],
                "score": float(sims[idx]),
                "model": models[idx],
            }
        )

    if args.json:
        print(json.dumps(results, indent=2))
        return

    print(f"Query model: {model_name}  top {args.top_k}")
    for r in results:
        print(f"- {r['item_id']} [{r['asset']}]  score: {r['score']:.3f}")


if __name__ == "__main__":
    main()
