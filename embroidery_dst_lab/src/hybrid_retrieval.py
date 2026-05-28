#!/usr/bin/env python3
"""Hybrid retrieval: text filter/score + CLIP image rerank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_index(library_root: Path) -> Dict[str, Dict]:
    index_path = library_root / "library_index.json"
    if not index_path.exists():
        raise SystemExit(f"Library index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8")).get("items", {})


def load_embeddings(path: Path) -> Dict[str, Dict[str, Dict]]:
    if not path.exists():
        raise SystemExit(f"Embeddings file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def text_score(item: Dict, tokens: List[str], require_classes: List[str]) -> int:
    classification_summary = {k.lower(): v for k, v in item.get("classification_summary", {}).items()}
    if require_classes:
        if not any(classification_summary.get(req, 0) > 0 for req in require_classes):
            return -1
    haystack = f"{item.get('label','')} {item.get('description','')} "
    meta = item.get("metadata", {})
    haystack += " ".join(str(meta.get(k, "")) for k in ("subject", "fabric", "stabilizer", "notes"))
    tags = meta.get("tags", []) or []
    haystack += " " + " ".join(tags)
    haystack = haystack.lower()
    return sum(haystack.count(tok) for tok in tokens) + sum(classification_summary.values())


def parse_clip_model(model_str: str) -> Tuple[str, str] | None:
    if not model_str.startswith("clip:"):
        return None
    parts = model_str.split(":")
    if len(parts) != 3:
        return None
    return parts[1], parts[2]


def compute_clip_image_embedding(image_path: Path, model_str: str) -> List[float]:
    parsed = parse_clip_model(model_str)
    if not parsed:
        raise SystemExit(f"Invalid CLIP model string: {model_str}")
    model_name, pretrained = parsed
    try:
        import torch
        import open_clip
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install CLIP deps: pip install open_clip_torch torch pillow") from exc
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


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid retrieval: text filter/boost + CLIP image rerank."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder (default: embroidery_library)",
    )
    parser.add_argument(
        "--embeddings-file",
        default="embroidery_library/image_embeddings_clip.json",
        help="Embeddings file to use (default: embroidery_library/image_embeddings_clip.json).",
    )
    parser.add_argument(
        "--model",
        default="clip:ViT-B-32:openai",
        help="Model tag matching the embeddings (default: clip:ViT-B-32:openai).",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Text query (applied to label/description/metadata).",
    )
    parser.add_argument(
        "--require-class",
        action="append",
        default=[],
        help="Filter by layer class; repeatable.",
    )
    parser.add_argument(
        "--query-image",
        default=None,
        help="Optional image path for rerank; if omitted, only text ranking is used.",
    )
    parser.add_argument(
        "--max-text",
        type=int,
        default=50,
        help="Max candidates to keep after text stage before image rerank (default: 50).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Final results to output (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON.",
    )
    args = parser.parse_args()

    index = load_index(Path(args.library_root))
    tokens = tokenize(args.query)
    require_classes = [c.lower() for c in args.require_class]

    # Text stage
    scored = []
    for item_id, data in index.items():
        s = text_score(data, tokens, require_classes)
        if s < 0:
            continue
        scored.append((s, item_id, data))
    scored.sort(key=lambda x: (-x[0], -(x[2].get("stitch_count") or 0)))
    text_candidates = scored[: args.max_text]

    if not args.query_image:
        results = [
            {"item_id": item_id, "score_text": s, "score": s, "label": data.get("label", item_id)}
            for s, item_id, data in text_candidates[: args.top_k]
        ]
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(f"Text-only results (top {args.top_k}):")
            for r in results:
                print(f"- {r['label']} ({r['item_id']})  score_text: {r['score_text']}")
        return

    # Image rerank stage (CLIP)
    embeddings = load_embeddings(Path(args.embeddings_file))
    # Collect corpus vectors for candidates that have image embeddings
    corpus_items: List[str] = []
    corpus_assets: List[str] = []
    corpus_vecs: List[List[float]] = []
    corpus_text_scores: List[int] = []
    for s, item_id, data in text_candidates:
        assets = embeddings.get(item_id)
        if not assets:
            continue
        for asset_type, entry in assets.items():
            if entry.get("model") != args.model:
                continue
            corpus_items.append(item_id)
            corpus_assets.append(asset_type)
            corpus_vecs.append(entry["embedding"])
            corpus_text_scores.append(s)

    if not corpus_vecs:
        print("No candidate had matching image embeddings; returning text-only ranking.")
        return

    img_path = Path(args.query_image)
    if not img_path.exists():
        raise SystemExit(f"Query image not found: {img_path}")
    query_vec = compute_clip_image_embedding(img_path, args.model)
    sims = cosine_similarity(np.asarray(query_vec, dtype=np.float32), np.asarray(corpus_vecs, dtype=np.float32))

    ranked = []
    for item_id, asset, text_s, sim in zip(corpus_items, corpus_assets, corpus_text_scores, sims):
        score = float(text_s) + float(sim)
        ranked.append(
            {
                "item_id": item_id,
                "asset": asset,
                "score_text": float(text_s),
                "score_image": float(sim),
                "score": score,
            }
        )
    ranked.sort(key=lambda r: -r["score"])
    results = ranked[: args.top_k]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"Hybrid results (text+image) top {args.top_k}:")
        for r in results:
            print(
                f"- {r['item_id']} [{r['asset']}]  score={r['score']:.3f}  "
                f"text={r['score_text']}  image={r['score_image']:.3f}"
            )


if __name__ == "__main__":
    main()
