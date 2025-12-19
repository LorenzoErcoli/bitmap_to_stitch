#!/usr/bin/env python3
"""Suggest a recipe by retrieving similar items (text + optional image)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from build_image_embeddings import compute_embedding as compute_simple_embedding


def load_index(library_root: Path) -> Dict[str, Dict]:
    index_path = library_root / "library_index.json"
    if not index_path.exists():
        raise SystemExit(f"Library index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8")).get("items", {})


def tokenize(text: str) -> List[str]:
    return [t for t in text.lower().split() if t]


def text_score(item: Dict, tokens: List[str], require_classes: List[str]) -> int:
    classification_summary = {k.lower(): v for k, v in item.get("classification_summary", {}).items()}
    if require_classes:
        if not any(classification_summary.get(req, 0) > 0 for req in require_classes):
            return -1
    haystack = f"{item.get('label','')} {item.get('description','')} "
    meta = item.get("metadata", {}) or {}
    haystack += " ".join(str(meta.get(k, "")) for k in ("subject", "fabric", "stabilizer", "notes"))
    tags = meta.get("tags", []) or []
    haystack += " " + " ".join(tags)
    haystack = haystack.lower()
    return sum(haystack.count(tok) for tok in tokens) + sum(classification_summary.values())


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


def compute_clip_embedding(image_path: Path, model_str: str) -> List[float]:
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


def compute_query_embedding(image_path: Path, model_str: str) -> List[float]:
    if model_str.startswith("simple"):
        _, vec = compute_simple_embedding(image_path, model="simple")
        return vec
    if model_str.startswith("clip:"):
        return compute_clip_embedding(image_path, model_str)
    raise SystemExit(f"Unsupported model '{model_str}' for query embedding.")


def cosine_similarity(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
    mat_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return mat_norm @ vec_norm


def load_recipe(library_root: Path, item_id: str, data: Dict) -> Dict:
    paths = data.get("paths", {})
    recipe_rel = paths.get("recipe")
    if not recipe_rel:
        raise SystemExit(f"No recipe path for item {item_id}")
    recipe_path = library_root / recipe_rel
    if not recipe_path.exists():
        raise SystemExit(f"Recipe file not found: {recipe_path}")
    text = recipe_path.read_text(encoding="utf-8")
    if not text.strip():
        raise SystemExit(f"Recipe file is empty: {recipe_path}")
    if recipe_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pyyaml is required to read recipe.yaml") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Suggest a recipe by retrieving similar items (text + optional image)."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder (default: embroidery_library)",
    )
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Embeddings JSON to use if doing image rerank. If omitted, defaults to <library_root>/image_embeddings_clip.json then image_embeddings.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model tag (e.g., clip:ViT-B-32:openai or simple_v1).",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Text query applied to label/description/metadata.",
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
        help="Optional image path to rerank top text matches.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matches to show and consider (default: 3).",
    )
    parser.add_argument(
        "--save-suggestion",
        default=None,
        help="Optional path to save suggested recipe JSON (top-1) with metadata.",
    )
    parser.add_argument(
        "--allow-text-only",
        action="store_true",
        help="When using --query-image, also include items without embeddings (text score only).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON instead of human-readable text.",
    )
    args = parser.parse_args()

    library_root = Path(args.library_root)
    index = load_index(library_root)
    tokens = tokenize(args.query)
    require_classes = [c.lower() for c in args.require_class]

    scored = []
    for item_id, data in index.items():
        s = text_score(data, tokens, require_classes)
        if s < 0:
            continue
        scored.append((s, item_id, data))
    scored.sort(key=lambda x: (-x[0], -(x[2].get("stitch_count") or 0)))

    results = []
    used_model = None

    if args.query_image:
        embeddings_path = (
            Path(args.embeddings_file)
            if args.embeddings_file
            else (
                library_root / "image_embeddings_clip.json"
                if (library_root / "image_embeddings_clip.json").exists()
                else library_root / "image_embeddings.json"
            )
        )
        embeddings = load_embeddings(embeddings_path)

        # Build corpus limited to text top candidates
        text_candidates = scored[: max(args.top_k * 5, 50)]

        corpus_items: List[str] = []
        corpus_assets: List[str] = []
        corpus_vecs: List[List[float]] = []
        corpus_text_scores: List[int] = []
        corpus_models: List[str] = []
        text_only_candidates: List[Dict] = []
        for s, item_id, data in text_candidates:
            assets = embeddings.get(item_id)
            if not assets:
                if args.allow_text_only:
                    text_only_candidates.append(
                        {
                            "item_id": item_id,
                            "asset": None,
                            "score_text": float(s),
                            "score_image": None,
                            "score": float(s),
                            "label": data.get("label", item_id),
                        }
                    )
                continue
            for asset_type, entry in assets.items():
                model_tag = entry.get("model") or "simple_v1"
                if args.model and model_tag != args.model:
                    continue
                corpus_items.append(item_id)
                corpus_assets.append(asset_type)
                corpus_vecs.append(entry["embedding"])
                corpus_text_scores.append(s)
                corpus_models.append(model_tag)

        if not corpus_vecs:
            raise SystemExit("No candidates with matching embeddings. Regenerate embeddings or adjust --model.")

        # Resolve model choice
        unique_models = {m for m in corpus_models if m}
        if args.model:
            used_model = args.model
        else:
            if len(unique_models) > 1:
                raise SystemExit(f"Multiple models found in corpus {unique_models}; specify --model.")
            used_model = unique_models.pop()

        img_path = Path(args.query_image)
        if not img_path.exists():
            raise SystemExit(f"Query image not found: {img_path}")
        query_vec = compute_query_embedding(img_path, used_model)
        sims = cosine_similarity(np.asarray(query_vec, dtype=np.float32), np.asarray(corpus_vecs, dtype=np.float32))

        for item_id, asset, text_s, sim, model_tag in zip(
            corpus_items, corpus_assets, corpus_text_scores, sims, corpus_models
        ):
            if model_tag != used_model:
                continue
            score = float(text_s) + float(sim)
            results.append(
                {
                    "item_id": item_id,
                    "asset": asset,
                    "score_text": float(text_s),
                    "score_image": float(sim),
                    "score": score,
                    "label": index[item_id].get("label", item_id),
                }
            )
        if args.allow_text_only:
            results.extend(text_only_candidates)
        results.sort(key=lambda r: -r["score"])
        results = results[: args.top_k]
    else:
        for s, item_id, data in scored[: args.top_k]:
            results.append(
                {
                    "item_id": item_id,
                    "asset": None,
                    "score_text": float(s),
                    "score_image": None,
                    "score": float(s),
                    "label": data.get("label", item_id),
                }
            )

    # Load top-1 recipe if available
    suggested_recipe = None
    if results:
        top = results[0]
        data = index[top["item_id"]]
        recipe = load_recipe(library_root, top["item_id"], data)
        recipe["suggested_from"] = {
            "item_id": top["item_id"],
            "label": data.get("label", top["item_id"]),
            "score_text": top["score_text"],
            "score_image": top["score_image"],
            "score_total": top["score"],
            "asset": top["asset"],
            "model": used_model,
        }
        suggested_recipe = recipe

    if args.json:
        payload = {"results": results, "suggested_recipe": suggested_recipe}
        print(json.dumps(payload, indent=2))
    else:
        print(f"Results top {args.top_k}:")
        for r in results:
            extra = f" img={r['score_image']:.3f}" if r["score_image"] is not None else ""
            print(
                f"- {r['label']} ({r['item_id']}) score={r['score']:.3f} "
                f"text={r['score_text']}{extra}"
            )
        if suggested_recipe:
            print(f"\nSuggested recipe from: {results[0]['item_id']}")
            print("Fields: recipe JSON merged with suggested_from metadata.")

    if suggested_recipe and args.save_suggestion:
        out_path = Path(args.save_suggestion)
        out_path.write_text(json.dumps(suggested_recipe, indent=2), encoding="utf-8")
        print(f"Saved suggested recipe to {out_path}")


if __name__ == "__main__":
    main()
