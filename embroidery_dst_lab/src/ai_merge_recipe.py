#!/usr/bin/env python3
"""AI-assisted recipe merge: retrieval + OpenAI plan + merged recipe."""

from __future__ import annotations

import argparse
import json
import os
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


def retrieve(
    library_root: Path,
    embeddings_path: Path,
    model_tag: str | None,
    query: str,
    require_classes: List[str],
    query_image: Path | None,
    top_k: int,
    allow_text_only: bool,
) -> Tuple[List[Dict], str | None]:
    index = load_index(library_root)
    tokens = tokenize(query)
    require_classes = [c.lower() for c in require_classes]

    scored = []
    for item_id, data in index.items():
        s = text_score(data, tokens, require_classes)
        if s < 0:
            continue
        scored.append((s, item_id, data))
    scored.sort(key=lambda x: (-x[0], -(x[2].get("stitch_count") or 0)))

    if not query_image:
        results = []
        for s, item_id, data in scored[:top_k]:
            results.append(
                {
                    "item_id": item_id,
                    "asset": None,
                    "score_text": float(s),
                    "score_image": None,
                    "score": float(s),
                    "label": data.get("label", item_id),
                    "data": data,
                }
            )
        return results, None

    embeddings = load_embeddings(embeddings_path)
    text_candidates = scored[: max(top_k * 5, 50)]

    corpus_items: List[str] = []
    corpus_assets: List[str] = []
    corpus_vecs: List[List[float]] = []
    corpus_text_scores: List[int] = []
    corpus_models: List[str] = []
    text_only_candidates: List[Dict] = []
    for s, item_id, data in text_candidates:
        assets = embeddings.get(item_id)
        if not assets:
            if allow_text_only:
                text_only_candidates.append(
                    {
                        "item_id": item_id,
                        "asset": None,
                        "score_text": float(s),
                        "score_image": None,
                        "score": float(s),
                        "label": data.get("label", item_id),
                        "data": data,
                    }
                )
            continue
        for asset_type, entry in assets.items():
            model_used = entry.get("model") or "simple_v1"
            if model_tag and model_used != model_tag:
                continue
            corpus_items.append(item_id)
            corpus_assets.append(asset_type)
            corpus_vecs.append(entry["embedding"])
            corpus_text_scores.append(s)
            corpus_models.append(model_used)

    if not corpus_vecs:
        raise SystemExit("No candidates with matching embeddings. Regenerate embeddings or adjust --model.")

    unique_models = {m for m in corpus_models if m}
    if model_tag:
        used_model = model_tag
    else:
        if len(unique_models) > 1:
            raise SystemExit(f"Multiple models in embeddings {unique_models}; specify --model.")
        used_model = unique_models.pop()

    if not query_image.exists():
        raise SystemExit(f"Query image not found: {query_image}")
    query_vec = compute_query_embedding(query_image, used_model)
    sims = cosine_similarity(np.asarray(query_vec, dtype=np.float32), np.asarray(corpus_vecs, dtype=np.float32))

    results = []
    for item_id, asset, text_s, sim, model_used in zip(
        corpus_items, corpus_assets, corpus_text_scores, sims, corpus_models
    ):
        if model_used != used_model:
            continue
        data = index[item_id]
        score = float(text_s) + float(sim)
        results.append(
            {
                "item_id": item_id,
                "asset": asset,
                "score_text": float(text_s),
                "score_image": float(sim),
                "score": score,
                "label": data.get("label", item_id),
                "data": data,
            }
        )
    if allow_text_only:
        results.extend(text_only_candidates)
    results.sort(key=lambda r: -r["score"])
    return results[:top_k], used_model


def build_prompt(query: str, results: List[Dict], recipes: Dict[str, Dict]) -> List[Dict]:
    items = []
    for r in results:
        rid = r["item_id"]
        recipe = recipes[rid]
        layers = recipe.get("stitch_profile", {}).get("layers", []) or []
        layer_summaries = []
        for idx, layer in enumerate(layers, start=1):
            layer_summaries.append(
                {
                    "index": idx,
                    "classification": layer.get("classification", {}).get("type"),
                    "density": layer.get("density_stitches_per_mm2"),
                    "avg_stitch_length_mm": layer.get("avg_stitch_length_mm"),
                    "coverage_ratio": layer.get("coverage_ratio"),
                    "summary": layer.get("summary"),
                }
            )
        items.append(
            {
                "id": rid,
                "label": recipe.get("label", rid),
                "description": recipe.get("description"),
                "metadata": r["data"].get("metadata", {}),
                "layer_summaries": layer_summaries,
            }
        )
    system = {
        "role": "system",
        "content": (
            "You are an embroidery planning assistant. Given candidate recipes (layers with stitch classes and densities) "
            "and a user intent, produce a merge plan choosing which layers to keep, reuse or discard. "
            "Output strictly JSON with fields: merge_plan.layers (list of {layer_class, source_item, layer_index}), "
            "and notes (string). Prefer consistent, minimal layer stack; pick tatami for fills, satin for outlines/details, "
            "run for travel/underlay. Layer indices are 1-based."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({"query": query, "candidates": items}, indent=2),
    }
    return [system, user]


def call_openai(messages: List[Dict], model: str) -> Dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in the environment.")
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install openai: pip install openai") from exc
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def build_merged_recipe(
    query: str,
    results: List[Dict],
    recipes: Dict[str, Dict],
    merge_plan: Dict,
) -> Dict:
    merged_layers = []
    sources = []
    for entry in merge_plan.get("merge_plan", {}).get("layers", []):
        source_item = entry.get("source_item")
        idx = entry.get("layer_index")
        if not source_item or idx is None:
            continue
        recipe = recipes.get(source_item)
        if not recipe:
            continue
        layers = recipe.get("stitch_profile", {}).get("layers", []) or []
        if not (1 <= idx <= len(layers)):
            continue
        src_layer = layers[idx - 1]
        new_layer = json.loads(json.dumps(src_layer))
        new_layer["color"] = len(merged_layers) + 1
        new_layer["source_item"] = source_item
        new_layer["source_layer_index"] = idx
        merged_layers.append(new_layer)
        sources.append(source_item)

    layer_classes = {}
    for layer in merged_layers:
        cls = layer.get("classification", {}).get("type")
        if cls:
            layer_classes[cls] = layer_classes.get(cls, 0) + 1

    merged_recipe = {
        "id": f"suggested_merge",
        "label": f"Suggested merge for: {query}",
        "sources": list(dict.fromkeys(sources)),
        "stitch_profile": {
            "layers": merged_layers,
            "layer_classification_summary": layer_classes,
        },
        "suggested_from": merge_plan,
    }
    return merged_recipe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-assisted merge of recipes using OpenAI to pick layers from top-k retrieval."
    )
    parser.add_argument("--library-root", default="embroidery_library", help="Library root.")
    parser.add_argument(
        "--embeddings-file",
        default=None,
        help="Embeddings JSON for image rerank (default: clip then simple).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Embedding model tag (clip:ViT-B-32:openai or simple_v1).",
    )
    parser.add_argument("--query", default="", help="Text query.")
    parser.add_argument(
        "--require-class",
        action="append",
        default=[],
        help="Filter by layer class; repeatable.",
    )
    parser.add_argument("--query-image", default=None, help="Optional image for rerank.")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k to consider (default 3).")
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model for planning (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--save",
        default="embroidery_dst_lab/output/ai_suggested_recipe.json",
        help="Where to save the merged recipe JSON.",
    )
    parser.add_argument(
        "--allow-text-only",
        action="store_true",
        help="When using --query-image, also include items without embeddings (text score only).",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout.")
    args = parser.parse_args()

    library_root = Path(args.library_root)
    embeddings_path = (
        Path(args.embeddings_file)
        if args.embeddings_file
        else (
            library_root / "image_embeddings_clip.json"
            if (library_root / "image_embeddings_clip.json").exists()
            else library_root / "image_embeddings.json"
        )
    )
    query_image = Path(args.query_image) if args.query_image else None

    results, used_model = retrieve(
        library_root=library_root,
        embeddings_path=embeddings_path,
        model_tag=args.model,
        query=args.query,
        require_classes=args.require_class,
        query_image=query_image,
        top_k=args.top_k,
        allow_text_only=args.allow_text_only,
    )
    if not results:
        raise SystemExit("No results after retrieval.")

    recipes: Dict[str, Dict] = {}
    for r in results:
        recipes[r["item_id"]] = load_recipe(library_root, r["item_id"], r["data"])

    messages = build_prompt(args.query, results, recipes)
    merge_plan = call_openai(messages, model=args.openai_model)

    merged_recipe = build_merged_recipe(args.query, results, recipes, merge_plan)
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged_recipe, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps({"results": results, "merge_plan": merge_plan, "merged_recipe": merged_recipe}, indent=2))
    else:
        print(f"Retrieved {len(results)} candidates (model: {used_model})")
        print(f"Merge plan: {json.dumps(merge_plan, indent=2)}")
        print(f"Merged recipe saved to {out_path}")


if __name__ == "__main__":
    main()
