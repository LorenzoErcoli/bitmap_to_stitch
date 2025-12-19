#!/usr/bin/env python3
"""Per-region recipe suggestion using OpenAI on top of region retrieval results."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_index(library_root: Path) -> Dict[str, Dict]:
    index_path = library_root / "library_index.json"
    if not index_path.exists():
        raise SystemExit(f"Library index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8")).get("items", {})


def load_recipe(library_root: Path, item_id: str, index_item: Dict) -> Dict:
    recipe_rel = index_item.get("paths", {}).get("recipe")
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
            raise SystemExit("pyyaml is required for YAML recipes") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def build_prompt(region: Dict, candidates: List[Dict]) -> List[Dict]:
    """Compose OpenAI messages for a single region."""
    region_info = {
        "region_id": region.get("region_id"),
        "bbox": region.get("bbox"),
        "area_pct": region.get("area_pct"),
        "mean_color_rgb": region.get("mean_color_rgb"),
    }
    # Reduce recipe info to summaries to keep prompt small
    compact_candidates = []
    for c in candidates:
        recipe = c["recipe"]
        layers = recipe.get("stitch_profile", {}).get("layers", []) or []
        layer_summaries = []
        for idx, layer in enumerate(layers, start=1):
            layer_summaries.append(
                {
                    "index": idx,
                    "class": (layer.get("classification") or {}).get("type"),
                    "density": layer.get("density_stitches_per_mm2"),
                    "avg_len": layer.get("avg_stitch_length_mm"),
                    "summary": layer.get("summary"),
                }
            )
        compact_candidates.append(
            {
                "item_id": c["item_id"],
                "label": recipe.get("label", c["item_id"]),
                "description": recipe.get("description"),
                "layer_summaries": layer_summaries,
            }
        )
    system = {
        "role": "system",
        "content": (
            "You are an embroidery planning assistant. Given a region (area, color) and candidate recipes, "
            "pick which recipe layers to reuse for this region. Respond strictly in JSON with fields: "
            "{region_id, selection: {item_id, layer_indices: [ints]}, notes}. "
            "Layer indices are 1-based as provided. Prefer coherent layers: tatami for fills, satin for outlines/details, run for travel. "
            "Keep layer_indices ordered as they should be stitched for this region."
        ),
    }
    user = {
        "role": "user",
        "content": json.dumps({"region": region_info, "candidates": compact_candidates}, indent=2),
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


def build_region_recipe(region: Dict, selection: Dict, recipes: Dict[str, Dict]) -> Dict:
    item_id = selection.get("item_id")
    layer_indices = selection.get("layer_indices") or []
    if not item_id or item_id not in recipes:
        return {"region_id": region.get("region_id"), "error": "invalid item_id in selection"}
    source_recipe = recipes[item_id]
    layers = source_recipe.get("stitch_profile", {}).get("layers", []) or []
    chosen_layers = []
    for idx in layer_indices:
        if 1 <= idx <= len(layers):
            layer_copy = json.loads(json.dumps(layers[idx - 1]))
            layer_copy["source_item"] = item_id
            layer_copy["source_layer_index"] = idx
            chosen_layers.append(layer_copy)
    return {
        "region_id": region.get("region_id"),
        "mask": region.get("mask"),
        "bbox": region.get("bbox"),
        "area_px": region.get("area_px"),
        "area_pct": region.get("area_pct"),
        "mean_color_rgb": region.get("mean_color_rgb"),
        "selection": selection,
        "layers": chosen_layers,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use OpenAI to select/merge recipe layers per region based on region retrieval results."
    )
    parser.add_argument(
        "--region-results",
        required=True,
        help="Path to region_retrieval.json produced by region_retrieval.py.",
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder of the embroidery library (default: embroidery_library).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-k matches per region to send to OpenAI (default: 3).",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model for planning (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--output",
        default="embroidery_dst_lab/output/region_suggestions.json",
        help="Where to save the per-region suggested recipes.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON to stdout.")
    args = parser.parse_args()

    region_results = load_json(Path(args.region_results))
    regions = region_results.get("results", [])
    if not regions:
        raise SystemExit("No regions found in region_results.")

    index = load_index(Path(args.library_root))

    # Preload recipes for all referenced items
    recipes: Dict[str, Dict] = {}
    for region in regions:
        for match in region.get("matches", [])[: args.top_k]:
            item_id = match["item_id"]
            if item_id in recipes:
                continue
            if item_id not in index:
                continue
            recipes[item_id] = load_recipe(Path(args.library_root), item_id, index[item_id])

    outputs = []
    for region in regions:
        matches = region.get("matches", [])[: args.top_k]
        enriched = []
        for m in matches:
            rid = m["item_id"]
            if rid not in recipes:
                continue
            enriched.append({"item_id": rid, "recipe": recipes[rid]})
        if not enriched:
            outputs.append(
                {
                    "region_id": region.get("region_id"),
                    "error": "no candidate recipes loaded",
                }
            )
            continue
        messages = build_prompt(region, enriched)
        plan = call_openai(messages, model=args.openai_model)
        region_recipe = build_region_recipe(region, plan.get("selection", {}), recipes)
        region_recipe["plan"] = plan
        outputs.append(region_recipe)

    payload = {
        "region_results": str(args.region_results),
        "openai_model": args.openai_model,
        "top_k": args.top_k,
        "regions": outputs,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Saved per-region suggestions to {out_path} (regions: {len(outputs)})")


if __name__ == "__main__":
    main()
