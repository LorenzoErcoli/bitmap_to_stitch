#!/usr/bin/env python3
"""Generate a JSON recipe card from computed stitch statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def build_recipe(
    recipe_id: str,
    stats: Dict[str, Any],
    examples: List[str],
    source: str,
) -> Dict[str, Any]:
    return {
        "id": recipe_id,
        "source": source,
        "tags": {
            "material": [],
            "object": [],
            "source": "dst_import",
        },
        "stitch_profile": {
            "avg_stitch_length_mm": stats.get("avg_stitch_length_mm", 0.0),
            "max_stitch_length_mm": stats.get("max_stitch_length_mm", 0.0),
            "directionality_histogram_deg": stats.get("angle_histogram_deg", []),
            "layers": stats.get("layers", []),
        },
        "notes": [
            "Derived automatically from DST data.",
            "Parameters estimated via pyembroidery + numpy pipeline.",
        ],
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a recipe card JSON from stats/preview artifacts."
    )
    parser.add_argument(
        "--stats",
        default="embroidery_dst_lab/output/test_stats.json",
        help="Path to the stats JSON (default: embroidery_dst_lab/output/test_stats.json)",
    )
    parser.add_argument(
        "--ir",
        default="embroidery_dst_lab/output/test_stitch_ir.json",
        help="Path to the Stitch IR JSON (default: embroidery_dst_lab/output/test_stitch_ir.json)",
    )
    parser.add_argument(
        "--preview",
        default="embroidery_dst_lab/output/test_preview.png",
        help="Optional preview path to include inside the recipe examples.",
    )
    parser.add_argument(
        "--output",
        default="embroidery_dst_lab/output/test_recipe.json",
        help="Where to save the recipe JSON (default: embroidery_dst_lab/output/test_recipe.json)",
    )
    parser.add_argument(
        "--recipe-id",
        default="RECIPE_FROM_DST_TEST",
        help="Identifier for the recipe card (default: RECIPE_FROM_DST_TEST)",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise SystemExit(f"Stats file not found: {stats_path}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    examples = []
    preview_path = Path(args.preview)
    if preview_path.exists():
        examples.append(str(preview_path))

    ir_path = Path(args.ir)
    if ir_path.exists():
        examples.append(str(ir_path))

    recipe = build_recipe(args.recipe_id, stats, examples, source=str(ir_path))
    output_path = Path(args.output)
    output_path.write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    print(f"Recipe saved to {output_path}")


if __name__ == "__main__":
    main()
