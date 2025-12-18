#!/usr/bin/env python3
"""Generate recipe cards from stats/previews."""

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
        description="Generate a recipe JSON from stats/preview artifacts."
    )
    parser.add_argument(
        "--stats",
        required=True,
        help="Path to the stats JSON.",
    )
    parser.add_argument(
        "--ir",
        required=True,
        help="Path to the Stitch IR JSON (used as example reference).",
    )
    parser.add_argument(
        "--preview",
        default=None,
        help="Optional preview path to include inside the recipe examples.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to save the recipe JSON.",
    )
    parser.add_argument(
        "--recipe-id",
        required=True,
        help="Identifier for the recipe.",
    )
    args = parser.parse_args()

    stats_path = Path(args.stats)
    if not stats_path.exists():
        raise SystemExit(f"Stats file not found: {stats_path}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    examples = []
    if args.preview:
        preview_path = Path(args.preview)
        if preview_path.exists():
            examples.append(str(preview_path))

    ir_path = Path(args.ir)
    if ir_path.exists():
        examples.append(str(ir_path))

    recipe = build_recipe(args.recipe_id, stats, examples, source=str(ir_path))
    Path(args.output).write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    print(f"Recipe saved to {args.output}")


if __name__ == "__main__":
    main()
