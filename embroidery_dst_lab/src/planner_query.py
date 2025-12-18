#!/usr/bin/env python3
"""CLI helper for querying embroidery_library items as a lightweight AI planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_index(library_root: Path) -> Dict[str, Dict]:
    index_path = library_root / "library_index.json"
    if not index_path.exists():
        raise SystemExit(f"Library index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8")).get("items", {})


def tokenize(text: str) -> List[str]:
    return [token for token in text.lower().split() if token]


def score_item(
    label: str,
    description: str | None,
    classification_summary: Dict[str, int],
    tokens: List[str],
    require_classes: List[str],
) -> Tuple[int, Dict[str, int]]:
    if require_classes:
        has_required = any(
            classification_summary.get(req, 0) > 0 for req in require_classes
        )
        if not has_required:
            return -1, classification_summary
    if not tokens:
        base_score = 1
    else:
        haystack = f"{label} {description or ''}".lower()
        base_score = sum(haystack.count(token) for token in tokens)
    class_boost = sum(classification_summary.values())
    return base_score + class_boost, classification_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the embroidery library index with simple keyword and classification filters."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Free-text keywords to search inside labels/descriptions.",
    )
    parser.add_argument(
        "--require-class",
        action="append",
        default=[],
        help="Filter by stitch classification (e.g., --require-class tatami). Repeatable.",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Number of results to display (default: 5).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output the ranked list as JSON instead of text.",
    )
    args = parser.parse_args()

    library_root = Path(args.library_root)
    index_items = load_index(library_root)
    tokens = tokenize(args.query)
    require_classes = [c.lower() for c in args.require_class]

    ranked = []
    for item_id, data in index_items.items():
        label = data.get("label", item_id)
        classification_summary = {
            k.lower(): v for k, v in data.get("classification_summary", {}).items()
        }
        description = data.get("description")
        score, _ = score_item(label, description, classification_summary, tokens, require_classes)
        if score < 0:
            continue
        ranked.append(
            {
                "id": item_id,
                "label": label,
                "score": score,
                "stitch_count": data.get("stitch_count"),
                "classification_summary": classification_summary,
                "description": description,
                "recipe": data["paths"].get("recipe"),
                "preview": data["paths"].get("preview"),
            }
        )

    ranked.sort(key=lambda item: (-item["score"], -item["stitch_count"]))
    results = ranked[: args.max_results]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("No items matched the query.")
            return
        for item in results:
            print(f"- {item['label']} ({item['id']})")
            print(f"  score: {item['score']}  stitches: {item['stitch_count']}")
            if item["classification_summary"]:
                summary_str = ", ".join(
                    f"{k}:{v}" for k, v in item["classification_summary"].items()
                )
                print(f"  classes: {summary_str}")
            if item["description"]:
                print(f"  desc: {item['description']}")
            if item["recipe"]:
                print(f"  recipe: {item['recipe']}")
            if item["preview"]:
                print(f"  preview: {item['preview']}")
            print()


if __name__ == "__main__":
    main()
