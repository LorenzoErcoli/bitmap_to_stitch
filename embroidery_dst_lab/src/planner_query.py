#!/usr/bin/env python3
"""CLI helper for querying embroidery_library items with filters."""

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
        description="Query the embroidery library with keyword + technical filters."
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--query",
        default="",
        help="Keyword query applied to label/description.",
    )
    parser.add_argument(
        "--require-class",
        action="append",
        default=[],
        help="Filter by layer class (run/satin_light/satin_dense/tatami/travel/detail). Repeatable.",
    )
    parser.add_argument(
        "--min-stitches",
        type=int,
        default=None,
        help="Minimum stitch count required.",
    )
    parser.add_argument(
        "--max-stitches",
        type=int,
        default=None,
        help="Maximum stitch count allowed.",
    )
    parser.add_argument(
        "--min-density",
        type=float,
        default=None,
        help="Minimum density (pts/mm^2).",
    )
    parser.add_argument(
        "--max-density",
        type=float,
        default=None,
        help="Maximum density (pts/mm^2).",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Filter by subject/category (case-insensitive substring match).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Require at least one of these tags (repeatable).",
    )
    parser.add_argument(
        "--fabric",
        default=None,
        help="Filter by fabric/support (substring match).",
    )
    parser.add_argument(
        "--stabilizer",
        default=None,
        help="Filter by stabilizer/backing (substring match).",
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=None,
        help="Minimum quality score (1-5).",
    )
    parser.add_argument(
        "--max-quality",
        type=int,
        default=None,
        help="Maximum quality score (1-5).",
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
        help="Output JSON instead of human-readable text.",
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
        stitch_count = data.get("stitch_count")
        if args.min_stitches is not None and stitch_count is not None:
            if stitch_count < args.min_stitches:
                continue
        if args.max_stitches is not None and stitch_count is not None:
            if stitch_count > args.max_stitches:
                continue
        metadata = data.get("metadata", {}) or {}
        if args.subject:
            subj = metadata.get("subject", "")
            if args.subject.lower() not in subj.lower():
                continue
        if args.fabric:
            fabric = metadata.get("fabric", "")
            if args.fabric.lower() not in fabric.lower():
                continue
        if args.stabilizer:
            stab = metadata.get("stabilizer", "")
            if args.stabilizer.lower() not in stab.lower():
                continue
        tags = [t.lower() for t in metadata.get("tags", [])]
        if args.tag:
            required = [t.lower() for t in args.tag]
            if tags and not any(req in tags for req in required):
                continue
            if not tags:
                continue
        q = metadata.get("quality_score")
        if args.min_quality is not None and q is not None and q < args.min_quality:
            continue
        if args.max_quality is not None and q is not None and q > args.max_quality:
            continue
        density_range = data.get("density_range_pts_per_mm2")
        if density_range:
            if args.min_density is not None and density_range.get("max") is not None:
                if density_range["max"] < args.min_density:
                    continue
            if args.max_density is not None and density_range.get("min") is not None:
                if density_range["min"] > args.max_density:
                    continue
        score, _ = score_item(label, description, classification_summary, tokens, require_classes)
        if score < 0:
            continue
        ranked.append(
            {
                "id": item_id,
                "label": label,
                "score": score,
                "stitch_count": stitch_count,
                "classification_summary": classification_summary,
                "description": description,
                "density_range": density_range,
                "recipe": data["paths"].get("recipe"),
                "preview": data["paths"].get("preview"),
                "artwork": data.get("paths", {}).get("artwork")
                if isinstance(data.get("paths"), dict)
                else None,
                "stitched_photo": data.get("paths", {}).get("stitched_photo")
                if isinstance(data.get("paths"), dict)
                else None,
                "metadata": metadata,
            }
        )

    ranked.sort(key=lambda item: (-item["score"], -item["stitch_count"]))
    results = ranked[: args.max_results]

    if args.json:
        print(json.dumps(results, indent=2))
        return

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
        if item["density_range"]:
            dr = item["density_range"]
            print(f"  density: {dr.get('min')} - {dr.get('max')} pts/mm^2")
        if item["recipe"]:
            print(f"  recipe: {item['recipe']}")
        if item["preview"]:
            print(f"  preview: {item['preview']}")
        if item.get("artwork"):
            print(f"  artwork: {item['artwork']}")
        if item.get("stitched_photo"):
            print(f"  stitched: {item['stitched_photo']}")
        meta = item.get("metadata") or {}
        if meta.get("subject"):
            print(f"  subject: {meta['subject']}")
        if meta.get("tags"):
            print(f"  tags: {', '.join(meta['tags'])}")
        if meta.get("fabric"):
            print(f"  fabric: {meta['fabric']}")
        if meta.get("stabilizer"):
            print(f"  stabilizer: {meta['stabilizer']}")
        if meta.get("quality_score") is not None:
            print(f"  quality: {meta['quality_score']}")
        print()


if __name__ == "__main__":
    main()
