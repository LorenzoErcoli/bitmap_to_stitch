#!/usr/bin/env python3
"""Package DST-derived artifacts into a library item for the AI planner."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_stats import compute_stats, plot_preview  # noqa: E402
from dst_to_ir import dst_to_ir  # noqa: E402
from ir_to_recipe import build_recipe  # noqa: E402


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _relative(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}


def _copy_optional_asset(src: Path | None, dest_dir: Path, basename: str) -> Optional[Path]:
    if src is None:
        return None
    if not src.exists():
        raise FileNotFoundError(src)
    suffix = src.suffix.lower()
    dest = dest_dir / f"{basename}{suffix}"
    shutil.copy2(src, dest)
    return dest


def _build_manifest(
    item_id: str,
    dst_copy: Path,
    ir_path: Path,
    stats_path: Path,
    recipe_path: Path,
    preview_path: Path | None,
    stats: Dict[str, Any],
    description: str | None = None,
    density_range: Dict[str, float] | None = None,
    artwork_path: Path | None = None,
    stitched_photo_path: Path | None = None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest = {
        "id": item_id,
        "stitch_count": stats.get("stitch_count", 0),
        "color_layers": len(stats.get("layers", [])),
        "classification_summary": stats.get("layer_classification_summary", {}),
        "paths": {
            "dst": str(dst_copy),
            "stitch_ir": str(ir_path),
            "stats": str(stats_path),
            "recipe": str(recipe_path),
        },
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if artwork_path is not None:
        manifest["paths"]["artwork"] = str(artwork_path)
    if stitched_photo_path is not None:
        manifest["paths"]["stitched_photo"] = str(stitched_photo_path)
    if preview_path is not None:
        manifest["paths"]["preview"] = str(preview_path)
    bounds = stats.get("layers", [])
    if bounds:
        manifest["bounds_mm"] = bounds[0].get("bounds")
    if description:
        manifest["description"] = description
    if density_range:
        manifest["density_range_pts_per_mm2"] = density_range
    if metadata:
        manifest["metadata"] = metadata
    return manifest


def package_dst(
    dst_path: Path,
    item_id: str,
    library_root: Path,
    label: str | None = None,
    description: str | None = None,
    skip_preview: bool = False,
    artwork: Path | None = None,
    stitched_photo: Path | None = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    if not dst_path.exists():
        raise FileNotFoundError(dst_path)

    items_root = library_root / "items"
    item_dir = items_root / item_id
    item_dir.mkdir(parents=True, exist_ok=True)

    label_value = label or dst_path.stem

    dst_copy_path = item_dir / "source.dst"
    shutil.copy2(dst_path, dst_copy_path)

    ir = dst_to_ir(dst_path)
    ir_path = item_dir / "stitch_ir.json"
    _write_json(ir_path, ir)

    stats = compute_stats(ir)
    stats_path = item_dir / "stats.json"
    _write_json(stats_path, stats)

    preview_path = None
    if not skip_preview:
        layers = defaultdict(list)
        for stitch in ir["stitches"]:
            layers[stitch["color"]].append(stitch)
        if layers:
            preview_path = item_dir / "preview.png"
            plot_preview(layers, preview_path, title=item_id)

    examples: List[str] = []
    if preview_path and preview_path.exists():
        examples.append(_relative(preview_path, library_root))
    examples.append(_relative(ir_path, library_root))

    recipe = build_recipe(
        recipe_id=item_id,
        stats=stats,
        examples=examples,
        source=str(_relative(dst_copy_path, library_root)),
    )
    recipe["label"] = label_value
    summaries = stats.get("layer_summaries") or []
    auto_description: Optional[str] = "; ".join(summaries) if summaries else None
    final_description = description or auto_description
    if final_description:
        recipe["description"] = final_description
    recipe["stitch_profile"]["layer_classification_summary"] = stats.get(
        "layer_classification_summary", {}
    )

    artwork_copy = _copy_optional_asset(artwork, item_dir, "artwork")
    stitched_photo_copy = _copy_optional_asset(stitched_photo, item_dir, "stitched_photo")
    if artwork_copy:
        recipe.setdefault("source_assets", {})["artwork"] = _relative(artwork_copy, library_root)
    if stitched_photo_copy:
        recipe.setdefault("source_assets", {})["stitched_photo"] = _relative(
            stitched_photo_copy, library_root
        )
    normalized_metadata: Dict[str, Any] = {}
    if metadata:
        for key, value in metadata.items():
            if value is None:
                continue
            if key == "tags":
                cleaned_tags = [t for t in value if t]
                if cleaned_tags:
                    normalized_metadata[key] = cleaned_tags
            else:
                normalized_metadata[key] = value
    if normalized_metadata:
        recipe["metadata"] = normalized_metadata

    recipe_json_path = item_dir / "recipe.json"
    recipe_yaml_path = item_dir / "recipe.yaml"
    _write_json(recipe_json_path, recipe)
    _write_yaml(recipe_yaml_path, recipe)

    density_values = [
        layer["density_stitches_per_mm2"]
        for layer in stats.get("layers", [])
        if layer.get("density_stitches_per_mm2") is not None
    ]
    density_range = (
        {
            "min": float(min(density_values)),
            "max": float(max(density_values)),
        }
        if density_values
        else None
    )

    manifest = _build_manifest(
        item_id,
        _relative(dst_copy_path, library_root),
        _relative(ir_path, library_root),
        _relative(stats_path, library_root),
        _relative(recipe_yaml_path, library_root),
        _relative(preview_path, library_root) if preview_path else None,
        stats,
        description=final_description,
        density_range=density_range,
        artwork_path=Path(_relative(artwork_copy, library_root)) if artwork_copy else None,
        stitched_photo_path=Path(_relative(stitched_photo_copy, library_root))
        if stitched_photo_copy
        else None,
        metadata=normalized_metadata,
    )
    manifest_path = item_dir / "manifest.json"
    _write_json(manifest_path, manifest)

    index_path = library_root / "library_index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        index = {"items": {}}

    index_entry = {
        "label": label_value,
        "paths": manifest["paths"],
        "stitch_count": stats.get("stitch_count", 0),
        "classification_summary": stats.get("layer_classification_summary", {}),
        "updated_at": manifest["generated_at"],
    }
    if final_description:
        index_entry["description"] = final_description
    if density_range:
        index_entry["density_range_pts_per_mm2"] = density_range
    if artwork_copy:
        index_entry.setdefault("paths", {})["artwork"] = _relative(artwork_copy, library_root)
    if stitched_photo_copy:
        index_entry.setdefault("paths", {})["stitched_photo"] = _relative(
            stitched_photo_copy, library_root
        )
    if normalized_metadata:
        index_entry["metadata"] = normalized_metadata
    index["items"][item_id] = index_entry

    _write_json(index_path, index)
    return item_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create /embroidery_library/items/<id>/ package with IR, stats, preview, recipe."
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Path to the DST file to ingest.",
    )
    parser.add_argument(
        "--item-id",
        required=True,
        help="Identifier for the library item (used as folder name).",
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder of the embroidery library (default: embroidery_library).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional human label; defaults to the DST filename.",
    )
    parser.add_argument(
        "--description",
        default=None,
        help="Optional textual description to help the AI planner.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip PNG preview generation (useful on headless setups without matplotlib).",
    )
    parser.add_argument(
        "--artwork",
        default=None,
        help="Optional path to the original artwork bitmap used to program the DST.",
    )
    parser.add_argument(
        "--stitched-photo",
        default=None,
        help="Optional path to a photo of the stitched result.",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Optional subject/category tag (e.g., floral, logo, lettering).",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Additional tags; repeatable.",
    )
    parser.add_argument(
        "--fabric",
        default=None,
        help="Fabric/support info (e.g., cotton twill, velluto).",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread info (brand/type/colorway) if relevant.",
    )
    parser.add_argument(
        "--stabilizer",
        default=None,
        help="Stabilizer/backing used.",
    )
    parser.add_argument(
        "--quality-score",
        type=int,
        default=None,
        help="Optional quality rating 1-5 of the stitched result.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Free-text notes (defects, machine, tension, operator).",
    )
    args = parser.parse_args()

    metadata = {
        "subject": args.subject,
        "tags": args.tag,
        "fabric": args.fabric,
        "thread": args.thread,
        "stabilizer": args.stabilizer,
        "quality_score": args.quality_score,
        "notes": args.notes,
    }

    item_dir = package_dst(
        dst_path=Path(args.dst),
        item_id=args.item_id,
        library_root=Path(args.library_root),
        label=args.label,
        description=args.description,
        skip_preview=args.skip_preview,
        artwork=Path(args.artwork) if args.artwork else None,
        stitched_photo=Path(args.stitched_photo) if args.stitched_photo else None,
        metadata=metadata,
    )
    print(f"Library item packaged under {item_dir}")


if __name__ == "__main__":
    main()
