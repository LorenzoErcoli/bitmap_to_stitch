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
    if preview_path is not None:
        manifest["paths"]["preview"] = str(preview_path)
    bounds = stats.get("layers", [])
    if bounds:
        manifest["bounds_mm"] = bounds[0].get("bounds")
    if description:
        manifest["description"] = description
    if density_range:
        manifest["density_range_pts_per_mm2"] = density_range
    return manifest


def package_dst(
    dst_path: Path,
    item_id: str,
    library_root: Path,
    label: str | None = None,
    description: str | None = None,
    skip_preview: bool = False,
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
    args = parser.parse_args()

    item_dir = package_dst(
        dst_path=Path(args.dst),
        item_id=args.item_id,
        library_root=Path(args.library_root),
        label=args.label,
        description=args.description,
        skip_preview=args.skip_preview,
    )
    print(f"Library item packaged under {item_dir}")


if __name__ == "__main__":
    main()
