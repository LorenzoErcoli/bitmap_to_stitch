#!/usr/bin/env python3
"""Batch import of DST files into the embroidery library."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

from build_library_item import IMAGE_EXTS, package_dst


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "item"


def item_exists(library_root: Path, item_id: str) -> bool:
    manifest = library_root / "items" / item_id / "manifest.json"
    return manifest.exists()


def _parse_suffixes(raw: str) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _find_asset(
    dst_path: Path,
    suffixes: Iterable[str],
    allow_plain: bool,
) -> Optional[Path]:
    candidates = []
    if allow_plain:
        candidates.append(dst_path.stem)
    for suffix in suffixes:
        candidates.append(f"{dst_path.stem}{suffix}")
    for stem in candidates:
        for ext in IMAGE_EXTS:
            candidate = dst_path.with_name(f"{stem}{ext}")
            if candidate.exists():
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
                description="Scan embroidery_dst_lab/input for DST files and build missing library items."
    )
    parser.add_argument(
        "--input-dir",
        default="embroidery_dst_lab/input",
        help="Folder containing DST files (default: embroidery_dst_lab/input)",
    )
    parser.add_argument(
        "--library-root",
        default="embroidery_library",
        help="Root folder of the embroidery library (default: embroidery_library)",
    )
    parser.add_argument(
        "--description-template",
        default=None,
        help="Optional template for descriptions, e.g. 'Canvas fill: {name}'. Fields: {name}, {item_id}.",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Skip preview generation for all items.",
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help="Force regeneration even if the library item already exists.",
    )
    parser.add_argument(
        "--auto-artwork",
        action="store_true",
        help="Attach artwork image if found next to the DST (matching stem or provided suffixes).",
    )
    parser.add_argument(
        "--artwork-suffixes",
        default="_artwork",
        help="Comma-separated suffixes for artwork lookup (default: _artwork).",
    )
    parser.add_argument(
        "--auto-stitched-photo",
        action="store_true",
        help="Attach stitched photo if found next to the DST.",
    )
    parser.add_argument(
        "--stitched-photo-suffixes",
        default="_stitched,_photo,_ricamo",
        help="Comma-separated suffixes for stitched photo lookup (default: _stitched,_photo,_ricamo).",
    )
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject/category tag applied to all imported items.",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="Additional tags applied to all imported items; repeatable.",
    )
    parser.add_argument(
        "--fabric",
        default=None,
        help="Fabric/support info applied to all items.",
    )
    parser.add_argument(
        "--thread",
        default=None,
        help="Thread info applied to all items.",
    )
    parser.add_argument(
        "--stabilizer",
        default=None,
        help="Stabilizer/backing info applied to all items.",
    )
    parser.add_argument(
        "--quality-score",
        type=int,
        default=None,
        help="Quality rating 1-5 applied to all items.",
    )
    parser.add_argument(
        "--notes",
        default=None,
        help="Notes applied to all items.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    library_root = Path(args.library_root)
    dst_files = sorted(input_dir.glob("*.dst"))
    if not dst_files:
        print(f"No DST files found inside {input_dir}")
        return

    created = 0
    skipped = 0
    artwork_suffixes = _parse_suffixes(args.artwork_suffixes)
    stitched_suffixes = _parse_suffixes(args.stitched_photo_suffixes)
    metadata = {
        "subject": args.subject,
        "tags": args.tag,
        "fabric": args.fabric,
        "thread": args.thread,
        "stabilizer": args.stabilizer,
        "quality_score": args.quality_score,
        "notes": args.notes,
    }

    for dst_path in dst_files:
        name = dst_path.stem
        item_id = slugify(name)

        if not args.rebuild_existing and item_exists(library_root, item_id):
            print(f"[skip] {dst_path.name} -> {item_id} already exists")
            skipped += 1
            continue

        description: Optional[str] = None
        if args.description_template:
            description = args.description_template.format(name=name, item_id=item_id)

        artwork_path = None
        if args.auto_artwork:
            artwork_path = _find_asset(dst_path, artwork_suffixes, allow_plain=True)

        stitched_photo_path = None
        if args.auto_stitched_photo:
            stitched_photo_path = _find_asset(dst_path, stitched_suffixes, allow_plain=False)

        package_dst(
            dst_path=dst_path,
            item_id=item_id,
            library_root=library_root,
            label=None,
            description=description,
            skip_preview=args.skip_preview,
            artwork=artwork_path,
            stitched_photo=stitched_photo_path,
            metadata=metadata,
        )
        print(f"[ok] {dst_path.name} -> {item_id}")
        created += 1

    print(f"Done. Created {created} item(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
