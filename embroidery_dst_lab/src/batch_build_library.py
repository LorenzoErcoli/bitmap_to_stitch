#!/usr/bin/env python3
"""Batch import of DST files into the embroidery library."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

from build_library_item import package_dst


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "item"


def item_exists(library_root: Path, item_id: str) -> bool:
    manifest = library_root / "items" / item_id / "manifest.json"
    return manifest.exists()


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
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    library_root = Path(args.library_root)
    dst_files = sorted(input_dir.glob("*.dst"))
    if not dst_files:
        print(f"No DST files found inside {input_dir}")
        return

    created = 0
    skipped = 0
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

        package_dst(
            dst_path=dst_path,
            item_id=item_id,
            library_root=library_root,
            label=None,
            description=description,
            skip_preview=args.skip_preview,
        )
        print(f"[ok] {dst_path.name} -> {item_id}")
        created += 1

    print(f"Done. Created {created} item(s), skipped {skipped}.")


if __name__ == "__main__":
    main()
