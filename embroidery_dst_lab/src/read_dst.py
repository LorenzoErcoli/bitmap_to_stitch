#!/usr/bin/env python3
"""Quick inspection utility for DST embroidery files."""

from __future__ import annotations

import argparse
from pathlib import Path

from pyembroidery import read


def read_dst(dst_path: Path, limit: int) -> None:
    """Read a DST file and print a short textual summary."""
    pattern = read(str(dst_path))

    print(f"File: {dst_path}")
    print(f"Stitch count: {len(pattern.stitches)}")
    print(f"Color changes: {len(pattern.threadlist)} thread entries")
    print("Threads (pyembroidery Thread objects):")
    for idx, thread in enumerate(pattern.threadlist):
        print(f"  [{idx}] {thread}")

    head = pattern.stitches[:limit]
    print(f"\nFirst {len(head)} stitches (x, y, command):")
    for x, y, cmd in head:
        print(f"  ({x}, {y}, {cmd})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a DST file and print the first stitches."
    )
    parser.add_argument(
        "dst",
        nargs="?",
        default="embroidery_dst_lab/input/test.dst",
        help="Path to the DST file (default: embroidery_dst_lab/input/test.dst)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="How many stitches to display from the head (default: 10)",
    )
    args = parser.parse_args()

    dst_path = Path(args.dst)
    if not dst_path.exists():
        raise SystemExit(f"File not found: {dst_path}")

    read_dst(dst_path, args.limit)


if __name__ == "__main__":
    main()
