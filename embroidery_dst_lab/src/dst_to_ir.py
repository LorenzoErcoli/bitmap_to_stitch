#!/usr/bin/env python3
"""Convert a DST file to a Stitch IR JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pyembroidery import COLOR_CHANGE, END, read

DST_UNIT_MM = 0.1


def dst_to_ir(dst_path: Path) -> Dict[str, Any]:
    pattern = read(str(dst_path))
    stitches: List[Dict[str, Any]] = []
    color_index = 0

    for i, (x, y, cmd) in enumerate(pattern.stitches):
        if cmd == COLOR_CHANGE:
            color_index += 1
        elif cmd == END:
            continue

        stitches.append(
            {
                "i": i,
                "x": x * DST_UNIT_MM,
                "y": y * DST_UNIT_MM,
                "cmd": int(cmd),
                "color": color_index,
            }
        )

    if stitches:
        bounds = {
            "minx": min(pt["x"] for pt in stitches),
            "miny": min(pt["y"] for pt in stitches),
            "maxx": max(pt["x"] for pt in stitches),
            "maxy": max(pt["y"] for pt in stitches),
        }
    else:
        bounds = {"minx": 0.0, "miny": 0.0, "maxx": 0.0, "maxy": 0.0}

    return {
        "source": str(dst_path),
        "unit": "mm",
        "stitch_count": len(stitches),
        "color_count": color_index + 1 if stitches else 0,
        "bounds": bounds,
        "stitches": stitches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a DST file into a structured Stitch IR JSON."
    )
    parser.add_argument(
        "dst",
        help="Path to the DST file",
    )
    parser.add_argument(
        "output",
        help="Path to the output JSON file",
    )
    args = parser.parse_args()

    dst_path = Path(args.dst)
    if not dst_path.exists():
        raise SystemExit(f"File not found: {dst_path}")

    ir = dst_to_ir(dst_path)
    Path(args.output).write_text(json.dumps(ir, indent=2), encoding="utf-8")
    print(f"Saved IR with {ir['stitch_count']} stitches to {args.output}")


if __name__ == "__main__":
    main()
