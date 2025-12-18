#!/usr/bin/env python3
"""Convert a DST file into a stitch intermediate representation (IR)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from pyembroidery import COLOR_CHANGE, END, read

DST_UNIT_MM = 0.1


def dst_to_ir(dst_path: Path) -> Dict[str, Any]:
    """Return a structured JSON-serialisable representation of a DST file."""
    pattern = read(str(dst_path))

    stitches: List[Dict[str, Any]] = []
    color_index = 0

    for i, (x, y, cmd) in enumerate(pattern.stitches):
        if cmd == COLOR_CHANGE:
            color_index += 1
        elif cmd == END:
            # pyembroidery still keeps the END instruction inside stitches,
            # but we avoid appending a bogus stitch entry.
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

    if not stitches:
        bounds = {"minx": 0.0, "miny": 0.0, "maxx": 0.0, "maxy": 0.0}
    else:
        bounds = {
            "minx": min(s["x"] for s in stitches),
            "miny": min(s["y"] for s in stitches),
            "maxx": max(s["x"] for s in stitches),
            "maxy": max(s["y"] for s in stitches),
        }

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
        description="Convert a DST file into a JSON Stitch IR."
    )
    parser.add_argument(
        "dst",
        nargs="?",
        default="embroidery_dst_lab/input/test.dst",
        help="Path to the DST file (default: embroidery_dst_lab/input/test.dst)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="embroidery_dst_lab/output/test_stitch_ir.json",
        help="Where to store the JSON IR (default: embroidery_dst_lab/output/test_stitch_ir.json)",
    )
    args = parser.parse_args()

    dst_path = Path(args.dst)
    if not dst_path.exists():
        raise SystemExit(f"File not found: {dst_path}")

    ir = dst_to_ir(dst_path)
    out_path = Path(args.output)
    out_path.write_text(json.dumps(ir, indent=2), encoding="utf-8")
    print(f"Saved IR with {ir['stitch_count']} stitches to {out_path}")


if __name__ == "__main__":
    main()
