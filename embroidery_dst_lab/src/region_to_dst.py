#!/usr/bin/env python3
"""Generate a multi-region DST from region_suggestions.json and their masks.

Uses the same simple tatami/satin generators as recipe_to_dst.py.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

try:
    import pyembroidery
except ImportError as exc:  # pragma: no cover
    raise SystemExit("pyembroidery is required: pip install pyembroidery") from exc


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: Path, threshold: int = 128) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L")
    mask = (np.asarray(img) >= threshold).astype(np.uint8)
    if mask.sum() == 0:
        raise SystemExit(f"Mask has no foreground pixels above threshold {threshold}: {path}")
    return mask


def hatch_fill(mask: np.ndarray, step_px: float, line_spacing_px: float, angle_deg: float) -> List[Tuple[float, float]]:
    """Generate hatch lines over the mask at given angle."""
    h, w = mask.shape
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    y_coords, x_coords = np.nonzero(mask)
    if len(x_coords) == 0:
        return []
    xr = x_coords * cos_a + y_coords * sin_a
    yr = -x_coords * sin_a + y_coords * cos_a
    min_xr, max_xr = xr.min(), xr.max()

    lines = []
    x_vals = np.arange(min_xr, max_xr + line_spacing_px, line_spacing_px)
    for x0 in x_vals:
        mask_line = np.abs(xr - x0) <= line_spacing_px / 2
        if not np.any(mask_line):
            continue
        yr_line = yr[mask_line]
        sorted_idx = np.argsort(yr_line)
        yr_sorted = yr_line[sorted_idx]
        xr_sorted = xr[mask_line][sorted_idx]
        x_img = xr_sorted * cos_a - yr_sorted * sin_a
        y_img = xr_sorted * sin_a + yr_sorted * cos_a
        if len(x_img) == 0:
            continue
        dist = np.sqrt(np.diff(x_img) ** 2 + np.diff(y_img) ** 2)
        keep = [0]
        acc = 0.0
        for i, d in enumerate(dist, start=1):
            acc += d
            if acc >= step_px:
                keep.append(i)
                acc = 0.0
        x_keep = x_img[keep]
        y_keep = y_img[keep]
        if len(lines) % 2 == 1:
            x_keep = x_keep[::-1]
            y_keep = y_keep[::-1]
        lines.extend(list(zip(x_keep, y_keep)))
    return lines


def extract_outline(mask: np.ndarray) -> np.ndarray:
    mask_int = mask.astype(np.uint8)
    padded = np.pad(mask_int, 1)
    neigh_sum = (
        mask_int
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[:-2, :-2]
        + padded[:-2, 2:]
        + padded[2:, :-2]
        + padded[2:, 2:]
    )
    erosion = (neigh_sum == 9).astype(np.uint8)
    outline = (mask_int == 1) & (erosion == 0)
    return outline.astype(np.uint8)


def outline_to_stitches(outline: np.ndarray, step_px: int = 1) -> List[Tuple[float, float]]:
    ys, xs = np.nonzero(outline)
    if len(xs) == 0:
        return []
    idx = np.lexsort((xs, ys))
    pts = [(xs[i], ys[i]) for i in idx[:: max(step_px, 1)]]
    return pts


def write_svg(stitches: List[Dict], path: Path) -> None:
    if not stitches:
        return
    xs = [s["x_mm"] for s in stitches]
    ys = [s["y_mm"] for s in stitches]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    margin = 2.0
    width = (max_x - min_x) + margin * 2
    height = (max_y - min_y) + margin * 2
    if width <= 0 or height <= 0:
        width = height = 10

    segments: Dict[int, List[List[Tuple[float, float]]]] = {}
    last_color = None
    for s in stitches:
        color = int(s["color"])
        pt = (s["x_mm"] - min_x + margin, s["y_mm"] - min_y + margin)
        if color != last_color:
            segments.setdefault(color, []).append([])
        segments[color][-1].append(pt)
        last_color = color

    palette = ["#e4572e", "#17bebb", "#ffc914", "#2e282a", "#76b041", "#5a189a", "#1e88e5"]
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.2f} {height:.2f}" '
        f'width="{width:.2f}mm" height="{height:.2f}mm">',
        '<rect width="100%" height="100%" fill="white" stroke="none"/>',
    ]
    for idx, (color, segs) in enumerate(segments.items()):
        stroke = palette[idx % len(palette)]
        for seg in segs:
            if len(seg) < 2:
                continue
            pts_str = " ".join(f"{x:.2f},{y:.2f}" for x, y in seg)
            lines.append(
                f'<polyline fill="none" stroke="{stroke}" stroke-width="0.4" points="{pts_str}" '
                f'data-color="{color}"/>'
            )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def build_pattern(region_entries: List[Dict], pixel_size_mm: float, threshold: int) -> Tuple[pyembroidery.EmbPattern, List[Dict]]:
    pattern = pyembroidery.EmbPattern()
    stitches: List[Dict] = []
    color_index = 0
    for region in region_entries:
        mask_path = region.get("mask")
        if not mask_path:
            continue
        mask = load_mask(Path(mask_path), threshold=threshold)
        for layer in region.get("layers", []):
            cls = (layer.get("classification") or {}).get("type", "tatami")
            density = layer.get("density_stitches_per_mm2") or 0.6
            avg_len = layer.get("avg_stitch_length_mm") or 2.0
            angle = layer.get("angle_deg") if layer.get("angle_deg") is not None else 0.0

            if cls == "satin":
                outline = extract_outline(mask)
                step_px = max(1, int(round((avg_len / pixel_size_mm) * 0.5)))
                pts = outline_to_stitches(outline, step_px=step_px)
            else:
                step_mm = float(avg_len)
                line_spacing_px = max(math.sqrt(1.0 / max(density, 1e-3)) / pixel_size_mm, (step_mm * 0.5) / pixel_size_mm)
                pts = hatch_fill(
                    mask,
                    step_px=step_mm / pixel_size_mm,
                    line_spacing_px=line_spacing_px,
                    angle_deg=angle,
                )
            if not pts:
                continue
            pts_mm = [(x * pixel_size_mm, y * pixel_size_mm) for x, y in pts]
            if color_index > 0:
                pattern.add_command(pyembroidery.COLOR_CHANGE)
            color_index += 1
            pattern.add_thread(pyembroidery.EmbThread())
            for x, y in pts_mm:
                pattern.add_stitch_absolute(x, y, pyembroidery.STITCH)
                stitches.append({"color": color_index, "x_mm": float(x), "y_mm": float(y)})
    pattern.end()
    return pattern, stitches


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate multi-region DST from region_suggestions.json (per-region recipes)."
    )
    parser.add_argument(
        "--region-suggestions",
        required=True,
        help="Path to region_suggestions.json (output of region_suggest_recipe.py).",
    )
    parser.add_argument(
        "--pixel-size-mm",
        type=float,
        default=0.2,
        help="Physical size of one pixel in mm (default: 0.2).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=128,
        help="Mask threshold (default 128).",
    )
    parser.add_argument(
        "--output-dst",
        default="embroidery_dst_lab/output/regions_suggested.dst",
        help="Where to save the DST file.",
    )
    parser.add_argument(
        "--output-ir",
        default="embroidery_dst_lab/output/regions_suggested_ir.json",
        help="Where to save the IR-like JSON.",
    )
    parser.add_argument(
        "--output-svg",
        default=None,
        help="Optional SVG preview path.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of human-readable summary.")
    args = parser.parse_args()

    payload = load_json(Path(args.region_suggestions))
    regions = payload.get("regions", [])
    if not regions:
        raise SystemExit("No regions found in region_suggestions.")

    pattern, stitches = build_pattern(regions, pixel_size_mm=args.pixel_size_mm, threshold=args.threshold)

    # Coerce flags to int for pyembroidery encoder
    coerced = []
    for cmd in pattern.stitches:
        if len(cmd) == 3:
            x, y, flag = cmd
        else:
            flag = cmd[0]
            x = cmd[1] if len(cmd) > 1 else 0
            y = cmd[2] if len(cmd) > 2 else 0
        coerced.append((float(x), float(y), int(flag)))
    pattern.stitches = coerced

    out_dst = Path(args.output_dst)
    out_dst.parent.mkdir(parents=True, exist_ok=True)
    pyembroidery.write_dst(pattern, str(out_dst))

    ir = {
        "regions": regions,
        "stitches": stitches,
        "pixel_size_mm": args.pixel_size_mm,
    }
    out_ir = Path(args.output_ir)
    out_ir.parent.mkdir(parents=True, exist_ok=True)
    out_ir.write_text(json.dumps(ir, indent=2), encoding="utf-8")

    if args.output_svg:
        write_svg(stitches, Path(args.output_svg))

    if args.json:
        print(json.dumps(ir, indent=2))
    else:
        print(f"Generated DST: {out_dst} stitches: {len(stitches)}")
        print(f"IR saved to: {out_ir}")
        if args.output_svg:
            print(f"SVG saved to: {args.output_svg}")


if __name__ == "__main__":
    main()
