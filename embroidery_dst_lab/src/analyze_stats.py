#!/usr/bin/env python3
"""Compute statistics, previews and stitch-type heuristics from a Stitch IR."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

try:
    from shapely.geometry import MultiPoint
except ImportError:  # pragma: no cover
    MultiPoint = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore

from pyembroidery import JUMP, STITCH, TRIM

ANGLE_BINS = 12
ANGLE_RANGE = (-180, 180)


def _stitch_pairs(points: List[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for a, b in zip(points[:-1], points[1:]):
        yield a, b


def _layer_bounds(points: List[Dict[str, Any]]) -> Dict[str, float]:
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    return {"minx": min(xs), "miny": min(ys), "maxx": max(xs), "maxy": max(ys)}


def _layer_hull_area(points: List[Dict[str, Any]]) -> float:
    if MultiPoint is None or len(points) < 3:
        return 0.0
    hull = MultiPoint([(p["x"], p["y"]) for p in points]).convex_hull
    return float(hull.area)


def _layer_metrics(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    lengths: List[float] = []
    angles: List[float] = []
    for a, b in _stitch_pairs(points):
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        if a["cmd"] == STITCH and b["cmd"] == STITCH:
            lengths.append(math.hypot(dx, dy))
            angles.append(math.degrees(math.atan2(dy, dx)))
    histogram, _ = np.histogram(angles, bins=ANGLE_BINS, range=ANGLE_RANGE)
    return {
        "lengths": lengths,
        "angles": angles,
        "histogram": histogram.tolist(),
        "std_length": float(np.std(lengths)) if lengths else 0.0,
    }


def _coverage_ratio(bounds: Dict[str, float], area: float) -> float | None:
    bbox_area = (bounds["maxx"] - bounds["minx"]) * (bounds["maxy"] - bounds["miny"])
    if bbox_area <= 0:
        return None
    return area / bbox_area if area else 0.0


def _classify_layer(layer_stat: Dict[str, Any]) -> Dict[str, Any]:
    density = layer_stat.get("density_stitches_per_mm2")
    density = density if density is not None else 0.0
    avg_len = layer_stat.get("avg_stitch_length_mm", 0.0)
    std_len = layer_stat.get("std_stitch_length_mm", 0.0)
    coverage = layer_stat.get("coverage_ratio") or 0.0
    hist: List[int] = layer_stat.get("angle_histogram_deg") or []
    total = sum(hist)

    if total <= 0 or avg_len <= 0:
        return {
            "type": "unknown",
            "confidence": 0.0,
            "reason": "Insufficient stitch data for classification.",
        }

    dominant = max(hist)
    bins_used = sum(1 for value in hist if value > 0.05 * total)
    alignment_ratio = dominant / total if total else 0.0

    def result(type_name: str, confidence: float, reason: str) -> Dict[str, Any]:
        return {"type": type_name, "confidence": confidence, "reason": reason}

    if density < 0.15 and avg_len < 2.0:
        return result(
            "travel",
            0.8,
            f"Very low density ({density:.2f}) with short stitches ({avg_len:.2f} mm).",
        )

    if density < 0.4 and avg_len >= 2.3 and bins_used <= 2 and alignment_ratio >= 0.7:
        return result(
            "run",
            0.8,
            f"Low density ({density:.2f}), long stitches ({avg_len:.2f} mm) and aligned directions ({alignment_ratio:.2f}).",
        )

    if 0.35 <= density <= 0.8 and 2.0 <= avg_len <= 5.0 and bins_used <= 3 and 0.45 <= alignment_ratio <= 0.85:
        return result(
            "satin_light",
            0.7,
            f"Medium density ({density:.2f}) satin-like stitches ({avg_len:.2f} mm) with few directions ({bins_used}).",
        )

    if 0.8 < density <= 1.4 and 1.0 <= avg_len <= 3.5 and std_len <= 1.1 and bins_used <= 4:
        return result(
            "satin_dense",
            0.75,
            f"High density satin ({density:.2f}) with controlled lengths (avg {avg_len:.2f} mm, std {std_len:.2f}).",
        )

    if density > 0.75 and avg_len <= 4.5 and bins_used >= 4 and alignment_ratio < 0.7:
        return result(
            "tatami",
            0.75,
            f"Fill pattern: density {density:.2f}, multi-directional ({bins_used} bins) with shorter stitches ({avg_len:.2f} mm).",
        )

    if coverage < 0.2 and density < 0.5:
        return result(
            "detail",
            0.55,
            f"Low coverage ({coverage:.2f}) and sparse density ({density:.2f}) suggests detail/outline work.",
        )

    return result(
        "unknown",
        0.3,
        f"Metrics outside heuristics (density {density:.2f}, avg {avg_len:.2f} mm, bins {bins_used}, alignment {alignment_ratio:.2f}).",
    )


def compute_stats(ir: Dict[str, Any]) -> Dict[str, Any]:
    stitches = ir["stitches"]
    if not stitches:
        return {
            "stitch_count": 0,
            "avg_stitch_length_mm": 0.0,
            "min_stitch_length_mm": 0.0,
            "max_stitch_length_mm": 0.0,
            "total_path_length_mm": 0.0,
            "jump_like_segments": 0,
            "angle_histogram_deg": [],
            "layers": [],
            "layer_classification_summary": {},
            "layer_summaries": [],
        }

    lengths: List[float] = []
    angles: List[float] = []
    jump_like = 0

    for a, b in _stitch_pairs(stitches):
        dx = b["x"] - a["x"]
        dy = b["y"] - a["y"]
        dist = math.hypot(dx, dy)
        if a["cmd"] == STITCH and b["cmd"] == STITCH:
            lengths.append(dist)
            angles.append(math.degrees(math.atan2(dy, dx)))
        elif b["cmd"] in (JUMP, TRIM):
            jump_like += 1

    layers = defaultdict(list)
    for stitch in stitches:
        layers[stitch["color"]].append(stitch)

    layer_stats = []
    for color, pts in sorted(layers.items()):
        metrics = _layer_metrics(pts)
        layer_lengths = metrics["lengths"]
        avg_layer_length = float(np.mean(layer_lengths)) if layer_lengths else 0.0
        std_layer_length = metrics["std_length"]
        layer_path = float(sum(layer_lengths))
        area = _layer_hull_area(pts)
        bounds = _layer_bounds(pts)
        density = (len(pts) / area) if area > 0 else None
        coverage = _coverage_ratio(bounds, area)
        layer_entry = {
            "color": color,
            "stitch_count": len(pts),
            "path_length_mm": layer_path,
            "avg_stitch_length_mm": avg_layer_length,
            "std_stitch_length_mm": std_layer_length,
            "hull_area_mm2": area,
            "bounds": bounds,
            "density_stitches_per_mm2": density,
            "coverage_ratio": coverage,
            "angle_histogram_deg": metrics["histogram"],
        }
        layer_entry["classification"] = _classify_layer(layer_entry)
        density_str = (
            f"{density:.2f} pts/mm^2" if density is not None else "density N/A"
        )
        coverage_str = f"{coverage:.2f}" if coverage is not None else "cov N/A"
        cls = layer_entry["classification"]["type"]
        layer_entry["summary"] = (
            f"Layer {color} [{cls}] {len(pts)} sts, avg {avg_layer_length:.2f} mm, {density_str}, coverage {coverage_str}"
        )
        layer_stats.append(layer_entry)

    histogram, _ = np.histogram(angles, bins=ANGLE_BINS, range=ANGLE_RANGE)
    summary = Counter(layer["classification"]["type"] for layer in layer_stats)
    summary_dict = {k: v for k, v in summary.items() if v > 0}

    return {
        "stitch_count": ir["stitch_count"],
        "avg_stitch_length_mm": float(np.mean(lengths)) if lengths else 0.0,
        "min_stitch_length_mm": float(min(lengths)) if lengths else 0.0,
        "max_stitch_length_mm": float(max(lengths)) if lengths else 0.0,
        "total_path_length_mm": float(sum(lengths)),
        "jump_like_segments": jump_like,
        "angle_histogram_deg": histogram.tolist(),
        "layers": layer_stats,
        "layer_classification_summary": summary_dict,
        "layer_summaries": [layer["summary"] for layer in layer_stats],
    }


def plot_preview(
    layers: Dict[int, List[Dict[str, Any]]],
    output_path: Path,
    title: str = "DST Preview",
) -> None:
    if plt is None:
        print("matplotlib is not available, skipping preview.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    for color, pts in sorted(layers.items()):
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        ax.plot(xs, ys, linewidth=0.5, label=f"C{color}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved preview to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute statistics from a Stitch IR and optionally save a preview."
    )
    parser.add_argument(
        "--ir",
        required=True,
        help="Path to the Stitch IR JSON file",
    )
    parser.add_argument(
        "--stats-out",
        required=True,
        help="Where to save the statistics JSON",
    )
    parser.add_argument(
        "--preview-out",
        default=None,
        help="Where to save the preview PNG (optional).",
    )
    parser.add_argument(
        "--skip-preview",
        action="store_true",
        help="Disable preview generation even if matplotlib is available.",
    )
    args = parser.parse_args()

    ir_path = Path(args.ir)
    if not ir_path.exists():
        raise SystemExit(f"IR file not found: {ir_path}")

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    stats = compute_stats(ir)
    Path(args.stats_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved stats to {args.stats_out}")

    if not args.skip_preview and args.preview_out:
        layers = defaultdict(list)
        for stitch in ir["stitches"]:
            layers[stitch["color"]].append(stitch)
        if layers:
            plot_preview(layers, Path(args.preview_out), title=ir_path.stem)


if __name__ == "__main__":
    main()
