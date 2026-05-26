import base64
import io
import math

import numpy as np
from PIL import Image


status_callback = None


def extract_source_dpi(image_info):
    """
    Estrae DPI da metadati comuni (dpi o JFIF).
    Ritorna (dpi_x, dpi_y) oppure None.
    """
    dpi = image_info.get("dpi")
    if isinstance(dpi, (tuple, list)) and len(dpi) >= 2:
        dx = float(dpi[0])
        dy = float(dpi[1])
        if dx > 0 and dy > 0:
            return (dx, dy)
    elif isinstance(dpi, (int, float)):
        d = float(dpi)
        if d > 0:
            return (d, d)

    jfif_unit = image_info.get("jfif_unit")
    jfif_density = image_info.get("jfif_density")
    if (
        isinstance(jfif_density, (tuple, list))
        and len(jfif_density) >= 2
        and jfif_density[0] > 0
        and jfif_density[1] > 0
    ):
        x = float(jfif_density[0])
        y = float(jfif_density[1])
        if jfif_unit == 1:  # dots per inch
            return (x, y)
        if jfif_unit == 2:  # dots per cm
            return (x * 2.54, y * 2.54)

    return None


def parse_hex_color(color_text):
    s = str(color_text).strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Colore non valido: {color_text}")
    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError as exc:
        raise ValueError(f"Colore non valido: {color_text}") from exc
    return (r, g, b)


def parse_sample_colors(raw_value):
    if raw_value is None:
        return []
    text = str(raw_value).strip()
    if not text:
        return []
    parts = [p.strip() for p in text.replace(";", ",").split(",")]
    colors = []
    for part in parts:
        if not part:
            continue
        colors.append(parse_hex_color(part))
    return colors


def build_color_match_mask(rgb_arr, target_colors, tolerance):
    tol = max(0.0, float(tolerance))
    tol2 = tol * tol
    h, w, _ = rgb_arr.shape
    mask = np.zeros((h, w), dtype=bool)
    for (tr, tg, tb) in target_colors:
        dr = rgb_arr[:, :, 0].astype(np.float32) - float(tr)
        dg = rgb_arr[:, :, 1].astype(np.float32) - float(tg)
        db = rgb_arr[:, :, 2].astype(np.float32) - float(tb)
        d2 = dr * dr + dg * dg + db * db
        mask |= d2 <= tol2
    return mask


def build_palette_from_selected_pixels(
    rgb_arr, mask, color_count, max_palette_sample=200000
):
    selected_rgb = rgb_arr[mask].astype(np.uint8)
    if selected_rgb.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    palette_size = max(1, int(color_count))
    sample = selected_rgb
    if len(sample) > max_palette_sample:
        idx = np.linspace(0, len(sample) - 1, num=max_palette_sample, dtype=int)
        sample = sample[idx]

    sample_img = Image.fromarray(sample.reshape((len(sample), 1, 3)), mode="RGB")
    quantized = sample_img.convert(
        "P", palette=Image.ADAPTIVE, colors=palette_size
    )
    raw_palette = quantized.getpalette() or []

    palette = []
    seen = set()
    for idx in range(palette_size):
        base = idx * 3
        if base + 2 >= len(raw_palette):
            break
        color = (
            int(raw_palette[base]),
            int(raw_palette[base + 1]),
            int(raw_palette[base + 2]),
        )
        if color in seen:
            continue
        seen.add(color)
        palette.append(color)

    if not palette:
        mean_color = selected_rgb.mean(axis=0).round().astype(int)
        palette.append(tuple(int(v) for v in mean_color))

    return np.asarray(palette, dtype=np.float32)


def color_distance2(a, b):
    diff = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return float(np.sum(diff * diff))


def weighted_color_average(colors, weights):
    total = float(sum(weights))
    if total <= 0:
        return tuple(int(v) for v in colors[0])
    acc = np.zeros(3, dtype=np.float64)
    for color, weight in zip(colors, weights):
        acc += np.asarray(color, dtype=np.float64) * float(weight)
    return tuple(int(round(max(0, min(255, v)))) for v in (acc / total))


def build_reduced_palette(rgb_arr, mask, target_count, priority_colors=None):
    target_count = max(1, int(target_count))
    priority = []
    for color in priority_colors or []:
        normalized = tuple(int(v) for v in color)
        if normalized not in priority:
            priority.append(normalized)

    if len(priority) >= target_count:
        selected_rgb = rgb_arr[mask].astype(np.float32)
        ranked = []
        for color in priority:
            if len(selected_rgb):
                diff = selected_rgb - np.asarray(color, dtype=np.float32)
                distances = np.sum(diff * diff, axis=1)
                score = float(np.min(distances))
            else:
                score = 0.0
            ranked.append((score, color))
        ranked.sort(key=lambda item: item[0])
        return np.asarray([color for _, color in ranked[:target_count]], dtype=np.float32)

    candidate_count = max(target_count * 6, target_count + 10, 12)
    candidate_count = min(candidate_count, 64)
    candidates = build_palette_from_selected_pixels(rgb_arr, mask, candidate_count)
    candidate_colors = [tuple(int(v) for v in color) for color in candidates.astype(np.uint8)]

    ys, xs = np.where(mask)
    selected_rgb = rgb_arr[ys, xs].astype(np.float32)
    weighted_candidates = []
    if candidate_colors and len(selected_rgb):
        palette = np.asarray(candidate_colors, dtype=np.float32)
        labels = np.empty(len(selected_rgb), dtype=np.int32)
        chunk_size = 250000
        for start in range(0, len(selected_rgb), chunk_size):
            end = min(start + chunk_size, len(selected_rgb))
            chunk = selected_rgb[start:end]
            diff = chunk[:, None, :] - palette[None, :, :]
            distances = np.sum(diff * diff, axis=2)
            labels[start:end] = np.argmin(distances, axis=1)
        for idx, color in enumerate(candidate_colors):
            weight = int(np.count_nonzero(labels == idx))
            if weight > 0:
                weighted_candidates.append({"color": color, "weight": weight, "protected": False})

    if not weighted_candidates and not priority:
        return build_palette_from_selected_pixels(rgb_arr, mask, target_count)

    selected = [{"color": color, "weight": 1, "protected": True} for color in priority]
    pool = [item for item in weighted_candidates if item["color"] not in priority]

    if not selected and pool:
        first = max(pool, key=lambda item: item["weight"])
        selected.append(first)
        pool.remove(first)

    max_weight = max([item["weight"] for item in weighted_candidates] or [1])
    while len(selected) < target_count and pool:
        best_idx = None
        best_score = None
        for idx, item in enumerate(pool):
            nearest_dist2 = min(
                color_distance2(item["color"], chosen["color"]) for chosen in selected
            ) if selected else 1.0
            # Keep large areas relevant, but make hue diversity strong enough that
            # small distinct colors like yellow/green survive large blue regions.
            weight_factor = 0.25 + 0.75 * math.sqrt(item["weight"] / float(max_weight))
            score = nearest_dist2 * weight_factor
            if best_score is None or score > best_score:
                best_score = score
                best_idx = idx

        selected.append(pool.pop(best_idx))

    return np.asarray([item["color"] for item in selected[:target_count]], dtype=np.float32)


def group_mask_points_by_palette(rgb_arr, mask, color_count):
    palette = build_palette_from_selected_pixels(rgb_arr, mask, color_count)
    return group_mask_points_with_palette(rgb_arr, mask, palette)


def group_points_with_priority_colors(
    rgb_arr, mask, color_count, priority_colors=None, priority_tolerance=0.0
):
    target_count = max(1, int(color_count))
    palette = build_reduced_palette(
        rgb_arr,
        mask,
        target_count,
        priority_colors=priority_colors,
    )
    return group_mask_points_with_palette(rgb_arr, mask, palette)


def group_mask_points_with_palette(rgb_arr, mask, palette):
    ys, xs = np.where(mask)
    if len(palette) == 0:
        return {}

    selected_rgb = rgb_arr[ys, xs].astype(np.float32)
    if len(palette) == 1:
        labels = np.zeros(len(selected_rgb), dtype=np.int32)
    else:
        diff = selected_rgb[:, None, :] - palette[None, :, :]
        distances = np.sum(diff * diff, axis=2)
        labels = np.argmin(distances, axis=1)

    color_points = {}
    for palette_idx, color in enumerate(palette.astype(np.uint8)):
        selected = labels == palette_idx
        if not np.any(selected):
            continue
        color_hex = f"#{color[0]:02X}{color[1]:02X}{color[2]:02X}"
        px = xs[selected]
        py = ys[selected]
        color_points[color_hex] = list(zip(px.tolist(), py.tolist()))

    return color_points


def label_mask_pixels_with_palette(rgb_arr, mask, palette):
    ys, xs = np.where(mask)
    if len(palette) == 0 or len(xs) == 0:
        return ys, xs, np.zeros(0, dtype=np.int32)

    selected_rgb = rgb_arr[ys, xs].astype(np.float32)
    if len(palette) == 1:
        labels = np.zeros(len(selected_rgb), dtype=np.int32)
    else:
        labels = np.empty(len(selected_rgb), dtype=np.int32)
        chunk_size = 250000
        for start in range(0, len(selected_rgb), chunk_size):
            end = min(start + chunk_size, len(selected_rgb))
            chunk = selected_rgb[start:end]
            diff = chunk[:, None, :] - palette[None, :, :]
            distances = np.sum(diff * diff, axis=2)
            labels[start:end] = np.argmin(distances, axis=1)
    return ys, xs, labels


def png_base64_from_array(arr):
    buffer = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_preview_images(rgb_arr, mask, palette):
    h, w, _ = rgb_arr.shape
    ys, xs, labels = label_mask_pixels_with_palette(rgb_arr, mask, palette)

    overlay = rgb_arr.astype(np.float32)
    color_layer = overlay.copy()
    mask_preview = np.full((h, w, 3), 245, dtype=np.uint8)
    color_masks = []
    counts = []

    for palette_idx, color in enumerate(palette.astype(np.uint8)):
        selected = labels == palette_idx
        count = int(np.count_nonzero(selected))
        counts.append(count)
        if count == 0:
            color_masks.append("")
            continue

        px = xs[selected]
        py = ys[selected]
        color_layer[py, px] = color
        mask_preview[py, px] = color

        single = np.full((h, w, 4), 0, dtype=np.uint8)
        single[py, px, :3] = color
        single[py, px, 3] = 255
        color_masks.append(png_base64_from_array(single))

    selected_mask = mask[:, :, None]
    overlay[selected_mask.repeat(3, axis=2)] = (
        overlay[selected_mask.repeat(3, axis=2)] * 0.35
        + color_layer[selected_mask.repeat(3, axis=2)] * 0.65
    )

    return {
        "overlay_png_base64": png_base64_from_array(overlay),
        "mask_png_base64": png_base64_from_array(mask_preview),
        "color_mask_pngs": color_masks,
        "counts": counts,
    }


def build_preview_from_color_points(rgb_arr, color_points):
    h, w, _ = rgb_arr.shape
    overlay = rgb_arr.astype(np.float32)
    mask_preview = np.full((h, w, 3), 245, dtype=np.uint8)
    color_payload = []

    for color_hex, points in color_points.items():
        if not points:
            continue
        r, g, b = parse_hex_color(color_hex)
        color = np.array([r, g, b], dtype=np.uint8)
        xs = np.array([p[0] for p in points], dtype=np.int32)
        ys = np.array([p[1] for p in points], dtype=np.int32)

        mask_preview[ys, xs] = color
        overlay[ys, xs] = overlay[ys, xs] * 0.35 + color.astype(np.float32) * 0.65

        single = np.full((h, w, 4), 0, dtype=np.uint8)
        single[ys, xs, :3] = color
        single[ys, xs, 3] = 255
        color_payload.append(
            {
                "color": color_hex,
                "pixel_count": int(len(points)),
                "mask_png_base64": png_base64_from_array(single),
            }
        )

    return {
        "overlay_png_base64": png_base64_from_array(overlay),
        "mask_png_base64": png_base64_from_array(mask_preview),
        "colors": color_payload,
    }


def build_points_preview(rgb_arr, color_points):
    h, w, _ = rgb_arr.shape
    preview = (
        rgb_arr.astype(np.float32) * 0.18
        + np.full((h, w, 3), 245, dtype=np.float32) * 0.82
    ).astype(np.uint8)
    radius = max(1, int(round(max(w, h) / 650.0)))
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if (dx * dx + dy * dy) <= radius * radius:
                offsets.append((dx, dy))

    total_points = 0
    for color_hex, points in color_points.items():
        if not points:
            continue
        total_points += len(points)
        color = np.array(parse_hex_color(color_hex), dtype=np.uint8)
        for x, y in points:
            xi = int(round(x))
            yi = int(round(y))
            for dx, dy in offsets:
                px = xi + dx
                py = yi + dy
                if 0 <= px < w and 0 <= py < h:
                    preview[py, px] = color

    return {
        "points_png_base64": png_base64_from_array(preview),
        "preview_points": int(total_points),
    }


def resolve_dpi_for_mm(effective_dpi, default_dpi=96.0):
    dpi_for_mm = float(default_dpi or 96.0)
    if isinstance(effective_dpi, (tuple, list)) and len(effective_dpi) >= 2:
        dpi_for_mm = (float(effective_dpi[0]) + float(effective_dpi[1])) / 2.0
    elif isinstance(effective_dpi, (int, float)) and effective_dpi > 0:
        dpi_for_mm = float(effective_dpi)
    return dpi_for_mm


def prepare_points_before_ordering(
    points,
    opts,
    image_size,
    effective_dpi,
    color_hex,
    color_idx=0,
    seed_base=None,
    max_points_for_color=0,
    report=False,
):
    working = points[:]
    dpi_for_mm = resolve_dpi_for_mm(
        effective_dpi, float(opts.get("default_dpi", 96.0) or 96.0)
    )
    analysis_cell_mm = float(opts.get("analysis_cell_mm", 0.0) or 0.0)

    if analysis_cell_mm >= 1.0 and opts["scale"] > 0:
        cell_px = analysis_cell_mm * (dpi_for_mm / 25.4) / float(opts["scale"])
        if cell_px > 1.0:
            before = len(working)
            working = regularize_points_on_grid(working, cell_px)
            if report:
                emit_status(
                    f"{color_hex}: analisi griglia metrica {analysis_cell_mm:.2f} mm ({cell_px:.2f} px) -> {len(working)} punti (prima {before})"
                )
        elif report:
            emit_status(
                f"{color_hex}: analysis-cell {analysis_cell_mm:.2f} mm troppo fine rispetto alla scala corrente ({opts['scale']:.4f}); nessuna regolarizzazione metrica."
            )

    if opts["style"] == "degrade":
        seed = (seed_base + color_idx) if seed_base is not None else None
        if report:
            emit_status(
                f"{color_hex}: applico stile degrade (drop={opts['degrade_drop']:.2f}, jitter={opts['degrade_jitter']:.2f})"
            )
        working = apply_random_degrade_effects(
            working,
            drop_probability=opts["degrade_drop"],
            jitter=opts["degrade_jitter"],
            image_size=image_size,
            seed=seed,
        )
    elif opts["grid_cell_size"] > 1:
        if report:
            emit_status(
                f"{color_hex}: regolarizzo griglia (cell={opts['grid_cell_size']} px)"
            )
        working = regularize_points_on_grid(working, opts["grid_cell_size"])

    max_points = int(max_points_for_color) if max_points_for_color else 0
    if max_points > 0 and len(working) > max_points:
        before = len(working)
        working = subsample_points(working, max_points=max_points)
        if report:
            emit_status(
                f"{color_hex}: limito max-points a {len(working)} (prima {before})"
            )

    return working


def analyze_preview(image_bytes, opts):
    if not isinstance(image_bytes, (bytes, bytearray)):
        image_bytes = bytes(image_bytes)

    max_width = opts["max_width"] if opts["max_width"] > 0 else None
    color_count = max(1, int(opts.get("color_count", 1)))
    sample_colors = parse_sample_colors(opts.get("sample_colors", ""))
    sample_tolerance = float(opts.get("sample_tolerance", 0.0) or 0.0)
    exclude_background = bool(opts.get("exclude_background", False))
    background_colors = parse_sample_colors(opts.get("background_colors", ""))
    background_tolerance = float(opts.get("background_tolerance", 0.0) or 0.0)

    original = Image.open(io.BytesIO(image_bytes))
    source_dpi = extract_source_dpi(original.info)
    img = original.convert("RGBA")
    resize_scale = 1.0
    if max_width is not None and max_width > 0:
        w, h = img.size
        if w > max_width:
            scale = max_width / float(w)
            resize_scale = scale
            img = img.resize((max_width, int(h * scale)), Image.Resampling.LANCZOS)

    effective_dpi = source_dpi
    if effective_dpi is not None and resize_scale != 1.0:
        effective_dpi = (
            float(effective_dpi[0]) * resize_scale,
            float(effective_dpi[1]) * resize_scale,
        )

    rgb_arr = np.array(img.convert("RGB"))
    luminance = np.array(img.convert("L"))
    mask = luminance < int(opts["threshold"])
    if sample_colors:
        mask |= build_color_match_mask(rgb_arr, sample_colors, sample_tolerance)
    if exclude_background and background_colors:
        mask &= ~build_color_match_mask(
            rgb_arr, background_colors, background_tolerance
        )
    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A"))
        mask &= alpha > 0

    total_pixels = int(mask.size)
    selected_pixels = int(np.count_nonzero(mask))
    if selected_pixels == 0:
        raise ValueError(
            "Nessun pixel utile trovato per la preview. Modifica soglia, colori o sfondo escluso."
        )

    color_points = group_points_with_priority_colors(
        rgb_arr,
        mask,
        color_count,
        priority_colors=sample_colors,
        priority_tolerance=sample_tolerance,
    )
    preview = build_preview_from_color_points(rgb_arr, color_points)
    base_seed = None
    if opts["style"] == "degrade":
        seed_raw = opts.get("degrade_seed")
        if seed_raw not in (None, ""):
            try:
                base_seed = int(seed_raw)
            except (TypeError, ValueError):
                base_seed = None
    budget_info = resolve_global_point_budget(
        color_points,
        opts.get("max_points", 0),
        opts.get("target_density", 0.0),
    )
    per_color_max = allocate_max_points_by_color(color_points, budget_info["budget"])
    preview_point_groups = {}
    for idx, color_hex in enumerate(sorted(color_points.keys())):
        preview_point_groups[color_hex] = prepare_points_before_ordering(
            color_points[color_hex],
            opts,
            img.size,
            effective_dpi,
            color_hex,
            color_idx=idx,
            seed_base=base_seed,
            max_points_for_color=per_color_max.get(color_hex, 0),
            report=False,
        )
    preview.update(build_points_preview(rgb_arr, preview_point_groups))
    colors = []
    for info in preview.pop("colors"):
        count = int(info["pixel_count"])
        info["area_pct"] = (100.0 * count / total_pixels) if total_pixels else 0.0
        colors.append(info)

    preview.update(
        {
            "width": int(img.size[0]),
            "height": int(img.size[1]),
            "selected_pixels": selected_pixels,
            "selected_pct": (100.0 * selected_pixels / total_pixels)
            if total_pixels
            else 0.0,
            "colors": colors,
        }
    )
    return preview


def emit_status(message):
    global status_callback
    if status_callback is not None:
        status_callback(str(message))


def emit_progress(percent, message=""):
    pct = max(0.0, min(100.0, float(percent)))
    emit_status(f"__PROGRESS__|{pct:.1f}|{message}")


def load_points_grouped_from_bytes(
    data,
    max_width=None,
    threshold=200,
    color_count=2,
    sample_colors=None,
    sample_tolerance=0.0,
    exclude_background=False,
    background_colors=None,
    background_tolerance=0.0,
):
    original = Image.open(io.BytesIO(data))
    source_dpi = extract_source_dpi(original.info)
    img = original.convert("RGBA")
    resize_scale = 1.0

    if max_width is not None and max_width > 0:
        w, h = img.size
        if w > max_width:
            scale = max_width / float(w)
            resize_scale = scale
            new_w = max_width
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    effective_dpi = source_dpi
    if effective_dpi is not None and resize_scale != 1.0:
        effective_dpi = (
            float(effective_dpi[0]) * resize_scale,
            float(effective_dpi[1]) * resize_scale,
        )

    rgb_arr = np.array(img.convert("RGB"))
    luminance = np.array(img.convert("L"))
    mask = luminance < int(threshold)
    if sample_colors:
        # I colori campionati si aggiungono alla selezione base, non la sostituiscono.
        mask |= build_color_match_mask(rgb_arr, sample_colors, sample_tolerance)
    if exclude_background and background_colors:
        mask &= ~build_color_match_mask(
            rgb_arr, background_colors, background_tolerance
        )
    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A"))
        mask &= alpha > 0

    if not np.any(mask):
        emit_status(
            f"Nessun pixel valido trovato con soglia={threshold}; aumenta la soglia o usa un'immagine diversa."
        )
        return {}, img.size, effective_dpi

    palette_colors = max(1, int(color_count))
    color_points = group_points_with_priority_colors(
        rgb_arr,
        mask,
        palette_colors,
        priority_colors=sample_colors,
        priority_tolerance=sample_tolerance,
    )

    emit_status(
        "Lettura bitmap completata: "
        f"{img.size[0]}x{img.size[1]} px, colori attivi={len(color_points)}"
    )
    return color_points, img.size, effective_dpi


def subsample_points(points, max_points=None):
    n = len(points)
    if (max_points is None) or (max_points <= 0) or (max_points >= n):
        return points
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [points[i] for i in idx]


def allocate_max_points_by_color(color_points, global_max_points):
    quotas = {color_hex: len(points) for color_hex, points in color_points.items()}
    if global_max_points is None or global_max_points <= 0:
        return quotas

    available = {color_hex: len(points) for color_hex, points in color_points.items() if points}
    if not available:
        return quotas

    total_available = sum(available.values())
    budget = min(int(global_max_points), total_available)
    if budget >= total_available:
        return quotas

    for color_hex in quotas.keys():
        quotas[color_hex] = 0

    color_order = sorted(available.keys(), key=lambda c: available[c], reverse=True)
    active_colors = len(color_order)

    if budget < active_colors:
        for color_hex in color_order[:budget]:
            quotas[color_hex] = 1
        return quotas

    for color_hex in color_order:
        quotas[color_hex] = 1
    remaining_budget = budget - active_colors

    if remaining_budget <= 0:
        return quotas

    residual_capacity = {
        color_hex: max(0, available[color_hex] - quotas[color_hex])
        for color_hex in color_order
    }
    residual_total = sum(residual_capacity.values())
    if residual_total <= 0:
        return quotas

    remainders = []
    distributed = 0
    for color_hex in color_order:
        share = remaining_budget * (residual_capacity[color_hex] / float(residual_total))
        add = int(math.floor(share))
        add = min(add, residual_capacity[color_hex])
        quotas[color_hex] += add
        distributed += add
        remainders.append((share - add, color_hex))

    leftover = remaining_budget - distributed
    if leftover > 0:
        remainders.sort(key=lambda x: x[0], reverse=True)
        for _, color_hex in remainders:
            if leftover <= 0:
                break
            headroom = available[color_hex] - quotas[color_hex]
            if headroom <= 0:
                continue
            quotas[color_hex] += 1
            leftover -= 1

    return quotas


def resolve_global_point_budget(color_points, max_points_raw, target_density_raw):
    total_available = sum(len(points) for points in color_points.values())
    max_points = int(max_points_raw or 0)
    try:
        density = float(target_density_raw or 0.0)
    except (TypeError, ValueError):
        density = 0.0
    density = max(0.0, min(100.0, density))

    budget_from_density = 0
    if density > 0.0 and total_available > 0:
        budget_from_density = int(
            math.ceil(total_available * (density / 100.0))
        )
        budget_from_density = min(budget_from_density, total_available)

    if budget_from_density > 0 and max_points > 0:
        budget = min(max_points, budget_from_density)
        mode = "density+cap"
    elif budget_from_density > 0:
        budget = budget_from_density
        mode = "density"
    else:
        budget = max_points
        mode = "max_points" if max_points > 0 else "none"

    return {
        "total_available": total_available,
        "density": density,
        "budget_from_density": budget_from_density,
        "budget": max(0, int(budget or 0)),
        "mode": mode,
    }


def order_points_nearest_neighbor(points):
    n = len(points)
    if n == 0:
        return []
    used = [False] * n
    ordered = []
    start_idx = 0
    min_x = points[0][0]
    for i in range(1, n):
        if points[i][0] < min_x:
            min_x = points[i][0]
            start_idx = i
    current_idx = start_idx
    used[current_idx] = True
    ordered.append(points[current_idx])
    for _ in range(1, n):
        cx, cy = points[current_idx]
        best_idx = None
        best_d2 = None
        for j, (px, py) in enumerate(points):
            if used[j]:
                continue
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                best_idx = j
        if best_idx is None:
            break
        used[best_idx] = True
        ordered.append(points[best_idx])
        current_idx = best_idx
    return ordered


def order_points_scanline(points, band_height=4, serpentine=True):
    if not points:
        return []
    bh = int(band_height)
    if bh <= 0:
        bh = 1
    buckets = {}
    for x, y in points:
        band = int(y // bh)
        buckets.setdefault(band, []).append((x, y))
    ordered = []
    for idx, band in enumerate(sorted(buckets.keys())):
        row_pts = sorted(buckets[band], key=lambda p: p[0])
        if serpentine and (idx % 2 == 1):
            row_pts.reverse()
        ordered.extend(row_pts)
    return ordered


def regularize_points_on_grid(points, cell_size):
    cs = float(cell_size)
    if cs <= 1.0 or not points:
        return points
    buckets = {}
    for x, y in points:
        gx = int(x // cs)
        gy = int(y // cs)
        buckets.setdefault((gx, gy), 0)
        buckets[(gx, gy)] += 1
    regularized = []
    half = cs / 2.0
    for (gx, gy) in sorted(buckets.keys(), key=lambda k: (k[1], k[0])):
        regularized.append((gx * cs + half, gy * cs + half))
    return regularized


def apply_random_degrade_effects(
    points, drop_probability=0.0, jitter=0.0, image_size=None, seed=None
):
    if not points:
        return points
    drop_probability = max(0.0, min(1.0, float(drop_probability)))
    jitter = float(jitter)
    if drop_probability <= 0.0 and jitter <= 0.0:
        return points
    rng = np.random.default_rng(seed)
    w = image_size[0] if image_size else None
    h = image_size[1] if image_size else None
    new_points = []
    for x, y in points:
        if drop_probability > 0.0 and rng.random() < drop_probability:
            continue
        nx = x
        ny = y
        if jitter > 0.0:
            nx = x + rng.uniform(-jitter, jitter)
            ny = y + rng.uniform(-jitter, jitter)
            if w is not None:
                nx = max(0.0, min(w - 1, nx))
            if h is not None:
                ny = max(0.0, min(h - 1, ny))
        new_points.append((nx, ny))
    return new_points


def filter_with_min_dist_and_standby(points, min_dist_px):
    if not points or min_dist_px <= 0:
        return points[:], []
    path_filtered = [points[0]]
    standby = []
    last_x, last_y = points[0]
    min2 = min_dist_px * min_dist_px
    for (x, y) in points[1:]:
        dx = x - last_x
        dy = y - last_y
        d2 = dx * dx + dy * dy
        if d2 >= min2:
            path_filtered.append((x, y))
            last_x, last_y = x, y
        else:
            standby.append((x, y))
    return path_filtered, standby


def try_reinsert_points(path, standby, min_dist_px):
    if not standby or not path:
        return path, standby
    min2 = min_dist_px * min_dist_px
    new_path = path[:]
    still_leftover = []

    def can_insert_between(p_prev, p, p_next):
        if p_prev is not None:
            dx = p[0] - p_prev[0]
            dy = p[1] - p_prev[1]
            if dx * dx + dy * dy < min2:
                return False
        if p_next is not None:
            dx = p[0] - p_next[0]
            dy = p[1] - p_next[1]
            if dx * dx + dy * dy < min2:
                return False
        return True

    def extra_length_for_insertion(path_local, s, pos):
        if len(path_local) == 0:
            return 0.0
        if pos == 0:
            old_first = path_local[0]
            return math.hypot(s[0] - old_first[0], s[1] - old_first[1])
        elif pos == len(path_local):
            old_last = path_local[-1]
            return math.hypot(s[0] - old_last[0], s[1] - old_last[1])
        else:
            prev_p = path_local[pos - 1]
            next_p = path_local[pos]
            old_len = math.hypot(next_p[0] - prev_p[0], next_p[1] - prev_p[1])
            new_len = (
                math.hypot(s[0] - prev_p[0], s[1] - prev_p[1])
                + math.hypot(next_p[0] - s[0], next_p[1] - s[1])
            )
            return new_len - old_len

    for s in standby:
        best_pos = None
        best_extra = None
        if len(new_path) == 1:
            candidates = [0, 1]
        else:
            candidates = range(0, len(new_path) + 1)
        for pos in candidates:
            if pos == 0:
                p_prev = None
                p_next = new_path[0]
            elif pos == len(new_path):
                p_prev = new_path[-1]
                p_next = None
            else:
                p_prev = new_path[pos - 1]
                p_next = new_path[pos]
            if not can_insert_between(p_prev, s, p_next):
                continue
            extra = extra_length_for_insertion(new_path, s, pos)
            if best_extra is None or extra < best_extra:
                best_extra = extra
                best_pos = pos
        if best_pos is not None:
            new_path.insert(best_pos, s)
        else:
            still_leftover.append(s)
    return new_path, still_leftover


def chunk_path(points, chunk_size):
    if not points:
        return []
    if not chunk_size or chunk_size <= 0 or len(points) <= chunk_size:
        return [points]
    chunk_size = int(chunk_size)
    chunks = []
    n = len(points)
    idx = 0
    while idx < n:
        end = min(idx + chunk_size, n)
        if idx == 0:
            chunks.append(points[idx:end])
        else:
            chunks.append([points[idx - 1]] + points[idx:end])
        idx = end
    return chunks


def build_svg_for_paths(
    color_paths,
    image_size,
    scale=1.0,
    stroke_width=0.3,
    chunk_size=0,
    source_dpi=None,
):
    if not color_paths:
        return ""
    scale = float(scale) if scale else 1.0
    stroke_width = float(stroke_width) if stroke_width else 0.3
    chunk_size = int(chunk_size) if chunk_size else 0
    width = max(image_size[0] * scale, 1.0)
    height = max(image_size[1] * scale, 1.0)
    groups = []
    for color_hex, path in color_paths:
        if not path:
            continue
        chunked = chunk_path(path, chunk_size)
        path_elems = []
        for chunk in chunked:
            if not chunk:
                continue
            cmds = []
            first = True
            for (x, y) in chunk:
                sx = x * scale
                sy = y * scale
                if first:
                    cmds.append(f"M {sx:.2f} {sy:.2f}")
                    first = False
                else:
                    cmds.append(f"L {sx:.2f} {sy:.2f}")
            d_attr = " ".join(cmds)
            path_elems.append(
                f'    <path d="{d_attr}" fill="none" stroke="{color_hex}" stroke-width="{stroke_width}"/>'
            )
        if path_elems:
            groups.append(
                f'  <g data-color="{color_hex}">\n'
                + "\n".join(path_elems)
                + "\n  </g>"
            )
    if not groups:
        return ""
    width_attr = f"{width:.2f}px"
    height_attr = f"{height:.2f}px"
    dpi_x = None
    dpi_y = None
    if isinstance(source_dpi, (tuple, list)) and len(source_dpi) >= 2:
        dpi_x = float(source_dpi[0])
        dpi_y = float(source_dpi[1])
    elif isinstance(source_dpi, (int, float)) and source_dpi > 0:
        dpi_x = float(source_dpi)
        dpi_y = float(source_dpi)
    if dpi_x and dpi_y and dpi_x > 0 and dpi_y > 0:
        mm_per_inch = 25.4
        width_mm = (width / dpi_x) * mm_per_inch
        height_mm = (height / dpi_y) * mm_per_inch
        width_attr = f"{width_mm:.2f}mm"
        height_attr = f"{height_mm:.2f}mm"

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     width="{width_attr}" height="{height_attr}"\n'
        f'     viewBox="0 0 {width:.2f} {height:.2f}">\n'
        + "\n".join(groups)
        + "\n</svg>\n"
    )
    return svg


def process_color_points(
    points,
    opts,
    image_size,
    effective_dpi,
    color_hex,
    color_idx=0,
    seed_base=None,
    max_points_for_color=0,
):
    working = points[:]
    initial_points = len(working)
    emit_status(f"{color_hex}: punti iniziali={initial_points}")

    if not working:
        return {
            "color": color_hex,
            "path": [],
            "initial_points": 0,
            "final_points": 0,
            "discarded_points": 0,
        }

    dpi_for_mm = resolve_dpi_for_mm(
        effective_dpi, float(opts.get("default_dpi", 96.0) or 96.0)
    )

    min_dist_px = 0.0
    if opts["min_dist"] > 0:
        if opts["scale"] > 0:
            min_dist_px = opts["min_dist"] * (dpi_for_mm / 25.4) / float(opts["scale"])
        else:
            min_dist_px = opts["min_dist"] * (dpi_for_mm / 25.4)

    working = prepare_points_before_ordering(
        working,
        opts,
        image_size,
        effective_dpi,
        color_hex,
        color_idx=color_idx,
        seed_base=seed_base,
        max_points_for_color=max_points_for_color,
        report=True,
    )

    if not working:
        return {
            "color": color_hex,
            "path": [],
            "initial_points": initial_points,
            "final_points": 0,
            "discarded_points": 0,
        }

    if opts["ordering"] == "scanline":
        ordered = order_points_scanline(
            working,
            band_height=opts["scanline_band"],
            serpentine=opts["serpentine"],
        )
    else:
        ordered = order_points_nearest_neighbor(working)

    final_path = ordered
    discarded = []
    if opts["min_dist"] > 0:
        path_filtered, standby = filter_with_min_dist_and_standby(
            ordered, min_dist_px
        )
        current_path = path_filtered
        current_standby = standby
        for _ in range(max(0, opts["reinsertion_rounds"])):
            if not current_standby:
                break
            current_path, leftover = try_reinsert_points(
                current_path, current_standby, min_dist_px
            )
            current_standby = leftover
        final_path = current_path
        discarded = current_standby

    return {
        "color": color_hex,
        "path": final_path,
        "initial_points": initial_points,
        "final_points": len(final_path),
        "discarded_points": len(discarded),
        "allocated_max_points": int(max_points_for_color) if max_points_for_color else 0,
    }


def run_pipeline(image_bytes, opts):
    emit_progress(22, "Avvio pipeline")
    emit_status("=== Inizio nuova conversione ===")
    try:
        analysis_cell_mm = float(opts.get("analysis_cell_mm", 0.0) or 0.0)
    except (TypeError, ValueError):
        analysis_cell_mm = 0.0
    if 0.0 < analysis_cell_mm < 1.0:
        emit_status(
            f"analysis-cell impostato a {analysis_cell_mm:.2f} mm: applico minimo 1.00 mm."
        )
        analysis_cell_mm = 1.0
    opts["analysis_cell_mm"] = analysis_cell_mm

    max_width = opts["max_width"] if opts["max_width"] > 0 else None
    color_count = max(1, int(opts.get("color_count", 1)))
    sample_colors = parse_sample_colors(opts.get("sample_colors", ""))
    sample_tolerance = float(opts.get("sample_tolerance", 0.0) or 0.0)
    exclude_background = bool(opts.get("exclude_background", False))
    background_colors = parse_sample_colors(opts.get("background_colors", ""))
    background_tolerance = float(opts.get("background_tolerance", 0.0) or 0.0)
    color_points, size, source_dpi = load_points_grouped_from_bytes(
        image_bytes,
        max_width=max_width,
        threshold=opts["threshold"],
        color_count=color_count,
        sample_colors=sample_colors if sample_colors else None,
        sample_tolerance=sample_tolerance,
        exclude_background=exclude_background,
        background_colors=background_colors if background_colors else None,
        background_tolerance=background_tolerance,
    )
    if not color_points:
        raise ValueError(
            "Nessun pixel utile trovato con la soglia/colori correnti. Prova ad abbassare la soglia o aumentare i colori."
        )
    emit_progress(35, "Bitmap analizzata")

    emit_status(f"Colori trovati: {len(color_points)} (richiesti {color_count})")
    if sample_colors:
        printable = [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b) in sample_colors]
        emit_status(
            "Campionamento colori attivo: "
            f"{', '.join(printable)} con tolleranza {sample_tolerance:.2f}"
        )
    if exclude_background and background_colors:
        printable_bg = [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b) in background_colors]
        emit_status(
            "Esclusione sfondo attiva: "
            f"{', '.join(printable_bg)} con tolleranza {background_tolerance:.2f}"
        )
    base_seed = None
    if opts["style"] == "degrade":
        seed_raw = opts.get("degrade_seed")
        if seed_raw not in (None, ""):
            try:
                base_seed = int(seed_raw)
            except (TypeError, ValueError):
                base_seed = None

    budget_info = resolve_global_point_budget(
        color_points,
        opts.get("max_points", 0),
        opts.get("target_density", 0.0),
    )
    global_max_points = budget_info["budget"]
    per_color_max = allocate_max_points_by_color(color_points, global_max_points)
    if global_max_points > 0:
        total_allocated = sum(per_color_max.values())
        if budget_info["mode"] == "density":
            emit_status(
                f"Budget da densita={budget_info['density']:.2f}% -> {global_max_points} punti globali, distribuiti={total_allocated}."
            )
        elif budget_info["mode"] == "density+cap":
            emit_status(
                "Budget da densita "
                f"{budget_info['density']:.2f}% -> {budget_info['budget_from_density']} "
                f"con cap max-points={int(opts.get('max_points', 0) or 0)}: uso {global_max_points}, distribuiti={total_allocated}."
            )
        else:
            emit_status(
                f"Budget max-points globale={global_max_points}, distribuito={total_allocated} sui colori attivi."
            )
    emit_progress(42, "Budget punti definito")

    color_results = []
    ordered_colors = sorted(color_points.keys())
    color_count_total = len(ordered_colors)
    for idx, color_hex in enumerate(ordered_colors):
        result = process_color_points(
            color_points[color_hex],
            opts,
            size,
            source_dpi,
            color_hex,
            color_idx=idx,
            seed_base=base_seed,
            max_points_for_color=per_color_max.get(color_hex, 0),
        )
        color_results.append(result)
        progress_span = 45.0
        progress_base = 45.0
        emit_progress(
            progress_base + progress_span * ((idx + 1) / float(max(1, color_count_total))),
            f"Elaboro colori {idx + 1}/{color_count_total}",
        )

    final_paths = [
        (res["color"], res["path"]) for res in color_results if res["path"]
    ]
    if not final_paths:
        raise ValueError(
            "Nessun percorso finale disponibile; riduci il min-dist o modifica le impostazioni."
        )

    svg_str = build_svg_for_paths(
        final_paths,
        size,
        scale=opts["scale"],
        stroke_width=opts["stroke_width"],
        chunk_size=opts["chunk_size"],
        source_dpi=source_dpi,
    )
    emit_progress(93, "SVG combinato creato")
    emit_status(
        f"SVG combinato pronto: colori attivi={len(final_paths)}, chunk={opts['chunk_size']}"
    )

    mono_svg = ""
    if final_paths:
        mono_path = []
        for _, path in final_paths:
            mono_path.extend(path)
        if mono_path:
            mono_svg = build_svg_for_paths(
                [("#000000", mono_path)],
                size,
                scale=opts["scale"],
                stroke_width=opts["stroke_width"],
                chunk_size=opts["chunk_size"],
                source_dpi=source_dpi,
            )

    color_payload = []
    for res in color_results:
        single_svg = (
            build_svg_for_paths(
                [(res["color"], res["path"])],
                size,
                scale=opts["scale"],
                stroke_width=opts["stroke_width"],
                chunk_size=opts["chunk_size"],
                source_dpi=source_dpi,
            )
            if res["path"]
            else ""
        )
        color_payload.append(
            {
                "color": res["color"],
                "initial_points": res["initial_points"],
                "final_points": res["final_points"],
                "discarded_points": res["discarded_points"],
                "allocated_max_points": res.get("allocated_max_points", 0),
                "svg": single_svg,
            }
        )

    summary_lines = [
        f"Colori elaborati: {len(final_paths)} su {len(color_results)} totali"
    ]
    for res in color_results:
        density_ratio = (
            (100.0 * res["final_points"] / res["initial_points"])
            if res["initial_points"] > 0
            else 0.0
        )
        summary_lines.append(
            f"{res['color']} -> iniziali {res['initial_points']} | finali {res['final_points']} | scartati {res['discarded_points']} | densita {density_ratio:.1f}%"
        )
    summary = "\n".join(summary_lines)
    emit_progress(98, "Output pronto")
    return svg_str, summary, color_payload, mono_svg


def run_pipeline_browser(image_bytes, options):
    if not isinstance(image_bytes, (bytes, bytearray)):
        image_bytes = bytes(image_bytes)
    multi_svg, summary, colors, mono_svg = run_pipeline(image_bytes, options)
    emit_progress(100, "Conversione completata")
    return {
        "svg": multi_svg,
        "mono_svg": mono_svg,
        "summary": summary,
        "colors": colors,
    }


def analyze_preview_browser(image_bytes, options):
    return analyze_preview(image_bytes, options)
