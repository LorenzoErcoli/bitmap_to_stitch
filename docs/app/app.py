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
    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A"))
        mask &= alpha > 0

    if not np.any(mask):
        emit_status(
            f"Nessun pixel valido trovato con soglia={threshold}; aumenta la soglia o usa un'immagine diversa."
        )
        return {}, img.size, effective_dpi

    rgb_img = img.convert("RGB")
    palette_colors = max(1, int(color_count))
    quantized = rgb_img.convert(
        "P", palette=Image.ADAPTIVE, colors=palette_colors
    )
    palette = quantized.getpalette()
    arr = np.array(quantized)

    ys, xs = np.where(mask)
    indexes = arr[ys, xs]
    unique_idxs = np.unique(indexes)

    def idx_to_hex(idx):
        base = idx * 3
        r = palette[base]
        g = palette[base + 1]
        b = palette[base + 2]
        return f"#{r:02X}{g:02X}{b:02X}"

    color_points = {}
    for idx in unique_idxs.tolist():
        color_hex = idx_to_hex(int(idx))
        color_mask = indexes == idx
        px = xs[color_mask]
        py = ys[color_mask]
        color_points[color_hex] = list(zip(px.tolist(), py.tolist()))

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

    analysis_cell_mm = float(opts.get("analysis_cell_mm", 0.0) or 0.0)
    dpi_for_mm = float(opts.get("default_dpi", 96.0) or 96.0)
    if isinstance(effective_dpi, (tuple, list)) and len(effective_dpi) >= 2:
        dpi_for_mm = (float(effective_dpi[0]) + float(effective_dpi[1])) / 2.0
    elif isinstance(effective_dpi, (int, float)) and effective_dpi > 0:
        dpi_for_mm = float(effective_dpi)

    if analysis_cell_mm >= 1.0 and opts["scale"] > 0:
        cell_px = analysis_cell_mm * (dpi_for_mm / 25.4) / float(opts["scale"])
        if cell_px > 1.0:
            before = len(working)
            working = regularize_points_on_grid(working, cell_px)
            emit_status(
                f"{color_hex}: analisi griglia metrica {analysis_cell_mm:.2f} mm ({cell_px:.2f} px) -> {len(working)} punti (prima {before})"
            )
        else:
            emit_status(
                f"{color_hex}: analysis-cell {analysis_cell_mm:.2f} mm troppo fine rispetto alla scala corrente ({opts['scale']:.4f}); nessuna regolarizzazione metrica."
            )

    if opts["style"] == "degrade":
        seed = (seed_base + color_idx) if seed_base is not None else None
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
        emit_status(f"{color_hex}: regolarizzo griglia (cell={opts['grid_cell_size']} px)")
        working = regularize_points_on_grid(working, opts["grid_cell_size"])

    max_points = int(max_points_for_color) if max_points_for_color else 0
    if max_points > 0 and len(working) > max_points:
        before = len(working)
        working = subsample_points(working, max_points=max_points)
        emit_status(
            f"{color_hex}: limito max-points a {len(working)} (prima {before})"
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
        if opts["scale"] > 0:
            min_dist_px = opts["min_dist"] * (dpi_for_mm / 25.4) / float(opts["scale"])
        else:
            min_dist_px = opts["min_dist"] * (dpi_for_mm / 25.4)
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
        "allocated_max_points": max_points,
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
    color_points, size, source_dpi = load_points_grouped_from_bytes(
        image_bytes,
        max_width=max_width,
        threshold=opts["threshold"],
        color_count=color_count,
        sample_colors=sample_colors if sample_colors else None,
        sample_tolerance=sample_tolerance,
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
