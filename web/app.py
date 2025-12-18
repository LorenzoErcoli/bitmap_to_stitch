import io
import math

import numpy as np
from PIL import Image


def load_points_from_bytes(data, max_width=None, threshold=200):
    img = Image.open(io.BytesIO(data)).convert("L")

    if max_width is not None and max_width > 0:
        w, h = img.size
        if w > max_width:
            scale = max_width / float(w)
            new_w = max_width
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = np.array(img)
    ys, xs = np.where(arr < threshold)
    points = list(zip(xs.tolist(), ys.tolist()))
    return points, img.size


def subsample_points(points, max_points=None):
    n = len(points)
    if (max_points is None) or (max_points <= 0) or (max_points >= n):
        return points
    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [points[i] for i in idx]


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
    cs = int(cell_size)
    if cs <= 1 or not points:
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


def build_svg_single_path(points, scale=1.0, stroke_width=0.3, chunk_size=0):
    if not points:
        return ""
    scaled = [(x * scale, y * scale) for (x, y) in points]
    max_x = max(p[0] for p in scaled)
    max_y = max(p[1] for p in scaled)
    width = max_x + 10
    height = max_y + 10

    def chunk_iter(data, size):
        if size <= 0 or len(data) <= size:
            yield data
            return
        n = len(data)
        idx = 0
        size = int(size)
        while idx < n:
            end = min(idx + size, n)
            if idx == 0:
                yield data[idx:end]
            else:
                yield [data[idx - 1]] + data[idx:end]
            idx += size

    path_elems = []
    for chunk in chunk_iter(scaled, int(chunk_size)):
        if not chunk:
            continue
        cmds = []
        first = True
        for (x, y) in chunk:
            if first:
                cmds.append(f"M {x:.2f} {y:.2f}")
                first = False
            else:
                cmds.append(f"L {x:.2f} {y:.2f}")
        d_attr = " ".join(cmds)
        path_elems.append(
            f'  <path d="{d_attr}" fill="none" stroke="black" stroke-width="{stroke_width}"/>'
        )

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.2f}" height="{height:.2f}"
     viewBox="0 0 {width:.2f} {height:.2f}">
{chr(10).join(path_elems)}
</svg>
"""
    return svg


def run_pipeline(image_bytes, opts):
    max_width = opts["max_width"] if opts["max_width"] > 0 else None
    points, size = load_points_from_bytes(
        image_bytes, max_width=max_width, threshold=opts["threshold"]
    )
    if not points:
        raise ValueError("Nessun pixel nero trovato con la soglia corrente.")

    if opts["style"] == "degrade":
        seed = int(opts["degrade_seed"]) if opts["degrade_seed"] else None
        points = apply_random_degrade_effects(
            points,
            drop_probability=opts["degrade_drop"],
            jitter=opts["degrade_jitter"],
            image_size=size,
            seed=seed,
        )
    elif opts["grid_cell_size"] > 1:
        points = regularize_points_on_grid(points, opts["grid_cell_size"])

    if opts["max_points"] > 0:
        points = subsample_points(points, max_points=opts["max_points"])

    if not points:
        raise ValueError("Tutti i punti sono stati filtrati dai parametri attuali.")

    if opts["ordering"] == "scanline":
        ordered = order_points_scanline(
            points,
            band_height=opts["scanline_band"],
            serpentine=opts["serpentine"],
        )
    else:
        ordered = order_points_nearest_neighbor(points)

    final_path = ordered
    discarded = []
    if opts["min_dist"] > 0 and opts["scale"] > 0:
        min_dist_px = opts["min_dist"] / opts["scale"]
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

    if not final_path:
        raise ValueError("Il percorso finale è vuoto; riduci il min-dist o cambia stile.")

    svg_str = build_svg_single_path(
        final_path,
        scale=opts["scale"],
        stroke_width=opts["stroke_width"],
        chunk_size=opts["chunk_size"],
    )
    summary = (
        f"Punti iniziali: {len(points)} | "
        f"Punti finali: {len(final_path)} | "
        f"Scartati: {len(discarded)}"
    )
    return svg_str, summary


def run_pipeline_browser(image_bytes, options):
    if not isinstance(image_bytes, (bytes, bytearray)):
        image_bytes = bytes(image_bytes)
    svg_str, summary = run_pipeline(image_bytes, options)
    return {"svg": svg_str, "summary": summary}
    download.download = (
        f"{upload.name.rsplit('.', 1)[0]}-stitch.svg" if upload.name else "stitch.svg"
    )
    download.style.display = "inline-block"

    status.innerText = summary
