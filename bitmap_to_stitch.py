#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image


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


# ------------------------------------------------------------
# 1. Lettura immagine e punti (pixel neri)
# ------------------------------------------------------------

def load_points_from_image(
    path, max_width=None, threshold=200, sample_colors=None, sample_tolerance=0.0
):
    """
    Carica l'immagine, la converte in scala di grigi (0-255),
    opzionalmente ridimensiona la larghezza a max_width (mantenendo le proporzioni),
    e ritorna:
      - lista di punti (x, y) per i pixel più scuri della soglia
      - size = (width_px, height_px)
    """
    original = Image.open(path)
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

    rgb = np.array(img.convert("RGB"))
    luminance = np.array(img.convert("L"))
    mask = luminance < int(threshold)
    if sample_colors:
        # I colori campionati si aggiungono alla selezione base, non la sostituiscono.
        mask |= build_color_match_mask(rgb, sample_colors, sample_tolerance)
    if "A" in img.getbands():
        alpha = np.array(img.getchannel("A"))
        mask &= alpha > 0

    ys, xs = np.where(mask)  # pixel selezionati
    points = list(zip(xs.tolist(), ys.tolist()))
    return points, img.size, effective_dpi  # (width_px, height_px), dpi


# ------------------------------------------------------------
# 2. Limite globale di sicurezza (opzionale)
# ------------------------------------------------------------

def subsample_points(points, max_points=None):
    """
    Riduce la lista di punti a max_points, distribuiti in modo uniforme sull'ordine attuale.
    Se max_points è None o non positivo o maggiore del numero di punti, non fa nulla.
    """
    n = len(points)
    if (max_points is None) or (max_points <= 0) or (max_points >= n):
        return points

    idx = np.linspace(0, n - 1, num=max_points, dtype=int)
    return [points[i] for i in idx]


# ------------------------------------------------------------
# 3. Ordinamento iniziale Nearest Neighbor (singolo percorso)
# ------------------------------------------------------------

def order_points_nearest_neighbor(points):
    """
    Ordina tutti i punti in un UNICO percorso usando Nearest Neighbor (O(n^2)).
    Non scarta nessun punto.
    """
    n = len(points)
    if n == 0:
        return []

    used = [False] * n
    ordered = []

    # punto iniziale: quello più a sinistra (min x)
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
    """
    Ordina i punti per "scanline": scorre le righe (o bande) dall'alto verso il basso
    e all'interno ordina per x. L'opzione serpentine fa sì che ogni banda alterni
    il verso (sinistra→destra, poi destra→sinistra) per ridurre i salti lunghi.
    """
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
    """
    Collassa i punti su una griglia: per ogni cella quadrata di lato cell_size
    posiziona un solo punto al centro della cella se contiene almeno un pixel nero.
    Garantisce quindi una disposizione più regolare nelle campiture uniformi.
    """
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
    points,
    drop_probability=0.0,
    jitter=0.0,
    image_size=None,
    seed=None,
):
    """
    Applica effetti "degradé":
      - drop_probability: probabilità (0-1) di scartare ciascun punto.
      - jitter: spostamento casuale max in px per x e y.
    """
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


# ------------------------------------------------------------
# 4. Utility per analisi percorso
# ------------------------------------------------------------

def segment_lengths(points):
    if len(points) < 2:
        return []
    lengths = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        lengths.append(math.hypot(x2 - x1, y2 - y1))
    return lengths


def max_segment_length(points):
    lens = segment_lengths(points)
    return max(lens) if lens else 0.0


# ------------------------------------------------------------
# 5. Fase 1: filtro con min-dist + standby
# ------------------------------------------------------------

def filter_with_min_dist_and_standby(points, min_dist_px):
    """
    Scorre il percorso e costruisce:
      - path_filtered = percorso che rispetta min_dist_px tra punti consecutivi
      - standby = lista di punti troppo vicini, messi in attesa (non scartati definitivamente)
    """
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


# ------------------------------------------------------------
# 6. Fase 2: tentativo di reinserimento dei punti in standby
# ------------------------------------------------------------

def try_reinsert_points(path, standby, min_dist_px):
    """
    Prova a reinserire i punti in standby dentro il percorso "path"
    rispettando il vincolo di distanza minima tra vicini.

    Strategia:
      - per ogni punto s in standby:
          - prova a inserirlo in tutte le posizioni possibili:
              - prima del primo
              - tra P[i] e P[i+1]
              - alla fine
          - posizione valida solo se tutte le distanze con i vicini >= min_dist_px
          - tra le posizioni valide, sceglie quella con extra-lunghezza minima
      - se nessuna posizione valida: il punto rimane scartato

    Ritorna:
      - nuovo percorso con i punti reinseriti dove possibile
      - lista dei punti ancora non inseriti (scartati definitivamente)
    """
    if not standby or not path:
        return path, standby

    min2 = min_dist_px * min_dist_px
    new_path = path[:]  # lavoriamo su una copia
    still_leftover = []

    def can_insert_between(p_prev, p, p_next):
        """Controlla se le distanze con p_prev e p_next rispettano min_dist."""
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
        """
        Calcola quanto aumenta la lunghezza del percorso se inseriamo s in 'pos'.
        pos = 0 -> prima del primo
        pos = len(path_local) -> alla fine
        altro -> tra path_local[pos-1] e path_local[pos]
        """
        if len(path_local) == 0:
            return 0.0

        if pos == 0:
            # nuovo primo punto: lunghezza aggiuntiva = dist(s, old_first)
            old_first = path_local[0]
            return math.hypot(s[0] - old_first[0], s[1] - old_first[1])
        elif pos == len(path_local):
            # inserito alla fine: extra = dist(last, s)
            old_last = path_local[-1]
            return math.hypot(s[0] - old_last[0], s[1] - old_last[1])
        else:
            prev_p = path_local[pos - 1]
            next_p = path_local[pos]
            old_len = math.hypot(next_p[0] - prev_p[0], next_p[1] - prev_p[1])
            new_len = (math.hypot(s[0] - prev_p[0], s[1] - prev_p[1]) +
                       math.hypot(next_p[0] - s[0], next_p[1] - s[1]))
            return new_len - old_len

    for s in standby:
        best_pos = None
        best_extra = None

        # caso speciale: path con un solo punto
        if len(new_path) == 1:
            # possiamo solo inserirlo prima o dopo
            # pos 0: s -> p0
            # pos 1: p0 -> s
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


# ------------------------------------------------------------
# 7. Costruzione SVG: UNA sola path
# ------------------------------------------------------------

def build_svg_single_path(
    points,
    image_size=None,
    scale=1.0,
    stroke_width=0.3,
    chunk_size=0,
    source_dpi=None,
):
    """
    Crea una stringa SVG con una o più <path>:
      M x0 y0 L x1 y1 L x2 y2 ...
    chunk_size>0 divide il percorso in blocchi consecutivi (per alleggerire Illustrator).
    """
    if not points:
        return ""

    scaled = [(x * scale, y * scale) for (x, y) in points]

    if image_size is not None:
        width = max(1.0, float(image_size[0]) * scale)
        height = max(1.0, float(image_size[1]) * scale)
    else:
        max_x = max(p[0] for p in scaled)
        max_y = max(p[1] for p in scaled)
        width = max(1.0, max_x + 1.0)
        height = max(1.0, max_y + 1.0)

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

    width_attr = f'{width:.2f}px'
    height_attr = f'{height:.2f}px'
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

    svg = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width_attr}" height="{height_attr}"
     viewBox="0 0 {width:.2f} {height:.2f}">
{chr(10).join(path_elems)}
</svg>
"""
    return svg


# ------------------------------------------------------------
# 8. main()
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bitmap B/N → punti → unica path SVG con min-dist rigido + reinserimento punti standby."
    )
    parser.add_argument("input", help="Percorso dell'immagine di input (bitmap B/N)")
    parser.add_argument("output", help="Percorso dell'SVG di output")

    parser.add_argument(
        "--max-width",
        type=int,
        default=0,
        help="Larghezza massima (in pixel) a cui ridimensionare l'immagine. 0 = nessun resize (default: 0).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="Soglia di luminosità (0–255) per considerare un pixel come 'nero' (default: 200).",
    )
    parser.add_argument(
        "--sample-color",
        action="append",
        default=[],
        help="Colore da campionare (es. #000000). Ripetibile per più colori.",
    )
    parser.add_argument(
        "--sample-tolerance",
        type=float,
        default=0.0,
        help="Tolleranza colore (distanza RGB euclidea). Usata con --sample-color.",
    )
    parser.add_argument(
        "--default-dpi",
        type=float,
        default=96.0,
        help="DPI usato per conversioni mm→px quando il file sorgente non contiene DPI.",
    )

    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Numero massimo di punti da usare (limite globale di sicurezza). 0 = nessun limite (default: 0).",
    )
    parser.add_argument(
        "--style",
        choices=["carpet", "degrade"],
        default="carpet",
        help="Seleziona la pipeline: 'carpet' = tratteggio uniforme, 'degrade' = punti casuali.",
    )
    parser.add_argument(
        "--ordering",
        choices=["nearest", "scanline"],
        default="nearest",
        help="Strategia per l'ordinamento iniziale dei punti (default: nearest).",
    )
    parser.add_argument(
        "--scanline-band-height",
        type=int,
        default=4,
        help="Altezza (in px) della banda usata nell'ordinamento scanline (default: 4).",
    )
    parser.add_argument(
        "--no-scanline-serpentine",
        action="store_true",
        help="Disattiva l'alternanza sinistra-destra nelle scanline.",
    )
    parser.add_argument(
        "--grid-cell-size",
        type=int,
        default=0,
        help="Se >0, collassa i punti in una griglia regolare di lato specificato (px).",
    )
    parser.add_argument(
        "--degrade-random-drop",
        type=float,
        default=0.0,
        help="Per style=degrade: probabilità (0-1) di scartare ciascun punto.",
    )
    parser.add_argument(
        "--degrade-jitter",
        type=float,
        default=0.0,
        help="Per style=degrade: jitter massimo (in px) applicato alle coordinate.",
    )
    parser.add_argument(
        "--degrade-seed",
        type=int,
        default=None,
        help="Per style=degrade: seed RNG per risultati replicabili.",
    )
    parser.add_argument(
        "--path-chunk-size",
        type=int,
        default=0,
        help="Numero massimo di punti per singola path SVG (0 = tutta una path).",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Fattore di scala per le coordinate SVG. Se 1.0 e 1 unità = 1 mm, le lunghezze sono in mm.",
    )
    parser.add_argument(
        "--stroke-width",
        type=float,
        default=0.3,
        help="Spessore della traccia nello SVG (default: 0.3).",
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=1.0,
        help="Distanza minima RIGIDA tra punti consecutivi (in mm reali).",
    )
    parser.add_argument(
        "--reinsertion-rounds",
        type=int,
        default=1,
        help="Numero di giri di reinserimento sui punti in standby (default: 1).",
    )

    args = parser.parse_args()

    sample_colors = []
    for color_text in args.sample_color:
        sample_colors.append(parse_hex_color(color_text))

    input_path = Path(args.input)
    output_path = Path(args.output)

    # 1) Lettura immagine e punti
    print(f"Carico immagine: {input_path}")
    points, size, source_dpi = load_points_from_image(
        input_path,
        max_width=args.max_width if args.max_width > 0 else None,
        threshold=args.threshold,
        sample_colors=sample_colors if sample_colors else None,
        sample_tolerance=args.sample_tolerance,
    )
    print(f"Dimensione immagine (dopo eventuale resize): {size[0]}x{size[1]} px")
    if source_dpi is not None:
        print(f"DPI sorgente rilevato: {source_dpi}")
    else:
        print(
            f"DPI sorgente non presente: uso default-dpi={args.default_dpi:.2f} per conversioni mm."
        )
    if sample_colors:
        printable = [f"#{r:02X}{g:02X}{b:02X}" for (r, g, b) in sample_colors]
        print(
            "Campionamento colore attivo: "
            f"{', '.join(printable)} con tolleranza {args.sample_tolerance:.2f}"
        )
    print(f"Punti trovati (pixel neri): {len(points)}")

    if not points:
        print("Nessun punto nero trovato. Esco.")
        return

    if args.style == "degrade":
        before = len(points)
        points = apply_random_degrade_effects(
            points,
            drop_probability=args.degrade_random_drop,
            jitter=args.degrade_jitter,
            image_size=size,
            seed=args.degrade_seed,
        )
        print(
            "Punti dopo effetti degrade/random: "
            f"{len(points)} (prima: {before})"
        )
    elif args.grid_cell_size > 1:
        before = len(points)
        points = regularize_points_on_grid(points, args.grid_cell_size)
        print(
            "Punti dopo regolarizzazione a griglia "
            f"(cell={args.grid_cell_size}px): {len(points)} (prima: {before})"
        )

    # 2) Limite globale (opzionale)
    if args.max_points > 0:
        before = len(points)
        points = subsample_points(points, max_points=args.max_points)
        print(f"Punti dopo max-points: {len(points)} (prima: {before})")

    # 3) Ordinamento iniziale Nearest Neighbor (un solo percorso, nessun punto scartato)
    if args.ordering == "nearest":
        print("Ordino i punti con Nearest Neighbor (percorso iniziale)...")
        ordered = order_points_nearest_neighbor(points)
    else:
        print(
            "Ordino i punti con modalità scanline serpentina "
            f"(band-height={args.scanline_band_height}px)..."
        )
        ordered = order_points_scanline(
            points,
            band_height=args.scanline_band_height,
            serpentine=not args.no_scanline_serpentine,
        )
    print(f"Punti nel percorso iniziale: {len(ordered)}")
    print(f"Segmento massimo iniziale: {max_segment_length(ordered):.2f} px")

    # 4) Applicazione min-dist con standby + reinserimento
    if args.min_dist > 0:
        if source_dpi is not None:
            dpi_x = float(source_dpi[0]) if isinstance(source_dpi, (tuple, list)) else float(source_dpi)
            dpi_y = float(source_dpi[1]) if isinstance(source_dpi, (tuple, list)) else float(source_dpi)
        else:
            dpi_x = float(args.default_dpi)
            dpi_y = float(args.default_dpi)
        dpi_avg = (dpi_x + dpi_y) / 2.0
        if args.scale > 0:
            min_dist_px = args.min_dist * (dpi_avg / 25.4) / args.scale
        else:
            min_dist_px = args.min_dist * (dpi_avg / 25.4)
        print(
            f"Applico min-dist rigido: {args.min_dist:.2f} mm -> {min_dist_px:.2f} px "
            f"(dpi medio={dpi_avg:.2f}, scale={args.scale:.4f})"
        )

        # Primo passaggio: filtro + standby
        path_filtered, standby = filter_with_min_dist_and_standby(ordered, min_dist_px)
        print(f"Punti dopo filtro min-dist: {len(path_filtered)}")
        print(f"Punti in standby: {len(standby)}")

        # Tentativi di reinserimento
        current_path = path_filtered
        current_standby = standby

        for r in range(args.reinsertion_rounds):
            if not current_standby:
                break
            print(f"Reinserimento round {r+1}...")
            current_path, leftover = try_reinsert_points(current_path, current_standby, min_dist_px)
            print(f"Punti dopo reinserimento round {r+1}: {len(current_path)}")
            print(f"Punti ancora scartati: {len(leftover)}")
            current_standby = leftover

        final_path = current_path
        discarded = current_standby
        print(f"Punti finali nel percorso: {len(final_path)}")
        print(f"Punti scartati definitivamente: {len(discarded)}")
    else:
        print("min-dist <= 0: nessun vincolo di distanza minima.")
        final_path = ordered

    # 5) Costruzione SVG (una sola path)
    print("Costruisco lo SVG (una sola path)...")
    svg_str = build_svg_single_path(
        final_path,
        image_size=size,
        scale=args.scale,
        stroke_width=args.stroke_width,
        chunk_size=args.path_chunk_size,
        source_dpi=source_dpi,
    )

    output_path.write_text(svg_str, encoding="utf-8")
    print(f"SVG salvato in: {output_path}")


if __name__ == "__main__":
    main()
