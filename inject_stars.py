import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def load_starless_tiff(tiff_path):
    """Carica un TIFF starless prodotto da StarNet2 (stessa convenzione di make_dataset.py)."""
    arr = tifffile.imread(tiff_path)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Dtype inatteso per TIFF starless: {arr.dtype}")


def load_library(lib_dir):
    lib_dir = Path(lib_dir)
    with open(lib_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(lib_dir / "field_density.json") as f:
        densities = json.load(f)
    if not metadata:
        raise RuntimeError(f"Libreria vuota in {lib_dir}. Lancia prima build_star_library.py")
    return lib_dir / "stamps", metadata, np.array(densities)


def load_stamp(stamps_dir, meta):
    data = np.load(stamps_dir / meta["file"])
    return data["rgb"], data["alpha"]


def augment_stamp(rgb, alpha):
    """Stessa logica di apply_augmentation in make_dataset.py, applicata allo stamp."""
    if np.random.rand() < 0.5:
        rgb, alpha = np.fliplr(rgb), np.fliplr(alpha)
    if np.random.rand() < 0.5:
        rgb, alpha = np.flipud(rgb), np.flipud(alpha)
    k = int(np.random.randint(4))
    if k:
        rgb, alpha = np.rot90(rgb, k=k), np.rot90(alpha, k=k)
    return np.ascontiguousarray(rgb), np.ascontiguousarray(alpha)


def paste_star(canvas, rgb, alpha, cy, cx, brightness_jitter):
    """Blending ADDITIVO (screen-like): la luce della stella si somma a quella
    di sfondo gia' presente, non la sostituisce.
    """
    h, w = alpha.shape
    y0, x0 = cy - h // 2, cx - w // 2
    y1, x1 = y0 + h, x0 + w

    H, W = canvas.shape[:2]
    cy0, cx0 = max(y0, 0), max(x0, 0)
    cy1, cx1 = min(y1, H), min(x1, W)
    if cy0 >= cy1 or cx0 >= cx1:
        return
    sy0, sx0 = cy0 - y0, cx0 - x0
    sy1, sx1 = sy0 + (cy1 - cy0), sx0 + (cx1 - cx0)

    scale = 1.0 + np.random.uniform(-brightness_jitter, brightness_jitter)
    a = alpha[sy0:sy1, sx0:sx1][..., None] * scale
    region = canvas[cy0:cy1, cx0:cx1, :]
    canvas[cy0:cy1, cx0:cx1, :] = np.clip(region + rgb[sy0:sy1, sx0:sx1, :] * a, 0.0, 1.0)


def synthesize_field(base, pixel_mask, star_pool, density, min_sep_factor, max_attempts, brightness_jitter):
    H, W = base.shape[:2]
    canvas = base.copy()

    if pixel_mask is not None and pixel_mask.any():
        valid_ys, valid_xs = np.where(pixel_mask)
        valid_area = len(valid_ys)
    else:
        valid_area = H * W
        yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
        valid_ys, valid_xs = yy.ravel(), xx.ravel()

    if valid_area == 0:
        return canvas, 0

    n_stars = int(np.random.poisson(density * valid_area))
    print(f"  -> Tentativo di posizionamento di {n_stars} stelle...")

    # --- INIZIALIZZAZIONE GRIGLIA SPAZIALE ---
    # Trova il raggio massimo assoluto nel pool per stabilire la dimensione di sicurezza delle celle
    max_radius = max(max(s[2], s[3]) / 2.0 for s in star_pool)
    cell_size = max(16, int(max_radius * 2) + 1)
    
    # Dizionario che mappa (cell_y, cell_x) -> lista di [cy, cx, radius] delle stelle piazzate
    spatial_grid = {}
    n_placed = 0

    for _ in range(n_stars):
        rgb, alpha, h, w = star_pool[np.random.randint(len(star_pool))]
        rgb, alpha = augment_stamp(rgb, alpha)
        radius = max(h, w) / 2.0

        for _attempt in range(max_attempts):
            k = np.random.randint(len(valid_ys))
            cy, cx = int(valid_ys[k]), int(valid_xs[k])

            cell_y = cy // cell_size
            cell_x = cx // cell_size

            # Recupera solo le stelle presenti nella cella corrente e nelle 8 vicine
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    neighbors.extend(spatial_grid.get((cell_y + dy, cell_x + dx), []))

            ok = True
            if neighbors:
                # Confronto locale vettorizzato fulmineo (frazioni di microsecondo)
                narr = np.array(neighbors, dtype=np.float32)
                dists_sq = (cy - narr[:, 0]) ** 2 + (cx - narr[:, 1]) ** 2
                min_dists_sq = (min_sep_factor * (radius + narr[:, 2])) ** 2
                
                if np.any(dists_sq < min_dists_sq):
                    ok = False

            if ok:
                paste_star(canvas, rgb, alpha, cy, cx, brightness_jitter)
                # Registra la stella nella griglia spaziale
                spatial_grid.setdefault((cell_y, cell_x), []).append([cy, cx, radius])
                n_placed += 1
                break

    return canvas, n_placed


def main():
    parser = argparse.ArgumentParser(
        description="Sintetizza campi stellari massivi sopra le immagini starless usando un algoritmo a griglia spaziale."
    )
    parser.add_argument("--starless-dir", default="assets/outputs-starless")
    parser.add_argument("--mask-dir", default="assets/outputs-mask")
    parser.add_argument("--library-dir", default="assets/star_library")
    parser.add_argument("--out-dir", default="assets/outputs-npy-synthetic")
    parser.add_argument("--min-sep-factor", type=float, default=0.55,
                        help="Distanza minima fra due stelle come frazione della somma dei raggi")
    parser.add_argument("--max-attempts", type=int, default=30,
                        help="Tentativi di posizionamento prima di scartare una stella")
    parser.add_argument("--density-scale", type=float, default=1.0,
                        help="Fattore moltiplicativo sulla densita' stellare empirica")
    parser.add_argument("--brightness-jitter", type=float, default=0.15,
                        help="Variazione casuale di luminosita' per stamp")
    parser.add_argument("--pool-size", type=int, default=5000,
                        help="Numero di stelle da pre-caricare in RAM dal dataset complessivo")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    np.random.seed(args.seed)

    starless_dir = Path(args.starless_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamps_dir, metadata, densities = load_library(args.library_dir)
    print(f"Libreria totale: {len(metadata)} stelle disponibili.")

    # Cache speculativa in RAM per azzerare l'I/O da disco
    pool_size = min(args.pool_size, len(metadata))
    print(f"--> Caricamento di {pool_size} stelle in RAM per performance massime...")
    
    chosen_indices = np.random.choice(len(metadata), pool_size, replace=False)
    star_pool = []
    for idx in chosen_indices:
        meta = metadata[idx]
        rgb, alpha = load_stamp(stamps_dir, meta)
        star_pool.append((rgb, alpha, meta["h"], meta["w"]))
    
    print(f"--> RAM Cache completata. Densita' media: {densities.mean():.6f} px^-2")

    tif_files = sorted(starless_dir.glob("*.tif"))
    if not tif_files:
        print(f"Nessun starless trovato in {starless_dir}")
        return

    for tif_path in tif_files:
        stem = tif_path.stem
        base = load_starless_tiff(tif_path)

        if base.ndim == 2:
            base = np.stack([base, base, base], axis=-1)
        elif base.ndim == 3 and base.shape[-1] == 4:
            base = base[:, :, :3]

        mask_path = mask_dir / f"{stem}.npy"
        pixel_mask = np.load(mask_path) if mask_path.exists() else None

        density = float(np.random.choice(densities)) * args.density_scale

        canvas, n_placed = synthesize_field(
            base, pixel_mask, star_pool, density,
            args.min_sep_factor, args.max_attempts, args.brightness_jitter,
        )
        np.save(out_dir / f"{stem}.npy", canvas.astype(np.float32))
        print(f"{stem}: {n_placed} stelle reali sintetizzate su {base.shape[1]}x{base.shape[0]} px.")

    print(f"\nCompletato con successo. Output in {out_dir}")


if __name__ == "__main__":
    main()
