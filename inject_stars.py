import os
import argparse
import csv
import numpy as np
import tifffile
from pathlib import Path
from skimage.filters import gaussian

def load_tiff(path):
    return tifffile.imread(str(path))

def save_tiff(path, image):
    # Ripristiniamo un formato sicuro per il salvataggio in virgola mobile
    tifffile.imwrite(str(path), image.astype(np.float32), photometric='rgb')

def augment_stamp(rgb, alpha):
    """Applica rotazioni e flip casuali per aumentare la varianza delle stelle."""
    k = np.random.randint(4)
    rgb = np.rot90(rgb, k=k, axes=(0, 1))
    alpha = np.rot90(alpha, k=k, axes=(0, 1))
    
    if np.random.rand() > 0.5:
        rgb = np.fliplr(rgb)
        alpha = np.fliplr(alpha)
        
    if np.random.rand() > 0.5:
        rgb = np.flipud(rgb)
        alpha = np.flipud(alpha)
        
    return rgb, alpha

def paste_star(canvas, rgb, alpha, cy, cx, brightness_jitter):
    """Incolla il ritaglio della stella sul canvas gestendo i bordi e il blending."""
    h, w = alpha.shape
    H, W = canvas.shape[:2]
    
    y0, y1 = cy - h // 2, cy + (h - h // 2)
    x0, x1 = cx - w // 2, cx + (w - w // 2)
    
    canvas_y0, canvas_y1 = max(0, y0), min(H, y1)
    canvas_x0, canvas_x1 = max(0, x0), min(W, x1)
    
    if canvas_y0 >= canvas_y1 or canvas_x0 >= canvas_x1:
        return 
        
    stamp_y0 = canvas_y0 - y0
    stamp_y1 = h - (y1 - canvas_y1)
    stamp_x0 = canvas_x0 - x0
    stamp_x1 = w - (x1 - canvas_x1)
    
    stamp_rgb = rgb[stamp_y0:stamp_y1, stamp_x0:stamp_x1]
    stamp_alpha = alpha[stamp_y0:stamp_y1, stamp_x0:stamp_x1]
    
    jitter = np.random.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter)
    stamp_rgb = np.clip(stamp_rgb * jitter, 0.0, 1.0)
    
    canvas_slice = canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1]
    alpha_3d = stamp_alpha[..., np.newaxis]
    
    # Alpha Blending classico
    canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = canvas_slice * (1 - alpha_3d) + stamp_rgb * alpha_3d

def synthesize_field(base, star_map, n_stars, star_pool, min_sep_factor, max_attempts, brightness_jitter):
    """Calcola densità e posiziona le stelle usando pre-campionamento basato sulla mappa sfocata."""
    H, W = base.shape[:2]
    canvas = base.copy()

    if n_stars <= 0:
        print("  -> Numero stelle richiesto = 0. Salto il posizionamento.")
        return canvas, 0

    print(f"  -> Tento di piazzare {n_stars} stelle da libreria...")

    # 1. Scala di grigi e preparazione formato
    if star_map.ndim == 3:
        gray_map = np.mean(star_map, axis=-1).astype(np.float32)
    else:
        gray_map = star_map.astype(np.float32)

    # 2. Mappa di probabilità tramite blur
    blurred_map = gaussian(gray_map, sigma=8.0)
    
    valid_ys, valid_xs = np.where(blurred_map > 1e-4)
    weights = blurred_map[valid_ys, valid_xs].astype(np.float64)
    
    if len(weights) == 0:
        print("  -> Mappa di probabilità piatta o vuota. Salto.")
        return canvas, 0

    # Normalizzazione per np.random.choice
    probs = weights / weights.sum()

    n_candidates = int(n_stars * 1.5) + 100
    candidate_indices = np.random.choice(len(valid_ys), size=n_candidates, p=probs)
    candidate_pointer = 0

    # Griglia Spaziale (ottimizzazione collisioni)
    max_radius = max(max(s[2], s[3]) / 2.0 for s in star_pool)
    cell_size = max(16, int(max_radius * 2) + 1)
    spatial_grid = {}
    n_placed = 0

    for _ in range(n_stars):
        rgb, alpha, h, w = star_pool[np.random.randint(len(star_pool))]
        rgb, alpha = augment_stamp(rgb, alpha)
        radius = max(h, w) / 2.0

        for _attempt in range(max_attempts):
            # Ricarica candidati se quelli precalcolati finiscono (troppi fallimenti)
            if candidate_pointer >= len(candidate_indices):
                candidate_indices = np.random.choice(len(valid_ys), size=n_candidates, p=probs)
                candidate_pointer = 0

            idx = candidate_indices[candidate_pointer]
            candidate_pointer += 1
            
            cy, cx = int(valid_ys[idx]), int(valid_xs[idx])

            cell_y = cy // cell_size
            cell_x = cx // cell_size

            # Check collisioni rapido
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    neighbors.extend(spatial_grid.get((cell_y + dy, cell_x + dx), []))

            ok = True
            if neighbors:
                narr = np.array(neighbors, dtype=np.float32)
                dists_sq = (cy - narr[:, 0]) ** 2 + (cx - narr[:, 1]) ** 2
                min_dists_sq = (min_sep_factor * (radius + narr[:, 2])) ** 2
                if np.any(dists_sq < min_dists_sq):
                    ok = False

            if ok:
                paste_star(canvas, rgb, alpha, cy, cx, brightness_jitter)
                spatial_grid.setdefault((cell_y, cell_x), []).append([cy, cx, radius])
                n_placed += 1
                break

    return canvas, n_placed

def main():
    parser = argparse.ArgumentParser(description="Inietta stelle usando le mappe TIFF originali come guida di probabilità e CSV per il conteggio.")
    parser.add_argument("--base-dir", default="assets/outputs-starless", help="Dir delle immagini base starless")
    parser.add_argument("--starmask-dir", default="assets/outputs-starmask", help="Dir delle starmask TIFF di riferimento")
    parser.add_argument("--stamps-dir", default="assets/star_library/stamps", help="Dir contenente i file npz del dataset stellare")
    parser.add_argument("--counts-csv", default="assets/star_library/star_counts.csv", help="CSV con i conteggi originali")
    parser.add_argument("--output-dir", default="assets/outputs-injected", help="Dir di salvataggio")
    parser.add_argument("--pool-size", type=int, default=5000, help="Numero di file npz da precaricare in RAM")
    parser.add_argument("--density-scale", type=float, default=1.0, help="Moltiplicatore stelle rispetto alla foto originale (1.0 = esatto)")
    parser.add_argument("--min-sep-factor", type=float, default=0.5, help="Distanza minima di separazione (0 = sovrapponibili)")
    parser.add_argument("--max-attempts", type=int, default=15, help="Tentativi limite per stamp")
    parser.add_argument("--brightness-jitter", type=float, default=0.2, help="Jitter per variare la luminosità +/-")
    parser.add_argument("--save-tiff", action="store_true", help="Salva anche una copia in formato TIFF")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 1. Caricamento del conteggio stelle da CSV
    star_counts = {}
    if Path(args.counts_csv).exists():
        with open(args.counts_csv, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Salta l'header
            for row in reader:
                if len(row) >= 2:
                    star_counts[row[0]] = int(row[1])
        print(f"--> Caricati i conteggi per {len(star_counts)} immagini dal CSV.")
    else:
        print(f"ERRORE: File CSV {args.counts_csv} non trovato.")
        return

    # 2. Setup RAM Pool
    stamp_files = list(Path(args.stamps_dir).glob("*.npz"))
    if not stamp_files:
        print(f"ERRORE: Nessun file .npz trovato nella cartella '{args.stamps_dir}'")
        return

    print(f"Libreria totale rilevata: {len(stamp_files)} stelle.")
    pool_size = min(args.pool_size, len(stamp_files))
    print(f"--> Caricamento {pool_size} stelle in RAM...")

    chosen_stamps = np.random.choice(stamp_files, size=pool_size, replace=False)
    star_pool = []
    for p in chosen_stamps:
        with np.load(p) as data:
            rgb = data['rgb'].astype(np.float32)
            alpha = data['alpha'].astype(np.float32)
            h, w = rgb.shape[:2]
            star_pool.append((rgb, alpha, h, w))
    
    print("--> Pool caricato con successo.\n")

    # 3. Esecuzione su file TIFF
    base_files = list(Path(args.base_dir).glob("*.tif")) + list(Path(args.base_dir).glob("*.tiff"))
    
    for base_path in base_files:
        stem = base_path.stem
        print(f"Elaborazione: {base_path.name}")
        
        if stem not in star_counts:
            print(f"  [!] ERRORE: Nessun conteggio trovato per {stem} in {args.counts_csv}. Salto immagine.")
            continue

        base = load_tiff(base_path)
        
        starmask_path = Path(args.starmask_dir) / base_path.name
        if not starmask_path.exists():
            print(f"  [!] ERRORE: Non ho trovato {starmask_path.name} in {args.starmask_dir}. Salto immagine.")
            continue
            
        star_map = load_tiff(starmask_path)
        
        # Normalizzazione automatica [0, 1] per evitare problemi di blending
        if base.max() > 2.0:
            base = base / (65535.0 if base.dtype == np.uint16 else 255.0)
        if star_map.max() > 2.0:
            star_map = star_map / (65535.0 if star_map.dtype == np.uint16 else 255.0)

        # Calcolo del numero di stelle effettivo basato sul CSV e sul density_scale
        target_stars = int(star_counts[stem] * args.density_scale)

        canvas, n_placed = synthesize_field(
            base, star_map, target_stars, star_pool,
            args.min_sep_factor, args.max_attempts, args.brightness_jitter
        )
        
        print(f"  -> {n_placed}/{target_stars} stelle iniettate con successo.")
        
        # Salvataggio di default in .npy
        out_path_npy = Path(args.output_dir) / f"{stem}.npy"
        np.save(out_path_npy, canvas.astype(np.float32))
        print(f"  -> Salvato: {out_path_npy}")

        # Salvataggio opzionale in .tiff
        if args.save_tiff:
            out_path_tiff = Path(args.output_dir) / "tiff" / base_path.name
            save_tiff(out_path_tiff, canvas)
            print(f"  -> Salvato (copia TIFF): {out_path_tiff}")
            
        print() # Riga vuota per pulizia log

if __name__ == "__main__":
    main()
