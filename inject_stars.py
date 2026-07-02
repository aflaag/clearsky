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

def paste_star(canvas, rgb, alpha, cy, cx, intensity_multiplier):
    """Incolla il ritaglio della stella sul canvas usando un modello additivo fisicamente accurato."""
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
    
    canvas_slice = canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1]
    alpha_3d = stamp_alpha[..., np.newaxis]
    
    # MODELLO ADDITIVO: In astrofotografia i fotoni si sommano. Nessuna sostituzione.
    # Scaliamo la stella in base alla luminosità stimata dalla mask originale
    added_light = stamp_rgb * alpha_3d * intensity_multiplier
    
    canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = np.clip(canvas_slice + added_light, 0.0, 1.0)

def synthesize_field(base, star_map, gray_map, n_stars, star_pool, min_sep_factor, max_attempts, brightness_jitter):
    """Calcola densità e posiziona le stelle stimando la luminosità dalla mask."""
    H, W = base.shape[:2]
    canvas = base.copy()

    if n_stars <= 0:
        print("  -> Numero stelle richiesto = 0. Salto il posizionamento.")
        return canvas, 0

    print(f"  -> Tento di piazzare {n_stars} stelle da libreria...")

    # 1. Mappa di probabilità tramite blur (sigma ridotto per mantenere i picchi puntiformi)
    blurred_map = gaussian(gray_map, sigma=8.0)
    
    valid_ys, valid_xs = np.where(blurred_map > 1e-4)
    weights = blurred_map[valid_ys, valid_xs].astype(np.float64)
    
    if len(weights) == 0:
        print("  -> Mappa di probabilità piatta o vuota. Salto.")
        return canvas, 0

    probs = weights / weights.sum()

    n_candidates = int(n_stars * 1.5) + 100
    
    # --- MODIFICA 3: Generazione e ordinamento dei candidati per peso (crescente) ---
    def get_sorted_candidates():
        indices = np.random.choice(len(valid_ys), size=n_candidates, p=probs)
        # Ordina gli indici scelti in base al peso (dal più debole al più forte)
        chosen_weights = weights[indices]
        return indices[np.argsort(chosen_weights)]

    candidate_indices = get_sorted_candidates()
    candidate_pointer = 0

    max_radius = max(s[2] for s in star_pool) 
    cell_size = max(16, int(max_radius * 2) + 1)
    spatial_grid = {}
    n_placed = 0

    pool_max_idx = len(star_pool) - 1
    
    # --- MODIFICA 2: Array per la ricerca binaria della luminosità intrinseca ---
    intrinsic_lums = np.array([s[4] for s in star_pool])

    for _ in range(n_stars):
        for _attempt in range(max_attempts):
            if candidate_pointer >= len(candidate_indices):
                candidate_indices = get_sorted_candidates()
                candidate_pointer = 0

            idx = candidate_indices[candidate_pointer]
            candidate_pointer += 1
            
            cy, cx = int(valid_ys[idx]), int(valid_xs[idx])

            # Lettura del valore grezzo dalla mask
            y_min, y_max = max(0, cy-1), min(H, cy+2)
            x_min, x_max = max(0, cx-1), min(W, cx+2)
            raw_mask_val = np.max(gray_map[y_min:y_max, x_min:x_max])
            raw_mask_val = np.clip(raw_mask_val, 0.0, 1.0)

            # Correzione Gamma: mappa la confidenza scura della rete verso una luminosità fisica più realistica
            # Potresti voler esporre questo parametro (es. gamma=0.45) in argparse
            gamma = 0.45 
            target_lum = raw_mask_val ** gamma

            cell_y, cell_x = cy // cell_size, cx // cell_size

            # Check collisioni rapido
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    neighbors.extend(spatial_grid.get((cell_y + dy, cell_x + dx), []))

            temp_radius = 5.0 
            ok = True
            
            if neighbors:
                for ny, nx, nr in neighbors:
                    dist_sq = (cy - ny) ** 2 + (cx - nx) ** 2
                    min_dist_sq = (min_sep_factor * (temp_radius + nr)) ** 2
                    if dist_sq < min_dist_sq:
                        ok = False
                        break

            if ok:
                # 1. Troviamo la stella ideale in base alla luminosità corretta
                ideal_idx = np.searchsorted(intrinsic_lums, target_lum)
                
                jitter_range = max(1, int(pool_max_idx * 0.05))
                pool_idx = ideal_idx + np.random.randint(-jitter_range, jitter_range + 1)
                pool_idx = np.clip(pool_idx, 0, pool_max_idx)

                rgb, alpha, h, w, base_lum = star_pool[pool_idx]
                rgb, alpha = augment_stamp(rgb, alpha)
                radius = max(h, w) / 2.0

                # 2. Calcoliamo la SCALA corretta ignorando la doppia penalizzazione
                scale = target_lum / max(base_lum, 1e-6)
                
                # 3. Applichiamo il jitter alla scala
                jitter_multiplier = np.random.uniform(1.0 - brightness_jitter, 1.0 + brightness_jitter)
                final_intensity = scale * jitter_multiplier
                
                # Hard-clip di sicurezza per evitare di "sparare" stelle esageratamente bruciate 
                # a causa di rapporti sballati tra target_lum e base_lum
                final_intensity = min(final_intensity, 5.0)

                paste_star(canvas, rgb, alpha, cy, cx, final_intensity)
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

    # 2. Setup RAM Pool e Calcolo Energia Intrinseca
    stamp_files = list(Path(args.stamps_dir).glob("*.npz"))
    if not stamp_files:
        print(f"ERRORE: Nessun file .npz trovato nella cartella '{args.stamps_dir}'")
        return

    print(f"Libreria totale rilevata: {len(stamp_files)} stelle.")
    pool_size = min(args.pool_size, len(stamp_files))
    print(f"--> Caricamento e profilazione di {pool_size} stelle in RAM...")

    chosen_stamps = np.random.choice(stamp_files, size=pool_size, replace=False)
    star_pool = []
    
    for p in chosen_stamps:
        with np.load(p) as data:
            rgb = data['rgb'].astype(np.float32)
            alpha = data['alpha'].astype(np.float32)
            h, w = rgb.shape[:2]
            
            # Stima della luminosità intrinseca (valore massimo pesato dall'alpha)
            intrinsic_lum = np.max(rgb * alpha[..., np.newaxis])
            star_pool.append((rgb, alpha, h, w, intrinsic_lum))
            
    # CRITICO: Ordiniamo la libreria dalla stella più debole a quella più luminosa
    star_pool.sort(key=lambda x: x[4])
    print("--> Pool caricato e ordinato con successo.\n")

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
        
        # Normalizzazione automatica [0, 1] per evitare problemi matematici
        if base.max() > 2.0:
            base = base / (65535.0 if base.dtype == np.uint16 else 255.0)
        if star_map.max() > 2.0:
            star_map = star_map / (65535.0 if star_map.dtype == np.uint16 else 255.0)

        # Preparazione mappa scala di grigi per target_lum
        if star_map.ndim == 3:
            gray_map = np.mean(star_map, axis=-1).astype(np.float32)
        else:
            gray_map = star_map.astype(np.float32)

        # Calcolo del numero di stelle effettivo basato sul CSV e sul density_scale
        target_stars = int(star_counts[stem] * args.density_scale)

        canvas, n_placed = synthesize_field(
            base, star_map, gray_map, target_stars, star_pool,
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
            out_path_tiff.parent.mkdir(exist_ok=True)
            save_tiff(out_path_tiff, canvas)
            print(f"  -> Salvato (copia TIFF): {out_path_tiff}")
            
        print() # Riga vuota per pulizia log

if __name__ == "__main__":
    main()
