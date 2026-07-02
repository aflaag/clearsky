import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile  # Aggiunto per gestire correttamente i TIFF a 16-bit


def build_candidate_mask(pixel_mask, patch_size):
    """Integral image (summed area table) sulla maschera pixel.

    Restituisce mappa bool (H, W) dei top-left validi per patch.
    """
    H, W = pixel_mask.shape
    p = patch_size

    if H < p or W < p:
        return np.zeros((H, W), dtype=bool)

    pv = pixel_mask.astype(np.int32)
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(pv, axis=0), axis=1)

    y_max = H - p
    x_max = W - p

    Y, X = np.meshgrid(np.arange(y_max), np.arange(x_max), indexing="ij")

    patch_sums = (
        sat[Y + p, X + p]
        - sat[Y, X + p]
        - sat[Y + p, X]
        + sat[Y, X]
    )

    candidate_mask = np.zeros((H, W), dtype=bool)
    candidate_mask[:y_max, :x_max] = patch_sums == p * p
    return candidate_mask


def sample_position(candidate_mask):
    """Campiona una posizione top-left casuale tra quelle valide."""
    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        return None, None
    idx = np.random.randint(len(ys))
    return int(ys[idx]), int(xs[idx])


def apply_augmentation(patch, flip_h, flip_v, rot_k):
    """Applica augmentation deterministica dato il set di parametri."""
    if flip_h:
        patch = np.fliplr(patch)
    if flip_v:
        patch = np.flipud(patch)
    if rot_k > 0:
        patch = np.rot90(patch, k=rot_k)
    return np.ascontiguousarray(patch)


def random_aug_params():
    """Campiona parametri di augmentation casuali."""
    return {
        "flip_h": np.random.rand() < 0.5,
        "flip_v": np.random.rand() < 0.5,
        "rot_k": int(np.random.randint(4)),
    }


def save_patch(patch, stem, npy_dir, png_dir):
    """Salva patch come NPY float32 e, se png_dir non è None, anche PNG uint8."""
    np.save(npy_dir / f"{stem}.npy", patch.astype(np.float32))
    if png_dir is not None:
        png = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(png, mode="RGB").save(png_dir / f"{stem}.png")


def load_starless_tiff(tiff_path):
    """Carica un TIFF starless prodotto da StarNet2.

    StarNet2 salva TIFF a 16-bit → normalizza in [0, 1] dividendo per 65535.
    """
    arr = tifffile.imread(tiff_path)

    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Dtype inatteso per TIFF starless: {arr.dtype}")


def process_file(
    npy_path,
    starless_path,
    mask_path,
    input_npy_dir,
    input_png_dir,
    target_npy_dir,
    target_png_dir,
    patch_size,
    crops_per_image,
):
    print(f"Processing {npy_path.name}")

    # --- Carica immagine input (sintetica con stelle) ---
    synthetic = np.load(npy_path)  # (H, W, 3) float32
    if synthetic.ndim != 3 or synthetic.shape[-1] != 3:
        print(f"  [SKIP] shape inattesa NPY sintetico: {synthetic.shape}")
        return 0

    H, W, _ = synthetic.shape

    if H < patch_size or W < patch_size:
        print(
            f"  [SKIP] {H}x{W} troppo piccola per patch {patch_size}x{patch_size}"
        )
        return 0

    # --- Carica starless (ground truth) ---
    if not starless_path.exists():
        print(f"  [SKIP] starless non trovato: {starless_path}")
        return 0

    starless = load_starless_tiff(starless_path)

    # --- FIX: Sanitizzazione canali prima del crop ---
    if starless.ndim == 2:
        # Se è in scala di grigi (1 canale), la duplichiamo su 3 canali RGB
        starless = np.stack([starless, starless, starless], axis=-1)
    elif starless.ndim == 3 and starless.shape[-1] == 4:
        # Se ha un canale Alpha (4 canali), lo rimuoviamo tenendo solo RGB
        starless = starless[:, :, :3]

    # Ora il controllo di sicurezza può essere assoluto
    if starless.shape != synthetic.shape:
        print(f"  [SKIP] Mismatch insanabile: starless {starless.shape} vs input sintetico {synthetic.shape}")
        return 0
    # -------------------------------------------------

    # --- Carica maschera ---
    if not mask_path.exists():
        print(f"  [SKIP] maschera non trovata: {mask_path}")
        return 0

    pixel_mask = np.load(mask_path)  # (H, W) bool
    if pixel_mask.shape != (H, W):
        print(f"  [SKIP] maschera shape {pixel_mask.shape} != {H}x{W}")
        return 0

    # --- Calcola posizioni valide ---
    candidate_mask = build_candidate_mask(pixel_mask, patch_size)
    n_candidates = int(candidate_mask.sum())

    if n_candidates == 0:
        print(
            f"  [SKIP] nessuna posizione valida per patch {patch_size}x{patch_size}"
        )
        return 0

    png_label = " + png" if input_png_dir is not None else ""
    print(
        f"  Shape: {H}x{W} | Candidati: {n_candidates} | Crop: {crops_per_image}{png_label}"
    )

    basename = npy_path.stem
    saved = 0
    
    # Soglie
    THRESH_BLACK = 0.05
    THRESH_VAR = 1e-5
    MAX_RETRIES = 10  # Evita loop infiniti se l'immagine è tutta buia

    for i in range(crops_per_image):
        valid_patch_found = False
        
        # Tenta fino a MAX_RETRIES volte di trovare una patch "interessante"
        for attempt in range(MAX_RETRIES):
            y, x = sample_position(candidate_mask)
            if y is None:
                break  # Nessuna posizione valida nell'intera maschera
            
            # Estrai solo l'input temporaneamente per valutarlo
            patch_input = synthetic[y : y + patch_size, x : x + patch_size, :]
            
            patch_mean = float(patch_input.mean())
            patch_var = float(patch_input.var())
            
            # Condizione di validità: non troppo nera E non troppo piatta
            if patch_mean >= THRESH_BLACK and patch_var >= THRESH_VAR:
                valid_patch_found = True
                break
        
        if not valid_patch_found:
            print(f"  [WARN] Impossibile trovare una patch non vuota per il crop {i} dopo {MAX_RETRIES} tentativi")
            continue

        # Se siamo qui, abbiamo coordinate (y, x) ottimali! Procediamo col target
        patch_target = starless[y : y + patch_size, x : x + patch_size, :]

        # Stessa augmentation per entrambe
        aug = random_aug_params()
        patch_input = apply_augmentation(patch_input, **aug)
        patch_target = apply_augmentation(patch_target, **aug)

        stem = f"{basename}_{i:04d}"
        save_patch(patch_input, stem, input_npy_dir, input_png_dir)
        save_patch(patch_target, stem, target_npy_dir, target_png_dir)
        saved += 1

    print(f"  Salvati {saved}/{crops_per_image} crop")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Crea coppie (input_sintetico_con_stelle, starless) per training DDPM."
    )
    parser.add_argument(
        "--npy-dir", 
        default="assets/outputs-npy-synthetic", 
        help="NPY con stelle sintetiche generati da inject_stars.py"
    )
    parser.add_argument(
        "--starless-dir",
        default="assets/outputs-starless",
        help="TIFF starless da StarNet2",
    )
    parser.add_argument(
        "--mask-dir",
        default="assets/outputs-mask",
        help="Maschere da make_masks.py",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset",
        help="Cartella output (default: dataset)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Dimensione patch (default: 256)",
    )
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=50,
        help="Crop per immagine (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed (default: 42)"
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Salva anche le patch PNG (utile per debug visivo)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    npy_dir = Path(args.npy_dir)
    starless_dir = Path(args.starless_dir)
    mask_dir = Path(args.mask_dir)
    out = Path(args.out_dir)

    input_npy_dir = out / "input" / "npy"
    target_npy_dir = out / "target" / "npy"

    # Cartelle PNG create solo se richieste
    input_png_dir = out / "input" / "png" if args.save_png else None
    target_png_dir = out / "target" / "png" if args.save_png else None

    for d in [input_npy_dir, target_npy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        input_png_dir.mkdir(parents=True, exist_ok=True)
        target_png_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(npy_dir.glob("*.npy"))
    if not npy_files:
        print(f"Nessun file .npy trovato in {npy_dir}")
        return

    print(f"Trovati {len(npy_files)} file NPY (Sintetici)")
    print(f"Patch size : {args.patch_size}x{args.patch_size}")
    print(f"Crops/image: {args.crops_per_image}")
    print(f"Save PNG   : {args.save_png}")
    print(f"Output     : {out}\n")

    total = 0
    for npy_file in npy_files:
        stem = npy_file.stem
        total += process_file(
            npy_path=npy_file,
            starless_path=starless_dir / f"{stem}.tif",  # TIFF mantenuto
            mask_path=mask_dir / f"{stem}.npy",
            input_npy_dir=input_npy_dir,
            input_png_dir=input_png_dir,
            target_npy_dir=target_npy_dir,
            target_png_dir=target_png_dir,
            patch_size=args.patch_size,
            crops_per_image=args.crops_per_image,
        )

    print(f"\nCompletato. Totale coppie salvate: {total}")


if __name__ == "__main__":
    main()
