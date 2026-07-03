import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_patch_sum_map(mask, patch_size):
    """Integral image (summed area table) su una maschera binaria.

    Restituisce (H, W) int64: somma dei pixel True in ogni patch patch_size x patch_size
    con top-left in (y, x). Valori validi solo per y <= H-p, x <= W-p (0 altrove).
    """
    H, W = mask.shape
    p = patch_size

    if H < p or W < p:
        return np.zeros((H, W), dtype=np.int64)

    mv = mask.astype(np.int32)
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(mv, axis=0), axis=1)

    y_max = H - p
    x_max = W - p
    Y, X = np.meshgrid(np.arange(y_max), np.arange(x_max), indexing="ij")

    patch_sums = (
        sat[Y + p, X + p]
        - sat[Y, X + p]
        - sat[Y + p, X]
        + sat[Y, X]
    )

    sum_map = np.zeros((H, W), dtype=np.int64)
    sum_map[:y_max, :x_max] = patch_sums
    return sum_map


def build_candidate_mask(pixel_mask, patch_size):
    """Top-left validi: patch interamente dentro la regione valida."""
    sum_map = build_patch_sum_map(pixel_mask, patch_size)
    return sum_map == patch_size * patch_size


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


def process_file(
    input_path,
    target_path,
    mask_path,
    input_npy_dir,
    input_png_dir,
    target_npy_dir,
    target_png_dir,
    patch_size,
    crops_per_image,
    defect_ratio,
    min_defect_pixels,
    defect_eps,
):
    print(f"Processing {input_path.name}")

    # --- Carica immagine input (originale, con dead/hot pixel) ---
    defective = np.load(input_path)  # (H, W, 3) float32
    if defective.ndim != 3 or defective.shape[-1] != 3:
        print(f"  [SKIP] shape inattesa NPY input: {defective.shape}")
        return 0

    H, W, _ = defective.shape

    if H < patch_size or W < patch_size:
        print(
            f"  [SKIP] {H}x{W} troppo piccola per patch {patch_size}x{patch_size}"
        )
        return 0

    # --- Carica target (corretto da detect_pixel_defects.py) ---
    if not target_path.exists():
        print(f"  [SKIP] target non trovato: {target_path}")
        return 0

    target = np.load(target_path)

    if target.shape != defective.shape:
        print(f"  [SKIP] Mismatch shape: target {target.shape} vs input {defective.shape}")
        return 0

    # --- Carica maschera di validità ---
    if not mask_path.exists():
        print(f"  [SKIP] maschera non trovata: {mask_path}")
        return 0

    pixel_mask = np.load(mask_path)  # (H, W) bool
    if pixel_mask.shape != (H, W):
        print(f"  [SKIP] maschera shape {pixel_mask.shape} != {H}x{W}")
        return 0

    # --- Mappa dei difetti: dove input e target differiscono ---
    # (equivalente alla bad_mask di detect_pixel_defects.py, ottenuta per sottrazione,
    # come già intuito: dove la correzione ha agito, input != target)
    defect_mask = np.abs(defective - target).sum(axis=-1) > defect_eps

    # --- Posizioni candidate (patch interamente nella regione valida) ---
    candidate_mask = build_candidate_mask(pixel_mask, patch_size)
    n_candidates = int(candidate_mask.sum())

    if n_candidates == 0:
        print(
            f"  [SKIP] nessuna posizione valida per patch {patch_size}x{patch_size}"
        )
        return 0

    # --- Posizioni la cui patch contiene abbastanza pixel difettosi ---
    defect_count_map = build_patch_sum_map(defect_mask, patch_size)
    defect_candidate_mask = candidate_mask & (defect_count_map >= min_defect_pixels)
    n_defect_candidates = int(defect_candidate_mask.sum())

    png_label = " + png" if input_png_dir is not None else ""
    print(
        f"  Shape: {H}x{W} | Candidati: {n_candidates} "
        f"(con difetti: {n_defect_candidates}) | Crop: {crops_per_image}{png_label}"
    )

    if n_defect_candidates == 0:
        print(
            f"  [WARN] nessuna patch contiene >= {min_defect_pixels} pixel difettosi: "
            f"tutti i crop saranno campionati a caso (controlla soglie di detect_pixel_defects.py)"
        )

    basename = input_path.stem
    saved = 0

    # Soglie (identiche a make_dataset_sr.py)
    THRESH_BLACK = 0.05
    THRESH_VAR = 1e-5
    MAX_RETRIES = 10  # Evita loop infiniti se l'immagine è tutta buia

    for i in range(crops_per_image):
        # Con probabilità defect_ratio privilegia patch con difetti reali;
        # il resto sono esempi "negativi" (il modello deve imparare anche
        # a non alterare pixel già puliti)
        use_defect_pool = n_defect_candidates > 0 and np.random.rand() < defect_ratio
        pool = defect_candidate_mask if use_defect_pool else candidate_mask

        valid_patch_found = False

        for attempt in range(MAX_RETRIES):
            y, x = sample_position(pool)
            if y is None:
                break  # Nessuna posizione valida in questo pool

            patch_input = defective[y : y + patch_size, x : x + patch_size, :]

            patch_mean = float(patch_input.mean())
            patch_var = float(patch_input.var())

            # Condizione di validità: non troppo nera E non troppo piatta
            if patch_mean >= THRESH_BLACK and patch_var >= THRESH_VAR:
                valid_patch_found = True
                break

        if not valid_patch_found:
            print(
                f"  [WARN] Impossibile trovare una patch valida per il crop {i} "
                f"dopo {MAX_RETRIES} tentativi"
            )
            continue

        patch_target = target[y : y + patch_size, x : x + patch_size, :]

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
        description="Crea coppie (input_con_difetti, corretto) per training DDPM di image restoration."
    )
    parser.add_argument(
        "--input-dir",
        default="assets/outputs-npy",
        help="NPY stretchati originali (con dead/hot pixel), output di astro_stretch.py",
    )
    parser.add_argument(
        "--target-dir",
        default="assets/outputs-pixelfix",
        help="NPY corretti, output di detect_pixel_defects.py",
    )
    parser.add_argument(
        "--mask-dir",
        default="assets/outputs-mask",
        help="Maschere da make_masks.py",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset_ir",
        help="Cartella output (default: dataset_ir)",
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
        "--defect-ratio",
        type=float,
        default=0.8,
        help="Frazione di crop campionati preferenzialmente da patch con difetti (default: 0.8)",
    )
    parser.add_argument(
        "--min-defect-pixels",
        type=int,
        default=1,
        help="Pixel corretti minimi richiesti in una patch 'con difetti' (default: 1)",
    )
    parser.add_argument(
        "--defect-eps",
        type=float,
        default=1e-6,
        help="Soglia sulla differenza assoluta input-target per contare un pixel come difetto (default: 1e-6)",
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

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
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

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        print(f"Nessun file .npy trovato in {input_dir}")
        return

    print(f"Trovati {len(input_files)} file NPY (input)")
    print(f"Patch size      : {args.patch_size}x{args.patch_size}")
    print(f"Crops/image     : {args.crops_per_image}")
    print(f"Defect ratio    : {args.defect_ratio}")
    print(f"Min defect px   : {args.min_defect_pixels}")
    print(f"Save PNG        : {args.save_png}")
    print(f"Output          : {out}\n")

    total = 0
    for input_file in input_files:
        stem = input_file.stem
        total += process_file(
            input_path=input_file,
            target_path=target_dir / f"{stem}.npy",
            mask_path=mask_dir / f"{stem}.npy",
            input_npy_dir=input_npy_dir,
            input_png_dir=input_png_dir,
            target_npy_dir=target_npy_dir,
            target_png_dir=target_png_dir,
            patch_size=args.patch_size,
            crops_per_image=args.crops_per_image,
            defect_ratio=args.defect_ratio,
            min_defect_pixels=args.min_defect_pixels,
            defect_eps=args.defect_eps,
        )

    print(f"\nCompletato. Totale coppie salvate: {total}")


if __name__ == "__main__":
    main()
