#!/usr/bin/env python3
"""
Genera il manifest condiviso dei crop (posizione + augmentation) a partire
dal riferimento "doppiamente pulito" (starless + defect-free, HR).

Il manifest garantisce che make_dataset_merged.py estragga LA STESSA porzione
di cielo, con LA STESSA augmentation, da tutte le 7 combinazioni di
degradazione: l'allineamento non dipende dal seed dei singoli
make_dataset_xx.py, che si e' visto divergere tra script diversi anche a
parita' di seed (il retry loop consuma un numero di chiamate RNG che dipende
dal contenuto pixel, quindi diverso tra combo diverse).

Le posizioni vengono scelte e validate SOLO sul riferimento pulito condiviso:
ogni combo derivata (SR, IR, SU, e le loro unioni) eredita esattamente le
stesse coordinate, senza ricampionare.

Uso tipico:
    python build_crop_manifest.py \
        --clean-dir assets/outputs-clean \
        --mask-dir assets/outputs-mask \
        --out crop_manifest.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


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


def random_aug_params():
    """Campiona parametri di augmentation casuali."""
    return {
        "flip_h": bool(np.random.rand() < 0.5),
        "flip_v": bool(np.random.rand() < 0.5),
        "rot_k": int(np.random.randint(4)),
    }


def load_reference(path):
    """Carica il riferimento pulito condiviso: .npy oppure .tif a 16/8-bit."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    arr = tifffile.imread(path)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Dtype inatteso per {path}: {arr.dtype}")

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def find_reference_file(clean_dir, basename):
    """Cerca prima .npy poi .tif per il basename dato."""
    npy_path = clean_dir / f"{basename}.npy"
    if npy_path.exists():
        return npy_path
    tif_path = clean_dir / f"{basename}.tif"
    if tif_path.exists():
        return tif_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Genera crop_manifest.json condiviso tra le 7 combinazioni "
                    "di degradazione, campionando posizione e augmentation UNA "
                    "SOLA VOLTA sul riferimento pulito (starless + defect-free, HR)."
    )
    parser.add_argument(
        "--clean-dir",
        default="assets/outputs-clean",
        help="Riferimento doppiamente pulito (starless + defect-free, HR). "
             "Cerca prima .npy poi .tif per ciascun basename.",
    )
    parser.add_argument(
        "--mask-dir",
        default="assets/outputs-mask",
        help="Maschere di validita' da make_masks.py",
    )
    parser.add_argument(
        "--out",
        default="crop_manifest.json",
        help="Percorso di output del manifest (default: crop_manifest.json)",
    )
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--crops-per-image", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thresh-black", type=float, default=0.05,
        help="Media minima della patch sul riferimento pulito (default: 0.05)",
    )
    parser.add_argument(
        "--thresh-var", type=float, default=1e-5,
        help="Varianza minima della patch sul riferimento pulito (default: 1e-5)",
    )
    parser.add_argument("--max-retries", type=int, default=10)
    args = parser.parse_args()

    np.random.seed(args.seed)

    clean_dir = Path(args.clean_dir)
    mask_dir = Path(args.mask_dir)

    mask_files = sorted(mask_dir.glob("*.npy"))
    if not mask_files:
        print(f"Nessuna maschera trovata in {mask_dir}")
        return

    manifest = []
    n_basenames_ok = 0
    n_basenames_skipped = 0
    total_saved = 0

    for mask_path in mask_files:
        basename = mask_path.stem
        ref_path = find_reference_file(clean_dir, basename)
        if ref_path is None:
            print(f"[SKIP] {basename}: riferimento pulito non trovato in {clean_dir}")
            n_basenames_skipped += 1
            continue

        reference = load_reference(ref_path)
        if reference.ndim != 3 or reference.shape[-1] != 3:
            print(f"[SKIP] {basename}: shape inattesa {reference.shape}")
            n_basenames_skipped += 1
            continue

        pixel_mask = np.load(mask_path)
        H, W = pixel_mask.shape
        if reference.shape[:2] != (H, W):
            print(
                f"[SKIP] {basename}: mismatch riferimento {reference.shape[:2]} "
                f"vs maschera {(H, W)}"
            )
            n_basenames_skipped += 1
            continue

        candidate_mask = build_candidate_mask(pixel_mask, args.patch_size)
        if candidate_mask.sum() == 0:
            print(f"[SKIP] {basename}: nessuna posizione valida per patch {args.patch_size}")
            n_basenames_skipped += 1
            continue

        saved_for_this_image = 0
        for i in range(args.crops_per_image):
            valid = False
            y = x = None
            for _ in range(args.max_retries):
                y, x = sample_position(candidate_mask)
                if y is None:
                    break
                patch = reference[y : y + args.patch_size, x : x + args.patch_size, :]
                if float(patch.mean()) >= args.thresh_black and float(patch.var()) >= args.thresh_var:
                    valid = True
                    break

            if not valid:
                print(f"  [WARN] {basename}: crop {i} scartato dopo {args.max_retries} tentativi")
                continue

            aug = random_aug_params()
            manifest.append({
                "basename": basename,
                "crop_index": i,
                "y": y,
                "x": x,
                **aug,
            })
            saved_for_this_image += 1

        print(f"{basename}: {saved_for_this_image}/{args.crops_per_image} crop nel manifest")
        total_saved += saved_for_this_image
        n_basenames_ok += 1

    with open(args.out, "w") as f:
        json.dump(
            {
                "patch_size": args.patch_size,
                "seed": args.seed,
                "entries": manifest,
            },
            f,
            indent=2,
        )

    print(f"\nCompletato. Immagini ok: {n_basenames_ok}, skippate: {n_basenames_skipped}")
    print(f"Crop totali nel manifest: {total_saved}")
    print(f"Manifest salvato in: {args.out}")


if __name__ == "__main__":
    main()
