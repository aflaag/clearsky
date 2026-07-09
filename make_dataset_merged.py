#!/usr/bin/env python3
"""
Estrae le patch (input, target) per UNA combinazione di degradazione,
usando le posizioni e le augmentation gia' decise in crop_manifest.json
(generato da build_crop_manifest.py sul riferimento pulito condiviso).

A differenza di make_dataset_su.py / make_dataset_ir.py / make_dataset_sr.py,
questo script NON campiona nulla: nessuna chiamata a np.random per la
posizione o per l'augmentation. Questo garantisce che lo stesso crop_index
dello stesso basename corrisponda ESATTAMENTE alla stessa porzione di cielo
in tutte le combinazioni, condizione necessaria per un confronto valido tra
loro.

Lanciare una volta per ciascuna delle 7 cartelle di input (stesso manifest,
stesso --target-dir condiviso, --input-dir/--out-dir diversi per ogni combo).

Uso tipico:
    python make_dataset_merged.py \
        --manifest crop_manifest.json \
        --input-dir assets/combos/SR_IR \
        --target-dir assets/outputs-clean \
        --out-dir dataset_merged/SR_IR
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


def apply_augmentation(patch, flip_h, flip_v, rot_k):
    if flip_h:
        patch = np.fliplr(patch)
    if flip_v:
        patch = np.flipud(patch)
    if rot_k > 0:
        patch = np.rot90(patch, k=rot_k)
    return np.ascontiguousarray(patch)


def save_patch(patch, stem, npy_dir, png_dir):
    np.save(npy_dir / f"{stem}.npy", patch.astype(np.float32))
    if png_dir is not None:
        png = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(png, mode="RGB").save(png_dir / f"{stem}.png")


def load_image(path):
    """Carica .npy direttamente, oppure .tif a 16/8-bit normalizzato in [0,1]."""
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


def find_file(directory, basename):
    npy_path = directory / f"{basename}.npy"
    if npy_path.exists():
        return npy_path
    tif_path = directory / f"{basename}.tif"
    if tif_path.exists():
        return tif_path
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Estrae crop (input, target) per una combinazione di "
                    "degradazione, usando le posizioni fissate in crop_manifest.json."
    )
    parser.add_argument("--manifest", default="crop_manifest.json")
    parser.add_argument(
        "--input-dir", required=True,
        help="Cartella con l'input di QUESTA combinazione (es. assets/combos/SR_IR)",
    )
    parser.add_argument(
        "--target-dir", required=True,
        help="Cartella del riferimento pulito condiviso (stessa per tutte le combo)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--save-png", action="store_true")
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    patch_size = manifest["patch_size"]
    entries = manifest["entries"]

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    out = Path(args.out_dir)

    input_npy_dir = out / "input" / "npy"
    target_npy_dir = out / "target" / "npy"
    input_png_dir = out / "input" / "png" if args.save_png else None
    target_png_dir = out / "target" / "png" if args.save_png else None

    for d in [input_npy_dir, target_npy_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        input_png_dir.mkdir(parents=True, exist_ok=True)
        target_png_dir.mkdir(parents=True, exist_ok=True)

    by_basename = defaultdict(list)
    for entry in entries:
        by_basename[entry["basename"]].append(entry)

    total_saved = 0
    total_skipped_images = 0

    for basename, crops in sorted(by_basename.items()):
        input_path = find_file(input_dir, basename)
        target_path = find_file(target_dir, basename)

        if input_path is None:
            print(f"[SKIP] {basename}: input non trovato in {input_dir}")
            total_skipped_images += 1
            continue
        if target_path is None:
            print(f"[SKIP] {basename}: target non trovato in {target_dir}")
            total_skipped_images += 1
            continue

        input_img = load_image(input_path)
        target_img = load_image(target_path)

        if input_img.shape != target_img.shape:
            print(
                f"[SKIP] {basename}: mismatch shape input {input_img.shape} "
                f"vs target {target_img.shape}"
            )
            total_skipped_images += 1
            continue

        H, W, _ = input_img.shape
        saved_here = 0

        for entry in crops:
            y, x = entry["y"], entry["x"]
            if y + patch_size > H or x + patch_size > W:
                print(
                    f"  [SKIP] {basename}_{entry['crop_index']:04d}: crop fuori "
                    f"dai bordi ({y},{x}) per shape {(H, W)} - probabile mismatch "
                    f"di risoluzione tra questa combo e il riferimento del manifest"
                )
                continue

            patch_input = input_img[y : y + patch_size, x : x + patch_size, :]
            patch_target = target_img[y : y + patch_size, x : x + patch_size, :]

            aug = {"flip_h": entry["flip_h"], "flip_v": entry["flip_v"], "rot_k": entry["rot_k"]}
            patch_input = apply_augmentation(patch_input, **aug)
            patch_target = apply_augmentation(patch_target, **aug)

            stem = f"{basename}_{entry['crop_index']:04d}"
            save_patch(patch_input, stem, input_npy_dir, input_png_dir)
            save_patch(patch_target, stem, target_npy_dir, target_png_dir)
            saved_here += 1

        print(f"{basename}: {saved_here}/{len(crops)} crop salvati")
        total_saved += saved_here

    print(f"\nCompletato. Crop totali salvati: {total_saved}")
    if total_skipped_images:
        print(f"Immagini saltate per file mancanti/mismatch: {total_skipped_images}")


if __name__ == "__main__":
    main()
