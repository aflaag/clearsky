#!/usr/bin/env python3
"""
Applica i difetti reali (dead/hot pixel) a una qualsiasi composizione,
usando le posizioni e i valori osservati REALMENTE nell'acquisizione
originale - non una sintesi statistica.

defect_mask  = dove l'immagine originale (con difetti) e quella corretta
               (detect_pixel_defects.py) differiscono
defect_value = il valore osservato REALE in quelle posizioni, nell'originale

Va applicato per ULTIMO nella catena (dopo eventuale inject_stars.py e
degrade_images.py): fisicamente i dead/hot pixel sono un difetto del
sensore in lettura, quindi il valore riportato in quelle posizioni e'
in gran parte indipendente dal segnale "vero" sottostante (sintetico o
reale che sia). Per questo la modalita' di default e' "replace" (copia
il valore osservato reale), non "additive" (somma un delta).
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile


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
        description="Applica i difetti reali a una composizione (ultimo step della catena)."
    )
    parser.add_argument(
        "--original-dir", default="assets/outputs-npy",
        help="NPY stretchati originali CON difetti (output di astro_stretch.py)",
    )
    parser.add_argument(
        "--corrected-dir", default="assets/clean/pixelfix-npy",
        help="NPY corretti (output di detect_pixel_defects.py --output-dir)",
    )
    parser.add_argument(
        "--composite-dir", required=True,
        help="Composizione corrente: .npy o .tif (output di inject_stars.py / "
             "degrade_images.py, oppure direttamente il riferimento pulito se "
             "i difetti sono l'unica degradazione attiva in questa combo)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--mode", choices=["replace", "additive"], default="replace",
        help="replace (default): sostituisce il pixel con il valore osservato "
             "reale, fisicamente piu' corretto per dead/hot pixel fissi. "
             "additive: somma il delta originale-corretto al valore corrente.",
    )
    parser.add_argument(
        "--defect-eps", type=float, default=1e-6,
        help="Soglia sulla differenza assoluta originale-corretto per contare "
             "un pixel come difetto (default: 1e-6, come in make_dataset_ir.py)",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    corrected_dir = Path(args.corrected_dir)
    composite_dir = Path(args.composite_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    composite_files = sorted(composite_dir.glob("*.npy")) + sorted(composite_dir.glob("*.tif"))
    if not composite_files:
        print(f"Nessun file .npy/.tif trovato in {composite_dir}")
        return

    n_ok = 0
    n_skipped = 0
    total_defect_px = 0

    for composite_path in composite_files:
        basename = composite_path.stem

        original_path = find_file(original_dir, basename)
        corrected_path = find_file(corrected_dir, basename)

        if original_path is None or corrected_path is None:
            print(f"[SKIP] {basename}: originale o corretto mancante")
            n_skipped += 1
            continue

        original = load_image(original_path)
        corrected = load_image(corrected_path)
        composite = load_image(composite_path)

        if not (original.shape == corrected.shape == composite.shape):
            print(
                f"[SKIP] {basename}: shape incoerenti - originale {original.shape}, "
                f"corretto {corrected.shape}, composito {composite.shape}"
            )
            n_skipped += 1
            continue

        defect_mask = np.abs(original - corrected).sum(axis=-1) > args.defect_eps

        result = composite.copy()
        if args.mode == "replace":
            result[defect_mask] = original[defect_mask]
        else:
            delta = original - corrected
            result[defect_mask] = composite[defect_mask] + delta[defect_mask]

        np.save(out_dir / f"{basename}.npy", result.astype(np.float32))

        n_defect_px = int(defect_mask.sum())
        total_defect_px += n_defect_px
        print(f"{basename}: {n_defect_px} pixel difettosi applicati")
        n_ok += 1

    print(f"\nCompletato. Immagini processate: {n_ok}, skippate: {n_skipped}")
    print(f"Pixel difettosi totali applicati: {total_defect_px}")


if __name__ == "__main__":
    main()
