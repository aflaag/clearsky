#!/usr/bin/env python3
"""
Converte una cartella di NPY float32 in [0,1] in TIFF 16-bit.

Serve da collante quando una combo richiede sia inject_stars.py sia
degrade_images.py in sequenza: entrambi producono NPY in output, ma
degrade_images.py richiede TIFF in input (--tiff-dir).
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(
        description="Converte NPY [0,1] float32 in TIFF 16-bit."
    )
    parser.add_argument("--npy-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        print(f"Nessun file .npy trovato in {npy_dir}")
        return

    for f in files:
        arr = np.load(f).astype(np.float32)
        arr16 = np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
        tifffile.imwrite(out_dir / f"{f.stem}.tif", arr16)

    print(f"Convertiti {len(files)} file in {out_dir}")


if __name__ == "__main__":
    main()
