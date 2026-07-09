"""
NPY to TIFF Converter

Converts a folder of float32 NPY files in [0, 1] to 16-bit TIFFs.

Acts as glue code when a pipeline sequence requires both inject_stars.py and 
degrade_images.py: both output NPY files, but degrade_images.py expects 
TIFFs as input (--tiff-dir).
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile


def main():
    parser = argparse.ArgumentParser(
        description="Converts [0,1] float32 NPY files to 16-bit TIFFs."
    )
    parser.add_argument("--npy-dir", required=True, help="Input directory containing NPY files")
    parser.add_argument("--out-dir", required=True, help="Output directory for 16-bit TIFF files")
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(npy_dir.glob("*.npy"))
    if not files:
        print(f"No .npy files found in {npy_dir}")
        return

    for f in files:
        arr = np.load(f).astype(np.float32)
        arr16 = np.clip(arr * 65535.0, 0, 65535).astype(np.uint16)
        tifffile.imwrite(out_dir / f"{f.stem}.tif", arr16)

    print(f"Converted {len(files)} files to {out_dir}")


if __name__ == "__main__":
    main()
