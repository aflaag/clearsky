"""
Applies real defects (dead/hot pixels) to any composition,
using the positions and values ACTUALLY observed in the original
acquisition - not a statistical synthesis.

defect_mask  = where the original image (with defects) and the corrected one
               (detect_pixel_defects.py) differ
defect_value = the ACTUAL observed value at those positions, in the original

It must be applied LAST in the chain (after any inject_stars.py and
degrade_images.py): physically, dead/hot pixels are a sensor readout defect, 
so the value reported at those positions is largely independent of the 
underlying "true" signal (whether synthetic or real). For this reason, 
the default mode is "replace" (copies the real observed value), not 
"additive" (adds a delta).
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile


def load_image(path):
    """Loads .npy directly, or 16/8-bit .tif normalized in [0,1]."""
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)

    arr = tifffile.imread(path)
    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected dtype for {path}: {arr.dtype}")

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
        description="Applies real defects to a composition (last step in the chain)."
    )
    parser.add_argument(
        "--original-dir", default="assets/outputs-npy",
        help="Original stretched NPYs WITH defects (output of astro_stretch.py)",
    )
    parser.add_argument(
        "--corrected-dir", default="assets/clean/pixelfix-npy",
        help="Corrected NPYs (output of detect_pixel_defects.py --output-dir)",
    )
    parser.add_argument(
        "--composite-dir", required=True,
        help="Current composition: .npy or .tif (output of inject_stars.py / "
             "degrade_images.py, or directly the clean reference if "
             "defects are the only active degradation in this combo)",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--mode", choices=["replace", "additive"], default="replace",
        help="replace (default): replaces the pixel with the real observed "
             "value, physically more correct for fixed dead/hot pixels. "
             "additive: adds the original-corrected delta to the current value.",
    )
    parser.add_argument(
        "--defect-eps", type=float, default=1e-6,
        help="Threshold on the absolute difference between original and corrected "
             "to count a pixel as defective (default: 1e-6, as in make_dataset_ir.py)",
    )
    args = parser.parse_args()

    original_dir = Path(args.original_dir)
    corrected_dir = Path(args.corrected_dir)
    composite_dir = Path(args.composite_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    composite_files = sorted(composite_dir.glob("*.npy")) + sorted(composite_dir.glob("*.tif"))
    if not composite_files:
        print(f"No .npy/.tif files found in {composite_dir}")
        return

    n_ok = 0
    n_skipped = 0
    total_defect_px = 0

    for composite_path in composite_files:
        basename = composite_path.stem

        original_path = find_file(original_dir, basename)
        corrected_path = find_file(corrected_dir, basename)

        if original_path is None or corrected_path is None:
            print(f"[SKIP] {basename}: missing original or corrected")
            n_skipped += 1
            continue

        original = load_image(original_path)
        corrected = load_image(corrected_path)
        composite = load_image(composite_path)

        if not (original.shape == corrected.shape == composite.shape):
            print(
                f"[SKIP] {basename}: inconsistent shapes - original {original.shape}, "
                f"corrected {corrected.shape}, composite {composite.shape}"
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
        print(f"{basename}: {n_defect_px} defective pixels applied")
        n_ok += 1

    print(f"\nCompleted. Processed images: {n_ok}, skipped: {n_skipped}")
    print(f"Total defective pixels applied: {total_defect_px}")


if __name__ == "__main__":
    main()
