"""
FITS Batch Processor with Adaptive Arcsinh Stretch

This script processes directories of astronomical FITS images (typically RGB) 
by applying an adaptive arcsinh stretch. It calculates the black and white 
points based on configurable percentiles and stretches the data to enhance 
faint details while preventing star core clipping.

Key Features:
- Processes single FITS files or batches of them.
- Supports "paired" processing: calculates the optimal stretch parameters on a 
  reference (e.g., noisy) image and strictly applies those exact same parameters 
  to a paired (e.g., denoised/background-extracted) image to ensure consistent 
  brightness and background scaling.
- Outputs normalized 32-bit floating-point `.npy` arrays.
- Optionally outputs 16-bit `.tif` images for standard visual inspection.
"""

import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import tifffile


def compute_stretch_params(data_array, black_percentile=1.0, white_percentile=99.7, linked=False):
    """Calculates black/white points (per channel, or a single value if linked) from a (C,H,W) array."""
    clean = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    if linked:
        black = np.percentile(clean, black_percentile)
        white = np.percentile(clean, white_percentile)
        if white <= black:
            white = black + 1e-10
        return {"linked": True, "black": float(black), "white": float(white)}
    else:
        channels = []
        for c in range(clean.shape[0]):
            channel = clean[c]
            black = np.percentile(channel, black_percentile)
            white = np.percentile(channel, white_percentile)
            if white <= black:
                white = black + 1e-10
            channels.append((float(black), float(white)))
        return {"linked": False, "channels": channels}


def apply_arcsinh_stretch(data_array, params, stretch=8.0):
    """Applies the arcsinh stretch using ALREADY CALCULATED (fixed) black/white parameters."""
    clean = np.nan_to_num(data_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    stretched = np.zeros_like(clean, dtype=np.float32)

    for c in range(clean.shape[0]):
        channel = clean[c]
        black, white = (params["black"], params["white"]) if params["linked"] else params["channels"][c]
        x = (channel - black) / (white - black)
        x = np.clip(x, 0.0, 1.0)
        y = np.arcsinh(stretch * x)
        y /= np.arcsinh(stretch)
        stretched[c] = np.clip(y, 0.0, 1.0)

    return stretched


def adaptive_arcsinh_stretch(data_array, stretch=8.0, black_percentile=1.0, white_percentile=99.7, linked=False):
    """
    Backward-compatible wrapper: calculates the parameters and applies the stretch in a single step.
    Now returns (stretched, params) instead of just the array, to allow reusing the
    parameters on a paired image.
    """
    params = compute_stretch_params(data_array, black_percentile, white_percentile, linked)
    stretched = apply_arcsinh_stretch(data_array, params, stretch=stretch)
    return stretched, params


def load_fits_rgb(fits_path):
    """Loads a FITS file and normalizes it to (3,H,W) float32. Raises ValueError if invalid."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError("missing data")
    if data.ndim != 3:
        raise ValueError(f"invalid shape {data.shape}")
    if data.shape[-1] == 3:
        data = np.transpose(data, (2, 0, 1))
    if data.shape[0] != 3:
        raise ValueError("requires 3 RGB channels")

    return data.astype(np.float32)


def save_outputs(rgb_stretched, basename, output_npy_dir, output_tiff_dir, save_tiff):
    rgb = np.transpose(rgb_stretched, (1, 2, 0))[..., :3]

    output_npy_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_npy_dir / f"{basename}.npy", rgb.astype(np.float32))

    if save_tiff:
        output_tiff_dir.mkdir(parents=True, exist_ok=True)
        tiff_data = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
        tifffile.imwrite(output_tiff_dir / f"{basename}.tif", tiff_data)


def process_fits_file(
    fits_path,
    output_npy_dir,
    output_tiff_dir,
    linked,
    save_tiff,
    paired_fits_path=None,
    paired_output_npy_dir=None,
    paired_output_tiff_dir=None,
):
    """
    Processes a FITS file by applying an arcsinh stretch.
    If paired_fits_path is provided (e.g., GraXpert output of the same target), the stretch 
    parameters are calculated ONLY on the main (noisy) image and applied IDENTICALLY 
    to the paired image, to avoid background/scale mismatches between input and target.
    """
    print(f"Processing {fits_path}")
    basename = fits_path.stem

    npy_path = output_npy_dir / f"{basename}.npy"
    tiff_path = output_tiff_dir / f"{basename}.tif" if save_tiff else None

    paired_npy_path = paired_output_npy_dir / f"{basename}.npy" if paired_fits_path else None
    paired_tiff_path = (
        paired_output_tiff_dir / f"{basename}.tif" if (paired_fits_path and save_tiff) else None
    )

    already_done = npy_path.exists() and (not save_tiff or tiff_path.exists())
    if paired_fits_path is not None:
        already_done = already_done and paired_npy_path.exists() and (not save_tiff or paired_tiff_path.exists())

    if already_done:
        print(f"[SKIP] {fits_path.name}: already processed")
        return

    try:
        data = load_fits_rgb(fits_path)
    except ValueError as e:
        print(f"[SKIP] {fits_path.name}: {e}")
        return
    except Exception as e:
        print(f"[ERROR] {fits_path.name}")
        print(e)
        return

    stretched, params = adaptive_arcsinh_stretch(data, linked=linked)
    save_outputs(stretched, basename, output_npy_dir, output_tiff_dir, save_tiff)
    print(f"[OK] {fits_path.name} (npy{' + 16-bit tiff' if save_tiff else ''})")

    if paired_fits_path is None:
        return

    if not paired_fits_path.exists():
        print(f"[WARN] {fits_path.name}: missing corresponding {paired_fits_path.name}, skipping paired output")
        return

    try:
        paired_data = load_fits_rgb(paired_fits_path)
    except ValueError as e:
        print(f"[SKIP] {paired_fits_path.name}: {e}")
        return
    except Exception as e:
        print(f"[ERROR] {paired_fits_path.name}")
        print(e)
        return

    paired_stretched = apply_arcsinh_stretch(paired_data, params)
    save_outputs(paired_stretched, basename, paired_output_npy_dir, paired_output_tiff_dir, save_tiff)
    print(f"[OK] {paired_fits_path.name} (paired, stretch reused from {fits_path.name})")


def main():
    parser = argparse.ArgumentParser(description="Batch processing FITS with adaptive arcsinh stretch.")
    parser.add_argument("--input-dir", default="assets/inputs", help="Folder containing the noisy FITS")
    parser.add_argument("--output-npy", default="assets/outputs-npy", help="Output folder for .npy files")
    parser.add_argument("--output-tiff", default="assets/outputs-tiff", help="Output folder for TIFF files (with --save-tiff)")
    parser.add_argument("--linked", action="store_true", help="Use the same stretch across all channels")
    parser.add_argument("--save-tiff", action="store_true", help="Also save the 16-bit stretched TIFF")

    parser.add_argument(
        "--paired-dir",
        default=None,
        help="Folder containing paired FITS (e.g., GraXpert output), using the same basename as --input-dir. "
        "If provided, the stretch is calculated ONLY on the noisy FITS and applied identically to the paired one.",
    )
    parser.add_argument("--paired-output-npy", default="assets/outputs-npy-clean")
    parser.add_argument("--paired-output-tiff", default="assets/outputs-tiff-clean")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_npy_dir = Path(args.output_npy)
    output_tiff_dir = Path(args.output_tiff)
    output_npy_dir.mkdir(parents=True, exist_ok=True)

    paired_dir = Path(args.paired_dir) if args.paired_dir else None
    paired_output_npy_dir = Path(args.paired_output_npy)
    paired_output_tiff_dir = Path(args.paired_output_tiff)
    if paired_dir:
        paired_output_npy_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(
        list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit")) + list(input_dir.glob("*.FITS"))
    )
    print(f"Found {len(fits_files)} FITS files in {input_dir}.")

    for fits_file in fits_files:
        paired_fits_path = None
        if paired_dir:
            candidates = [paired_dir / f"{fits_file.stem}{ext}" for ext in (".fits", ".fit", ".FITS")]
            paired_fits_path = next((c for c in candidates if c.exists()), None)
            if paired_fits_path is None:
                print(f"[WARN] {fits_file.name}: no match found in {paired_dir}, skipping paired output")

        process_fits_file(
            fits_file,
            output_npy_dir,
            output_tiff_dir,
            linked=args.linked,
            save_tiff=args.save_tiff,
            paired_fits_path=paired_fits_path,
            paired_output_npy_dir=paired_output_npy_dir if paired_dir else None,
            paired_output_tiff_dir=paired_output_tiff_dir if paired_dir else None,
        )

    print("Completed.")


if __name__ == "__main__":
    main()
