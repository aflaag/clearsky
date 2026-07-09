"""
Pixel Defect Detector & Corrector

Detects dead/hot pixels in the stretched images (output of astro_stretch.py)
by comparing each pixel to a local median filter, and produces:
  - <output-dir>/<basename>.npy                 (corrected image, target for training)
  - <tiff-dir>/<basename>.tiff                  (corrected TIFF, only with --save-tiff)
  - <mask-tiff-dir>/<basename>.tiff             (mask of corrected pixels, only with --save-mask-tiff)

A pixel is considered hot/dead if the residual relative to the local median
(window --window) exceeds --sigma times the robust standard deviation
(MAD, Median Absolute Deviation) of the residual, estimated only on the valid
region (make_masks.py mask, if available in --mask-dir).

Detection and correction are done per channel (a defect might appear
on only one channel since HLA is an RGB composite from different exposures);
the final mask is the boolean union across channels.

Usage:
    python detect_pixel_defects.py
    python detect_pixel_defects.py --input-dir assets/outputs-stretch --output-dir assets/outputs-pixelfix
    python detect_pixel_defects.py --sigma 5.0 --window 7 --save-tiff --save-mask-tiff
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter, binary_dilation
import argparse

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def to_chw(data):
    """
    Normalizes an array to (C, H, W).
    Returns (data_chw, was_hwc) where was_hwc indicates if the input was (H, W, C),
    in order to restore the original orientation on output.
    """
    if data.ndim == 2:
        return data[None, ...], False
    if data.shape[0] in (1, 3, 4):
        return data, False
    if data.shape[-1] in (1, 3, 4):
        return np.transpose(data, (2, 0, 1)), True
    raise ValueError(f"Unrecognized shape: {data.shape}")


def from_chw(data_chw, was_hwc):
    """Inverse of to_chw: restores the original orientation."""
    if was_hwc:
        return np.transpose(data_chw, (1, 2, 0))
    if data_chw.shape[0] == 1:
        return data_chw[0]
    return data_chw


def robust_sigma(residual, valid=None):
    """Robust sigma estimation using MAD (Median Absolute Deviation)."""
    vals = residual[valid] if valid is not None else residual.ravel()
    if vals.size == 0:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return 1.4826 * mad  # MAD -> Gaussian sigma conversion factor


def detect_bad_pixels(data_chw, valid_mask=None, window=5, sigma_thresh=6.0, dilate=0):
    """
    Detects dead/hot pixels per channel.

    data_chw    : (C, H, W) float32
    valid_mask  : (H, W) bool optional, region on which to estimate the sigma
                  and restrict the detection (e.g., make_masks.py mask)
    window      : local median filter window size
    sigma_thresh: threshold in robust sigmas (MAD) for hot/dead flag
    dilate      : binary dilation iterations on the resulting mask

    Returns:
        bad_mask   : (H, W) bool, union of channels
        bad_per_ch : (C, H, W) bool, mask per channel
        medians    : (C, H, W) float32, local median values (reused for correction)
    """
    C, H, W = data_chw.shape
    bad_per_ch = np.zeros((C, H, W), dtype=bool)
    medians = np.zeros_like(data_chw)

    for c in range(C):
        ch = data_chw[c]
        med = median_filter(ch, size=window)
        medians[c] = med
        residual = ch - med

        sigma = robust_sigma(residual, valid_mask)
        if sigma <= 0:
            # flat channel or sigma not estimable: no defect detectable
            continue

        hot = residual > sigma_thresh * sigma
        dead = residual < -sigma_thresh * sigma
        bad = hot | dead

        if valid_mask is not None:
            bad &= valid_mask  # ignore already invalid edges/padding

        if dilate > 0:
            bad = binary_dilation(bad, iterations=dilate)

        bad_per_ch[c] = bad

    bad_mask = bad_per_ch.any(axis=0)
    return bad_mask, bad_per_ch, medians


def correct_pixels(data_chw, bad_per_ch, medians):
    """
    Replaces detected pixels with the local median filter value
    already computed in detect_bad_pixels (no recalculation).
    """
    corrected = data_chw.copy()
    for c in range(data_chw.shape[0]):
        m = bad_per_ch[c]
        corrected[c][m] = medians[c][m]
    return corrected

def save_as_tiff(image, path):
    """Saves an image as an RGB uint16 TIFF compatible with StarNet2."""
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile not installed: pip install tifffile")

    img = image

    # Convert from CHW to HWC if necessary
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] != img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))

    # Float [0,1] -> uint16 [0,65535]
    img = np.clip(img, 0.0, 1.0)
    img = np.round(img * 65535.0).astype(np.uint16)

    tifffile.imwrite(str(path), img)


def save_mask_as_tiff(bad_mask, path):
    """Saves the boolean mask of corrected pixels as an 8-bit TIFF (0/255)."""
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile not installed: pip install tifffile")
    tifffile.imwrite(str(path), (bad_mask.astype(np.uint8) * 255))


def process_file(npy_path, mask_dir, out_dir, tiff_dir, mask_tiff_dir,
                 window, sigma_thresh, dilate, save_tiff, save_mask_tiff):
    basename = npy_path.stem
    out_npy = out_dir / f"{basename}.npy"

    already_done = (
        out_npy.exists()
        and (not save_tiff or (tiff_dir / f"{basename}.tiff").exists())
        and (not save_mask_tiff or (mask_tiff_dir / f"{basename}.tiff").exists())
    )
    if already_done:
        print(f"[SKIP] {npy_path.name}: already processed")
        return

    data = np.load(npy_path).astype(np.float32)
    data_chw, was_hwc = to_chw(data)

    valid_mask = None
    if mask_dir is not None:
        mask_path = mask_dir / f"{basename}.npy"
        if mask_path.exists():
            valid_mask = np.load(mask_path)
        else:
            print(f"  [WARN] validity mask not found for {basename}: "
                  f"sigma estimation will use the entire image (including any padding)")

    bad_mask, bad_per_ch, medians = detect_bad_pixels(
        data_chw, valid_mask=valid_mask, window=window,
        sigma_thresh=sigma_thresh, dilate=dilate,
    )

    n_bad = int(bad_mask.sum())
    pct = 100 * n_bad / bad_mask.size
    print(f"Processing {npy_path.name}: {n_bad} defective pixels ({pct:.4f}%)")

    corrected_chw = correct_pixels(data_chw, bad_per_ch, medians)
    corrected = from_chw(corrected_chw, was_hwc)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, corrected.astype(np.float32))

    if save_tiff:
        tiff_dir.mkdir(parents=True, exist_ok=True)
        save_as_tiff(corrected, tiff_dir / f"{basename}.tiff")

    if save_mask_tiff:
        mask_tiff_dir.mkdir(parents=True, exist_ok=True)
        save_mask_as_tiff(bad_mask, mask_tiff_dir / f"{basename}.tiff")

    print(f"  [OK] -> {out_npy.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Detects and corrects dead/hot pixels in the stretched images (npy)."
    )
    parser.add_argument("--input-dir", default="assets/outputs-npy",
                         help="Input folder for stretched npy (output of astro_stretch.py)")
    parser.add_argument("--mask-dir", default="assets/outputs-mask",
                         help="Validity masks folder (make_masks.py); used for sigma estimation and search region")
    parser.add_argument("--output-dir", default="assets/outputs-pixelfix",
                         help="Corrected npy folder (target)")
    parser.add_argument("--tiff-dir", default="assets/outputs-pixelfix/tiff",
                         help="Corrected TIFF folder (with --save-tiff)")
    parser.add_argument("--mask-tiff-dir", default="assets/outputs-pixelfix/mask-preview",
                         help="Defect mask TIFF folder (with --save-mask-tiff)")
    parser.add_argument("--window", type=int, default=5,
                         help="Local median filter window in pixels (default: 5)")
    parser.add_argument("--sigma", type=float, default=6.0,
                         help="Threshold in robust sigmas (MAD) for hot/dead flag (default: 6.0)")
    parser.add_argument("--dilate", type=int, default=0,
                         help="Binary dilation iterations on the mask (default: 0)")
    parser.add_argument("--no-valid-mask", action="store_true",
                         help="Ignore validity masks even if present")
    parser.add_argument("--save-tiff", action="store_true",
                         help="Also save the TIFF of the corrected image")
    parser.add_argument("--save-mask-tiff", action="store_true",
                         help="Save a TIFF with the corrected pixels mask")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mask_dir = None if args.no_valid_mask else Path(args.mask_dir)
    out_dir = Path(args.output_dir)
    tiff_dir = Path(args.tiff_dir)
    mask_tiff_dir = Path(args.mask_tiff_dir)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"No npy files found in {input_dir}")
        return

    print(f"Found {len(npy_files)} npy files")
    print(f"window={args.window}px | sigma={args.sigma} | dilate={args.dilate} | "
          f"save_tiff={args.save_tiff} | save_mask_tiff={args.save_mask_tiff}\n")

    for f in npy_files:
        try:
            process_file(f, mask_dir, out_dir, tiff_dir, mask_tiff_dir,
                         args.window, args.sigma, args.dilate,
                         args.save_tiff, args.save_mask_tiff)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    print("\nCompleted.")


if __name__ == "__main__":
    main()
