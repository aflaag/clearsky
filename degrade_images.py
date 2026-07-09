"""
Image Degradation Simulator

This script applies a classical super-resolution degradation model
(Gaussian blur -> downsample -> bicubic upsample) to high-resolution 
reference images. The resulting degraded images are used as inputs for the 
low-resolution (SU) degradation combinations in the dataset generation pipeline.
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


def load_tiff_normalized(tiff_path):
    """Loads a TIFF (16-bit or 8-bit) and normalizes it to [0, 1]."""
    arr = tifffile.imread(tiff_path)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected dtype for TIFF: {arr.dtype}")


def match_shape(arr, target_shape):
    """Forces arr to the exact (H, W, C) of target_shape via safety crop/pad.

    zoom() with non-exactly reciprocal factors can produce a shape
    1-2 px different from the original due to rounding.
    """
    if arr.shape == target_shape:
        return arr
    out = np.zeros(target_shape, dtype=arr.dtype)
    h, w, _ = arr.shape
    th, tw, _ = target_shape
    ch, cw = min(h, th), min(w, tw)
    out[:ch, :cw, :] = arr[:ch, :cw, :]
    return out


def degrade_image(hr, scale, blur_sigma):
    """Classical degradation model for SR: y = (x * blur)-downsample-upsample.

    The blur simulates the PSF broadening, the downsample the resolution loss,
    the final bicubic upsample brings the image back to the HR shape so it can
    be used as a conditioning channel for the DDPM (same logic as the image
    "with stars" in the star removal model).
    """
    if blur_sigma > 0:
        # sigma only on spatial dimensions, not on color channels
        blurred = gaussian_filter(hr, sigma=(blur_sigma, blur_sigma, 0))
    else:
        blurred = hr

    lr = zoom(blurred, (1.0 / scale, 1.0 / scale, 1.0), order=3)
    lr = np.clip(lr, 0.0, 1.0)

    zoom_factors = (hr.shape[0] / lr.shape[0], hr.shape[1] / lr.shape[1], 1.0)
    degraded = zoom(lr, zoom_factors, order=3)
    degraded = np.clip(degraded, 0.0, 1.0)

    return match_shape(degraded, hr.shape)


def save_png_preview(arr, path):
    png = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(png, mode="RGB").save(path)


def process_file(tiff_path, out_npy_dir, out_png_dir, scale, blur_sigma):
    print(f"Processing {tiff_path.name}")
    hr = load_tiff_normalized(tiff_path)

    if hr.ndim == 2:
        hr = np.stack([hr, hr, hr], axis=-1)
    elif hr.ndim == 3 and hr.shape[-1] == 4:
        hr = hr[:, :, :3]

    degraded = degrade_image(hr, scale=scale, blur_sigma=blur_sigma)

    stem = tiff_path.stem
    np.save(out_npy_dir / f"{stem}.npy", degraded.astype(np.float32))

    if out_png_dir is not None:
        save_png_preview(degraded, out_png_dir / f"{stem}.png")

    print(f"  HR: {hr.shape} -> LR x{scale} -> degraded (upsampled): {degraded.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Generates degraded versions (blur + downsample + upsample) of the full HR images, for the super-resolution dataset."
    )
    parser.add_argument(
        "--tiff-dir",
        default="assets/outputs-tiff",
        help="HR TIFFs (output of astro_stretch.py --save-tiff)",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/outputs-degraded",
        help="Output folder (default: assets/outputs-degraded)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Downsampling factor (default: 4)"
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.5,
        help="Sigma of the pre-downsample Gaussian blur, 0 to disable (default: 1.5)",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG previews of the degraded images (for visual inspection)",
    )
    args = parser.parse_args()

    tiff_dir = Path(args.tiff_dir)
    out_dir = Path(args.out_dir)
    out_npy_dir = out_dir / "npy"
    out_png_dir = out_dir / "png" if args.save_png else None

    out_npy_dir.mkdir(parents=True, exist_ok=True)
    if out_png_dir is not None:
        out_png_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(tiff_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No TIFF files found in {tiff_dir}")
        return

    print(f"Found {len(tiff_files)} HR TIFF files")
    print(f"Scale: x{args.scale} | Blur sigma: {args.blur_sigma} | Save PNG: {args.save_png}\n")

    for tiff_path in tiff_files:
        process_file(tiff_path, out_npy_dir, out_png_dir, args.scale, args.blur_sigma)

    print("\nCompleted.")


if __name__ == "__main__":
    main()
