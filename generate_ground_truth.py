
# generate_ground_truth.py
#
# Pseudo ground truth generator for astronomical FITS images.
#
# Requirements:
#   pip install numpy astropy bm3d pywavelets scipy
#
# BM3D expects Gaussian noise. Sigma is estimated robustly using
# the MAD estimator on the HH wavelet coefficients.

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
import pywt
from bm3d import bm3d


def load_fits_rgb(path):
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(np.float32)
        header = hdul[0].header.copy()

    if data.ndim != 3:
        raise ValueError(f"Expected 3D FITS, got {data.shape}")

    if data.shape[-1] == 3:
        data = np.transpose(data, (2, 0, 1))

    if data.shape[0] != 3:
        raise ValueError("Image must have exactly 3 channels")

    return data, header


def print_stats(name, img):
    valid = np.isfinite(img)

    print("=" * 60)
    print(name)
    print("=" * 60)
    print("shape :", img.shape)
    print("dtype :", img.dtype)
    print("NaN   :", np.isnan(img).sum())
    print("Inf   :", np.isinf(img).sum())

    if valid.any():
        v = img[valid]
        print("min   :", float(v.min()))
        print("max   :", float(v.max()))
        print("mean  :", float(v.mean()))
        print("std   :", float(v.std()))
    print()


def estimate_sigma_mad(img):
    coeffs = pywt.wavedec2(img, wavelet="db2", level=1)
    _, (lh, hl, hh) = coeffs
    sigma = np.median(np.abs(hh)) / 0.6745
    sigma = max(float(sigma), 1e-6)
    return sigma


def denoise_channel(channel):

    nan_mask = np.isnan(channel)

    work = channel.copy()
    work[nan_mask] = 0.0

    valid = work[~nan_mask]

    if valid.size == 0:
        return channel.copy()

    sigma = estimate_sigma_mad(work)

    print(f"Estimated sigma: {sigma:.6f}")

    offset = 0.0

    mn = work.min()

    if mn < 0:
        offset = -mn
        work = work + offset

    mx = work.max()

    if mx <= 0:
        out = work - offset
        out[nan_mask] = np.nan
        return out.astype(np.float32)

    scale = mx

    normalized = work / scale

    sigma_norm = sigma / scale

    denoised = bm3d(normalized, sigma_psd=sigma_norm)

    denoised *= scale

    denoised -= offset

    denoised[nan_mask] = np.nan

    return denoised.astype(np.float32)


def process_image(data):

    out = np.empty_like(data, dtype=np.float32)

    for c in range(3):
        print(f"Processing channel {c}")
        out[c] = denoise_channel(data[c])

    return out


def process_file(path, output_dir):

    print()
    print("#" * 70)
    print(path.name)
    print("#" * 70)

    data, header = load_fits_rgb(path)

    print_stats("INPUT", data)

    result = process_image(data)

    print_stats("OUTPUT", result)

    header["DENOISE"] = (
        "BM3D+MAD",
        "Pseudo Ground Truth generated with BM3D"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / path.name

    fits.PrimaryHDU(
        result.astype(np.float32),
        header=header
    ).writeto(out_path, overwrite=True)

    print("Saved:", out_path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        default="assets/inputs"
    )

    parser.add_argument(
        "--output-dir",
        default="assets/outputs-denoised-clean"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    files = sorted(
        list(input_dir.glob("*.fits")) +
        list(input_dir.glob("*.fit")) +
        list(input_dir.glob("*.FITS"))
    )

    print(f"Found {len(files)} FITS files.")

    for f in files:
        process_file(f, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
