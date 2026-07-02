import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits
from skimage.restoration import (
    denoise_wavelet,
    denoise_nl_means,
    estimate_sigma,
)


def load_fits_rgb(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

    if data is None:
        raise ValueError("dati mancanti")

    if data.ndim != 3:
        raise ValueError(f"shape non supportata: {data.shape}")

    # accetta sia (3,H,W) che (H,W,3)
    if data.shape[-1] == 3:
        data = np.transpose(data, (2, 0, 1))

    if data.shape[0] != 3:
        raise ValueError("servono esattamente 3 canali RGB")

    return data.astype(np.float32), header


def print_stats(name, img):
    print(f"\n{name}")
    print(f"dtype : {img.dtype}")
    print(f"shape : {img.shape}")
    print(f"NaN   : {np.isnan(img).sum()}")
    print(f"Inf   : {np.isinf(img).sum()}")

    valid = np.isfinite(img)

    if np.any(valid):
        values = img[valid]

        print(f"min   : {values.min()}")
        print(f"max   : {values.max()}")
        print(f"mean  : {values.mean()}")
        print(f"std   : {values.std()}")

    else:
        print("nessun pixel valido")


def denoise_channel(channel, method):

    mask_nan = np.isnan(channel)

    work = np.nan_to_num(
        channel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)

    print_stats("Input canale", work)

    valid_pixels = work[~mask_nan]

    if valid_pixels.size == 0:
        return channel.copy()

    sigma = estimate_sigma(valid_pixels, channel_axis=None)

    print(f"\nSigma stimato: {sigma}")

    if method == "wavelet":

        denoised = denoise_wavelet(
            work,
            sigma=sigma,
            wavelet="db2",
            mode="soft",
            method="BayesShrink",
            rescale_sigma=True,
            channel_axis=None,
        )

    elif method == "nlmeans":

        denoised = denoise_nl_means(
            work,
            h=0.9 * sigma,
            fast_mode=True,
            patch_size=5,
            patch_distance=6,
            channel_axis=None,
        )

    else:
        raise ValueError(method)

    denoised = denoised.astype(np.float32)

    denoised[mask_nan] = np.nan

    print_stats("Output canale", denoised)

    return denoised


def denoise_fits(data, method):

    output = np.empty_like(data, dtype=np.float32)

    for c in range(3):
        print("\n====================================")
        print(f"Canale {c}")
        print("====================================")
        output[c] = denoise_channel(data[c], method)

    return output


def process_file(fits_path, output_dir, method):

    basename = fits_path.stem
    output_path = output_dir / f"{basename}.fits"

    if output_path.exists():
        print(f"[SKIP] {basename}")
        return

    try:
        data, header = load_fits_rgb(fits_path)

    except ValueError as e:
        print(f"[SKIP] {basename}: {e}")
        return

    print("\n=================================================")
    print(f"FILE: {basename}")
    print("=================================================")

    print_stats("INPUT", data)

    denoised = denoise_fits(data, method)

    print_stats("OUTPUT", denoised)

    header["DENOISE"] = (
        f"skimage-{method}",
        "Pseudo ground truth generated with scikit-image",
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    fits.PrimaryHDU(
        data=denoised,
        header=header,
    ).writeto(output_path, overwrite=True)

    print(f"\n[OK] Salvato: {output_path}")


def main():

    parser = argparse.ArgumentParser(
        description="Pseudo ground truth generator"
    )

    parser.add_argument(
        "--input-dir",
        default="assets/inputs",
    )

    parser.add_argument(
        "--output-dir",
        default="assets/outputs-denoised-clean",
    )

    parser.add_argument(
        "--method",
        choices=["wavelet", "nlmeans"],
        default="wavelet",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    fits_files = sorted(
        list(input_dir.glob("*.fits")) +
        list(input_dir.glob("*.fit")) +
        list(input_dir.glob("*.FITS"))
    )

    print(f"Trovati {len(fits_files)} FITS.\n")

    for file in fits_files:
        process_file(file, output_dir, args.method)

    print("\nCompletato.")


if __name__ == "__main__":
    main()
