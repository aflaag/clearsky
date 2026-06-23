from astropy.io import fits
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def siril_mtf_transformation(data_channel, shadows, midtones, highlights=1.0):
    xp = (data_channel - shadows) / (highlights - shadows)
    xp = np.clip(xp, 0.0, 1.0)

    numerator = (midtones - 1.0) * xp
    denominator = ((2.0 * midtones - 1.0) * xp) - midtones

    with np.errstate(divide='ignore', invalid='ignore'):
        stretched = np.where(
            denominator != 0,
            numerator / denominator,
            0.0
        )

    return np.clip(stretched, 0.0, 1.0)


def siril_autostretch(data_array,
                      shadowsclip=-0.2,
                      targetbg=0.25,
                      linked=False):

    num_channels, H, W = data_array.shape
    stretched_image = np.zeros_like(data_array, dtype=np.float32)

    clean_data = np.nan_to_num(data_array)

    if linked:
        median_global = np.median(clean_data)
        std_global = np.std(clean_data)

        shadows = median_global + (shadowsclip * std_global)

        xp_bg = (median_global - shadows) / (1.0 - shadows)
        xp_bg = np.clip(xp_bg, 0.001, 0.999)

        midtones = (
            xp_bg * (1.0 - targetbg)
        ) / (
            xp_bg + targetbg - (2.0 * targetbg * xp_bg)
        )

        for c in range(num_channels):
            stretched_image[c] = siril_mtf_transformation(
                clean_data[c],
                shadows,
                midtones
            )

    else:
        for c in range(num_channels):
            channel = clean_data[c]

            median = np.median(channel)
            std = np.std(channel)

            shadows = median + (shadowsclip * std)

            xp_bg = (median - shadows) / (1.0 - shadows)
            xp_bg = np.clip(xp_bg, 0.001, 0.999)

            midtones = (
                xp_bg * (1.0 - targetbg)
            ) / (
                xp_bg + targetbg - (2.0 * targetbg * xp_bg)
            )

            stretched_image[c] = siril_mtf_transformation(
                channel,
                shadows,
                midtones
            )

    return stretched_image


def process_fits_file(
    fits_path,
    output_npy_dir,
    output_png_dir,
    shadowsclip,
    targetbg,
    linked
):
    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data

        if data is None:
            print(f"[SKIP] {fits_path.name}: dati mancanti")
            return

        # atteso: (3, H, W)
        if data.ndim != 3 or data.shape[0] != 3:
            print(
                f"[SKIP] {fits_path.name}: shape {data.shape} non supportata"
            )
            return

        data = data.astype(np.float32)

        data_stretched = siril_autostretch(
            data,
            shadowsclip=shadowsclip,
            targetbg=targetbg,
            linked=linked
        )

        # (H, W, 3)
        data_rgb = np.transpose(data_stretched, (1, 2, 0))

        # Rimuove eventuale alpha se presente
        if data_rgb.shape[-1] > 3:
            data_rgb = data_rgb[..., :3]

        basename = fits_path.stem

        # -------------------------
        # Salvataggio .npy
        # -------------------------
        npy_path = output_npy_dir / f"{basename}.npy"

        # float32 in [0,1], ideale per DL
        np.save(npy_path, data_rgb.astype(np.float32))

        # -------------------------
        # Salvataggio PNG RGB
        # -------------------------
        png = (data_rgb * 255.0)
        png = np.clip(png, 0, 255).astype(np.uint8)

        image = Image.fromarray(png, mode="RGB")

        png_path = output_png_dir / f"{basename}.png"
        image.save(png_path)

        print(f"[OK] {fits_path.name}")

    except Exception as e:
        print(f"[ERROR] {fits_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Cartella contenente i FITS"
    )

    parser.add_argument(
        "--output-npy",
        required=True,
        help="Cartella di output per gli array .npy"
    )

    parser.add_argument(
        "--output-png",
        required=True,
        help="Cartella di output per le PNG"
    )

    parser.add_argument(
        "--shadowsclip",
        type=float,
        default=-0.2
    )

    parser.add_argument(
        "--targetbg",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--linked",
        action="store_true"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_npy_dir = Path(args.output_npy)
    output_png_dir = Path(args.output_png)

    output_npy_dir.mkdir(parents=True, exist_ok=True)
    output_png_dir.mkdir(parents=True, exist_ok=True)

    fits_files = sorted(
        list(input_dir.glob("*.fits")) +
        list(input_dir.glob("*.fit")) +
        list(input_dir.glob("*.FITS"))
    )

    print(f"Trovati {len(fits_files)} file FITS")

    for fits_file in fits_files:
        process_fits_file(
            fits_file,
            output_npy_dir,
            output_png_dir,
            args.shadowsclip,
            args.targetbg,
            args.linked
        )

if __name__ == "__main__":
    main()
