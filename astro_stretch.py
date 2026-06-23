from astropy.io import fits
import numpy as np
from pathlib import Path
from PIL import Image
import argparse

def adaptive_arcsinh_stretch(
    data_array,
    stretch=8.0,
    black_percentile=1.0,
    white_percentile=99.7,
    linked=False,
):
    """
    Stretch arcsinh percentile-based.

    Parameters
    ----------
    stretch : float
        Intensità dello stretch.
        5-8 -> conservativo
        8-12 -> più aggressivo

    black_percentile : float
        Percentile del punto di nero.

    white_percentile : float
        Percentile del punto di bianco.

    linked : bool
        Se True usa gli stessi percentili per tutti i canali.
    """

    clean = np.nan_to_num(
        data_array,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    ).astype(np.float32)

    stretched = np.zeros_like(clean, dtype=np.float32)

    if linked:
        black = np.percentile(clean, black_percentile)
        white = np.percentile(clean, white_percentile)

        if white <= black:
            white = black + 1e-10

        for c in range(clean.shape[0]):
            channel = clean[c]

            x = (channel - black) / (white - black)
            x = np.clip(x, 0.0, 1.0)

            y = np.arcsinh(stretch * x)
            y /= np.arcsinh(stretch)

            stretched[c] = np.clip(y, 0.0, 1.0)

    else:
        for c in range(clean.shape[0]):
            channel = clean[c]

            black = np.percentile(channel, black_percentile)
            white = np.percentile(channel, white_percentile)

            if white <= black:
                white = black + 1e-10

            x = (channel - black) / (white - black)
            x = np.clip(x, 0.0, 1.0)

            y = np.arcsinh(stretch * x)
            y /= np.arcsinh(stretch)

            stretched[c] = np.clip(y, 0.0, 1.0)

    return stretched

def process_fits_file(
    fits_path,
    output_npy_dir,
    output_png_dir,
    linked,
):
    print(f"Processing {fits_path}")

    basename = fits_path.stem

    npy_path = output_npy_dir / f"{basename}.npy"
    png_path = output_png_dir / f"{basename}.png"

    # ----------------------------
    # SKIP LOGIC
    # ----------------------------
    # if npy_path.exists() and png_path.exists():
    #     print(f"[SKIP] {fits_path.name}: già processato")
    #     return

    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data

        if data is None:
            print(f"[SKIP] {fits_path.name}: dati mancanti")
            return

        if data.ndim != 3:
            print(f"[SKIP] {fits_path.name}: shape {data.shape}")
            return

        # Caso (H,W,3) -> (3,H,W)
        if data.shape[-1] == 3:
            data = np.transpose(data, (2, 0, 1))

        if data.shape[0] != 3:
            print(f"[SKIP] {fits_path.name}: servono 3 canali RGB")
            return

        data = data.astype(np.float32)

        stretched = adaptive_arcsinh_stretch(
            data,
            linked=linked,
        )

        rgb = np.transpose(stretched, (1, 2, 0))
        rgb = rgb[..., :3]

        # ----------------------------
        # SAVE .npy
        # ----------------------------
        np.save(
            npy_path,
            rgb.astype(np.float32)
        )

        # ----------------------------
        # SAVE PNG
        # ----------------------------
        png = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

        Image.fromarray(png, mode="RGB").save(png_path)

        print(f"[OK] {fits_path.name}")

    except Exception as e:
        print(f"[ERROR] {fits_path.name}")
        print(e)

def main():
    parser = argparse.ArgumentParser(
        description="Batch processing FITS con stretch arcsinh adattivo."
    )

    parser.add_argument(
        "--input-dir",
        default="assets/inputs",
        help="Cartella contenente i FITS"
    )

    parser.add_argument(
        "--output-npy",
        default="assets/outputs-npy",
        help="Cartella output file .npy"
    )

    parser.add_argument(
        "--output-png",
        default="assets/outputs-png",
        help="Cartella output PNG"
    )

    parser.add_argument(
        "--sigma-factor",
        type=float,
        default=4.0,
        help="Intensità dello stretch"
    )

    parser.add_argument(
        "--linked",
        action="store_true",
        help="Usa lo stesso stretch per tutti i canali"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_npy_dir = Path(args.output_npy)
    output_png_dir = Path(args.output_png)

    output_npy_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_png_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    fits_files = (
        list(input_dir.glob("*.fits"))
        + list(input_dir.glob("*.fit"))
        + list(input_dir.glob("*.FITS"))
    )

    fits_files = sorted(fits_files)

    print(f"Trovati {len(fits_files)} file FITS dentro {input_dir}.")

    for fits_file in fits_files:
        process_fits_file(
            fits_file,
            output_npy_dir,
            output_png_dir,
            linked=args.linked,
        )

    print("Completato.")


if __name__ == "__main__":
    main()
