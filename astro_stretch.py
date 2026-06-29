import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
# Usiamo tifffile al posto di PIL per il supporto nativo a RGB 16-bit
import tifffile


def adaptive_arcsinh_stretch(
    data_array,
    stretch=8.0,
    black_percentile=1.0,
    white_percentile=99.7,
    linked=False,
):
    """Stretch arcsinh percentile-based."""
    clean = np.nan_to_num(
        data_array, nan=0.0, posinf=0.0, neginf=0.0
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
    output_tiff_dir,
    linked,
    save_tiff,
):
    print(f"Processing {fits_path}")
    basename = fits_path.stem

    npy_path = output_npy_dir / f"{basename}.npy"
    tiff_path = output_tiff_dir / f"{basename}.tif" if save_tiff else None

    # ----------------------------
    # SKIP LOGIC
    # ----------------------------
    already_done = npy_path.exists() and (not save_tiff or tiff_path.exists())
    if already_done:
        print(f"[SKIP] {fits_path.name}: già processato")
        return

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
        # SAVE .npy (Float32 in [0, 1])
        # ----------------------------
        np.save(npy_path, rgb.astype(np.float32))

        # ----------------------------
        # FIX QUANTIZZAZIONE: SAVE TIFF 16-bit
        # ----------------------------
        if save_tiff:
            output_tiff_dir.mkdir(parents=True, exist_ok=True)
            # Scaliamo a 16-bit (0 - 65535) senza perdere precisione
            tiff_data = np.clip(rgb * 65535.0, 0, 65535).astype(np.uint16)
            tifffile.imwrite(tiff_path, tiff_data)
            print(f"[OK] {fits_path.name} (npy + tiff 16-bit)")
        else:
            print(f"[OK] {fits_path.name} (npy)")

    except Exception as e:
        print(f"[ERROR] {fits_path.name}")
        print(e)


def main():
    parser = argparse.ArgumentParser(
        description="Batch processing FITS con stretch arcsinh adattivo."
    )
    parser.add_argument(
        "--input-dir", default="assets/inputs", help="Cartella contenente i FITS"
    )
    parser.add_argument(
        "--output-npy", default="assets/outputs-npy", help="Cartella output file .npy"
    )
    parser.add_argument(
        "--output-tiff",
        default="assets/outputs-tiff",
        help="Cartella output TIFF (usata solo con --save-tiff)",
    )
    parser.add_argument(
        "--linked",
        action="store_true",
        help="Usa lo stesso stretch per tutti i canali",
    )
    parser.add_argument(
        "--save-tiff",
        action="store_true",
        help="Salva il TIFF stretched a 16-bit per StarNet2",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_npy_dir = Path(args.output_npy)
    output_tiff_dir = Path(args.output_tiff)

    output_npy_dir.mkdir(parents=True, exist_ok=True)

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
            output_tiff_dir,
            linked=args.linked,
            save_tiff=args.save_tiff,
        )

    print("Completato.")


if __name__ == "__main__":
    main()
