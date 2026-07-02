import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np
import tifffile


def compute_stretch_params(data_array, black_percentile=1.0, white_percentile=99.7, linked=False):
    """Calcola black/white (per canale, o unico se linked) da un array (C,H,W)."""
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
    """Applica lo stretch arcsinh usando parametri black/white GIÀ CALCOLATI (fissi)."""
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
    """Wrapper retrocompatibile: calcola i parametri e applica lo stretch in un solo step.
    Ora ritorna (stretched, params) invece del solo array, per permettere il riuso dei
    parametri su un'immagine accoppiata (es. output GraXpert)."""
    params = compute_stretch_params(data_array, black_percentile, white_percentile, linked)
    stretched = apply_arcsinh_stretch(data_array, params, stretch=stretch)
    return stretched, params


def load_fits_rgb(fits_path):
    """Carica un FITS e lo normalizza in (3,H,W) float32. Solleva ValueError se non valido."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data

    if data is None:
        raise ValueError("dati mancanti")
    if data.ndim != 3:
        raise ValueError(f"shape {data.shape}")
    if data.shape[-1] == 3:
        data = np.transpose(data, (2, 0, 1))
    if data.shape[0] != 3:
        raise ValueError("servono 3 canali RGB")

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
    Processa un FITS applicando lo stretch arcsinh.
    Se paired_fits_path è fornito (es. l'output GraXpert dello stesso target), i parametri
    di stretch vengono calcolati SOLO sull'immagine principale (rumorosa) e riapplicati
    IDENTICI all'immagine accoppiata, per evitare mismatch di background/scala tra
    input e target — lo stesso tipo di problema già visto con StarNet2.
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
        print(f"[SKIP] {fits_path.name}: già processato")
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
    print(f"[OK] {fits_path.name} (npy{' + tiff 16-bit' if save_tiff else ''})")

    if paired_fits_path is None:
        return

    if not paired_fits_path.exists():
        print(f"[WARN] {fits_path.name}: manca il corrispondente {paired_fits_path.name}, salto il paired output")
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

    # Riuso ESATTO dei parametri calcolati sull'immagine rumorosa
    paired_stretched = apply_arcsinh_stretch(paired_data, params)
    save_outputs(paired_stretched, basename, paired_output_npy_dir, paired_output_tiff_dir, save_tiff)
    print(f"[OK] {paired_fits_path.name} (paired, stretch riusato da {fits_path.name})")


def main():
    parser = argparse.ArgumentParser(description="Batch processing FITS con stretch arcsinh adattivo.")
    parser.add_argument("--input-dir", default="assets/inputs", help="Cartella contenente i FITS rumorosi")
    parser.add_argument("--output-npy", default="assets/outputs-npy", help="Cartella output .npy")
    parser.add_argument("--output-tiff", default="assets/outputs-tiff", help="Cartella output TIFF (con --save-tiff)")
    parser.add_argument("--linked", action="store_true", help="Usa lo stesso stretch per tutti i canali")
    parser.add_argument("--save-tiff", action="store_true", help="Salva anche il TIFF stretched a 16-bit")

    parser.add_argument(
        "--paired-dir",
        default=None,
        help="Cartella con i FITS accoppiati (es. output GraXpert), stesso basename di --input-dir. "
        "Se fornita, lo stretch viene calcolato SOLO sul FITS rumoroso e riapplicato identico al paired.",
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
    print(f"Trovati {len(fits_files)} file FITS dentro {input_dir}.")

    for fits_file in fits_files:
        paired_fits_path = None
        if paired_dir:
            candidates = [paired_dir / f"{fits_file.stem}{ext}" for ext in (".fits", ".fit", ".FITS")]
            paired_fits_path = next((c for c in candidates if c.exists()), None)
            if paired_fits_path is None:
                print(f"[WARN] {fits_file.name}: nessun corrispondente in {paired_dir}, salto il paired output")

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

    print("Completato.")


if __name__ == "__main__":
    main()
