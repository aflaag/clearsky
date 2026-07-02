import argparse
from pathlib import Path
from astropy.io import fits
import numpy as np


def load_fits_data(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()
    return data, header


def inject_poisson_gaussian_noise(
    data,
    peak_photons_range=(800, 30000),
    read_noise_range=(2.0, 8.0),
    white_percentile=99.7,
    rng=None,
):
    """
    Inietta rumore Poisson-Gaussian su dati lineari puliti.

    Non assumiamo un gain fisico assoluto (i FITS HLA non sono ADU raw), quindi
    normalizziamo l'immagine al proprio white point e la scaliamo a un budget di
    fotoni plausibile, randomizzato per immagine (domain randomization: il
    modello vede una varietà di livelli di rumore, non uno solo fisso).

    peak_photons_range: budget di fotoni al white point. Valori bassi = più
    rumoroso (es. sub-esposizione corta/poco segnale), valori alti = più pulito.
    read_noise_range: rumore di lettura gaussiano additivo, in "fotoni equivalenti".
    """
    if rng is None:
        rng = np.random.default_rng()

    clean = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    clean = np.clip(clean, 0.0, None)

    white = np.percentile(clean, white_percentile)
    if white <= 0:
        white = clean.max() if clean.max() > 0 else 1.0

    peak_photons = rng.uniform(*peak_photons_range)
    scale = peak_photons / white

    photons = clean * scale
    noisy_photons = rng.poisson(np.clip(photons, 0, None)).astype(np.float32)

    read_noise = rng.uniform(*read_noise_range)
    noisy_photons += rng.normal(0.0, read_noise, size=clean.shape).astype(np.float32)

    noisy = np.clip(noisy_photons / scale, 0.0, None)

    return noisy.astype(np.float32), {
        "peak_photons": float(peak_photons),
        "read_noise": float(read_noise),
        "scale": float(scale),
    }


def process_file(fits_path, output_dir, peak_photons_range, read_noise_range, rng):
    basename = fits_path.stem
    output_path = output_dir / f"{basename}.fits"

    if output_path.exists():
        print(f"[SKIP] {basename}: già processato")
        return

    data, header = load_fits_data(fits_path)
    if data is None:
        print(f"[SKIP] {basename}: dati mancanti")
        return

    noisy, params = inject_poisson_gaussian_noise(
        data, peak_photons_range=peak_photons_range, read_noise_range=read_noise_range, rng=rng
    )

    header["NOISESIM"] = (True, "Rumore sintetico Poisson-Gaussian iniettato")
    header["SIMPEAK"] = (round(params["peak_photons"], 1), "budget fotoni al white point")
    header["SIMRDN"] = (round(params["read_noise"], 3), "read noise sintetico")

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".fits.tmp")
    fits.PrimaryHDU(data=noisy, header=header).writeto(tmp_path, overwrite=True)
    tmp_path.rename(output_path)  # scrittura atomica: niente file corrotti a metà
    print(f"[OK] {basename} (peak={params['peak_photons']:.0f}ph, read_noise={params['read_noise']:.2f})")


def main():
    parser = argparse.ArgumentParser(description="Inietta rumore sintetico Poisson-Gaussian su FITS lineari puliti.")
    parser.add_argument("--input-dir", default="assets/inputs")
    parser.add_argument("--output-dir", default="assets/outputs-noisy")
    parser.add_argument("--peak-min", type=float, default=800)
    parser.add_argument("--peak-max", type=float, default=30000)
    parser.add_argument("--read-noise-min", type=float, default=2.0)
    parser.add_argument("--read-noise-max", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    fits_files = sorted(
        list(input_dir.glob("*.fits")) + list(input_dir.glob("*.fit")) + list(input_dir.glob("*.FITS"))
    )
    print(f"Trovati {len(fits_files)} file FITS dentro {input_dir}.")

    for f in fits_files:
        process_file(
            f, output_dir,
            peak_photons_range=(args.peak_min, args.peak_max),
            read_noise_range=(args.read_noise_min, args.read_noise_max),
            rng=rng,
        )

    print("Completato.")


if __name__ == "__main__":
    main()
