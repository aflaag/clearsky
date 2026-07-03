import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom


def load_tiff_normalized(tiff_path):
    """Carica un TIFF (16-bit o 8-bit) e normalizza in [0, 1]."""
    arr = tifffile.imread(tiff_path)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Dtype inatteso per TIFF: {arr.dtype}")


def match_shape(arr, target_shape):
    """Forza arr alla stessa (H, W, C) di target_shape via crop/pad di sicurezza.

    zoom() con fattori non esattamente reciproci puo' produrre uno shape
    di 1-2 px diverso dall'originale per via degli arrotondamenti.
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
    """Modello di degradazione classico per SR: y = (x * blur)-downsample-upsample.

    Il blur simula l'allargamento della PSF, il downsample la perdita di
    risoluzione, l'upsample bicubico finale riporta l'immagine alla shape
    dell'HR cosi' da poterla usare come canale di conditioning per il DDPM
    (stessa logica dell'immagine "con stelle" nel modello di star removal).
    """
    if blur_sigma > 0:
        # sigma solo sulle dimensioni spaziali, non sui canali colore
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

    print(f"  HR: {hr.shape} -> LR x{scale} -> degradata (upsampled): {degraded.shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera versioni degradate (blur + downsample + upsample) delle immagini HR intere, per il dataset di super-resolution."
    )
    parser.add_argument(
        "--tiff-dir",
        default="assets/outputs-tiff",
        help="TIFF HR (output di astro_stretch.py --save-tiff)",
    )
    parser.add_argument(
        "--out-dir",
        default="assets/outputs-degraded",
        help="Cartella output (default: assets/outputs-degraded)",
    )
    parser.add_argument(
        "--scale", type=int, default=4, help="Fattore di downsampling (default: 4)"
    )
    parser.add_argument(
        "--blur-sigma",
        type=float,
        default=1.5,
        help="Sigma del blur gaussiano pre-downsample, 0 per disattivare (default: 1.5)",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Salva anche PNG di preview delle immagini degradate (per ispezione visiva)",
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
        print(f"Nessun TIFF trovato in {tiff_dir}")
        return

    print(f"Trovati {len(tiff_files)} TIFF HR")
    print(f"Scale: x{args.scale} | Blur sigma: {args.blur_sigma} | Save PNG: {args.save_png}\n")

    for tiff_path in tiff_files:
        process_file(tiff_path, out_npy_dir, out_png_dir, args.scale, args.blur_sigma)

    print("\nCompletato.")


if __name__ == "__main__":
    main()
