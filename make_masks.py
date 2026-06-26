"""
make_masks.py

Calcola le maschere di validità dai FITS raw e le salva come:
  - assets/outputs-mask/<basename>.npy  (bool, per make_dataset.py)
  - assets/outputs-mask/preview/<basename>.png  (debug visivo, solo con --save-png)

La maschera è calcolata sui dati raw FITS prima dello stretch,
usando varianza locale con soglia 1e-05 sul canale verde.

Uso:
    python make_masks.py
    python make_masks.py --input-dir assets/inputs --mask-dir assets/outputs-mask
    python make_masks.py --threshold 1e-5 --window 16 --margin 0 --save-png
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from PIL import Image
from scipy.ndimage import uniform_filter
import argparse


def compute_mask(data, window, threshold, margin):
    """
    Calcola la maschera di validità (H, W) bool.

    data : np.ndarray (C, H, W) float32, dati raw prima dello stretch
    window : int, dimensione finestra per varianza locale
    threshold : float, soglia sulla varianza
    margin : int, pixel da escludere sul perimetro (per il bordo fringe)

    Restituisce maschera bool (H, W): True = pixel valido
    """
    C, H, W = data.shape

    # Usa il canale verde (indice 1) per la varianza
    gray = data[1].copy()
    gray = np.nan_to_num(gray, nan=0.0).astype(np.float32)

    mean    = uniform_filter(gray,      size=window)
    mean_sq = uniform_filter(gray ** 2, size=window)
    variance = np.clip(mean_sq - mean ** 2, 0, None)

    mask = variance > threshold

    # Margine perimetrale fisso (esclude il bordo fringe)
    if margin > 0:
        mask[:margin,  :] = False
        mask[-margin:, :] = False
        mask[:,  :margin] = False
        mask[:, -margin:] = False

    return mask


def save_debug_png(gray_raw, mask, out_path, max_size=1024):
    """
    Salva un PNG di debug: immagine in scala di grigi con
    overlay rosso dove la maschera è invalida.
    """
    H, W = gray_raw.shape

    # Stretch per la preview
    valid_pixels = gray_raw[mask] if mask.any() else gray_raw.ravel()
    if len(valid_pixels) == 0 or valid_pixels.max() == valid_pixels.min():
        gray_norm = np.zeros_like(gray_raw)
    else:
        vmin = np.percentile(valid_pixels, 1)
        vmax = np.percentile(valid_pixels, 99.5)
        if vmax <= vmin:
            vmax = vmin + 1e-6
        gray_norm = np.clip((gray_raw - vmin) / (vmax - vmin), 0, 1)

    # Scala a max_size
    scale = min(max_size / H, max_size / W, 1.0)
    th, tw = int(H * scale), int(W * scale)

    gray_8bit = (gray_norm * 255).astype(np.uint8)
    gray_img  = Image.fromarray(gray_8bit).resize((tw, th), Image.BILINEAR)
    rgb_img   = gray_img.convert("RGB")

    # Overlay rosso dove invalido
    invalid_small = Image.fromarray(((~mask).astype(np.uint8) * 255)).resize(
        (tw, th), Image.NEAREST
    )
    red = Image.new("RGB", (tw, th), (220, 50, 50))
    rgb_img.paste(red, mask=invalid_small)

    rgb_img.save(out_path)


def process_fits(fits_path, mask_npy_dir, mask_png_dir, window, threshold, margin, save_png):
    basename = fits_path.stem
    npy_out  = mask_npy_dir / f"{basename}.npy"
    png_out  = mask_png_dir / f"{basename}.png" if save_png else None

    already_done = npy_out.exists() and (not save_png or png_out.exists())
    if already_done:
        print(f"[SKIP] {fits_path.name}: già processato")
        return

    print(f"Processing {fits_path.name}")

    try:
        with fits.open(fits_path) as hdul:
            data = hdul[0].data.astype(np.float32)

        if data.shape[-1] == 3:
            data = np.transpose(data, (2, 0, 1))  # (H,W,3) -> (3,H,W)

        if data.shape[0] != 3:
            print(f"  [SKIP] shape inattesa: {data.shape}")
            return

        mask = compute_mask(data, window, threshold, margin)
        pct  = mask.mean() * 100
        print(f"  Validi: {pct:.1f}%  |  shape: {data.shape[1]}x{data.shape[2]}")

        # Salva maschera NPY
        np.save(npy_out, mask)

        # Salva debug PNG (opzionale)
        if save_png:
            mask_png_dir.mkdir(parents=True, exist_ok=True)
            gray_raw = np.nan_to_num(data[1], nan=0.0)
            save_debug_png(gray_raw, mask, png_out)
            print(f"  [OK] -> {npy_out.name}, {png_out.name}")
        else:
            print(f"  [OK] -> {npy_out.name}")

    except Exception as e:
        print(f"  [ERROR] {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Genera maschere di validità dai FITS raw."
    )
    parser.add_argument("--input-dir",  default="assets/inputs",       help="Cartella FITS")
    parser.add_argument("--mask-dir",   default="assets/outputs-mask",  help="Cartella output maschere")
    parser.add_argument("--threshold",  type=float, default=1e-5,       help="Soglia varianza (default: 1e-5)")
    parser.add_argument("--window",     type=int,   default=16,         help="Finestra varianza in pixel (default: 16)")
    parser.add_argument("--margin",     type=int,   default=0,          help="Margine perimetrale da escludere in pixel (default: 0)")
    parser.add_argument("--save-png",   action="store_true",            help="Salva PNG di debug con overlay maschera")
    args = parser.parse_args()

    input_dir    = Path(args.input_dir)
    mask_npy_dir = Path(args.mask_dir)
    mask_png_dir = Path(args.mask_dir) / "preview"

    mask_npy_dir.mkdir(parents=True, exist_ok=True)
    # mask_png_dir creata solo se necessario, dentro process_fits

    fits_files = sorted(
        list(input_dir.glob("*.fits"))
        + list(input_dir.glob("*.fit"))
        + list(input_dir.glob("*.FITS"))
    )

    if not fits_files:
        print(f"Nessun file FITS trovato in {input_dir}")
        return

    print(f"Trovati {len(fits_files)} file FITS")
    print(f"threshold={args.threshold:.0e} | window={args.window}px | margin={args.margin}px | save_png={args.save_png}\n")

    for f in fits_files:
        process_fits(f, mask_npy_dir, mask_png_dir, args.window, args.threshold, args.margin, args.save_png)

    print("\nCompletato.")


if __name__ == "__main__":
    main()
