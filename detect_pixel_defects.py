"""
detect_pixel_defects.py

Individua dead/hot pixel nelle immagini stretchate (output di astro_stretch.py)
confrontando ogni pixel con un filtro mediano locale, e produce:
  - <output-dir>/<basename>.npy                 (immagine corretta, target per training)
  - <tiff-dir>/<basename>.tiff                   (TIFF corretto, solo con --save-tiff)
  - <mask-tiff-dir>/<basename>.tiff              (maschera dei pixel corretti, solo con --save-mask-tiff)

Un pixel è considerato hot/dead se il residuo rispetto alla mediana locale
(finestra --window) supera --sigma volte la deviazione standard robusta
(MAD, Median Absolute Deviation) del residuo, stimata solo sulla regione
valida (maschera di make_masks.py, se disponibile in --mask-dir).

Il rilevamento e la correzione sono per canale (un difetto può comparire
su un solo canale, essendo HLA un composito RGB da esposizioni diverse);
la maschera finale è l'unione booleana sui canali.

Uso:
    python detect_pixel_defects.py
    python detect_pixel_defects.py --input-dir assets/outputs-stretch --output-dir assets/outputs-pixelfix
    python detect_pixel_defects.py --sigma 5.0 --window 7 --save-tiff --save-mask-tiff
"""

import numpy as np
from pathlib import Path
from scipy.ndimage import median_filter, binary_dilation
import argparse

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def to_chw(data):
    """
    Normalizza un array a (C, H, W).
    Ritorna (data_chw, was_hwc) dove was_hwc indica se l'input era (H, W, C),
    per poter ripristinare l'orientamento originale in output.
    """
    if data.ndim == 2:
        return data[None, ...], False
    if data.shape[0] in (1, 3, 4):
        return data, False
    if data.shape[-1] in (1, 3, 4):
        return np.transpose(data, (2, 0, 1)), True
    raise ValueError(f"Shape non riconosciuta: {data.shape}")


def from_chw(data_chw, was_hwc):
    """Inverso di to_chw: ripristina l'orientamento originale."""
    if was_hwc:
        return np.transpose(data_chw, (1, 2, 0))
    if data_chw.shape[0] == 1:
        return data_chw[0]
    return data_chw


def robust_sigma(residual, valid=None):
    """Stima robusta di sigma tramite MAD (Median Absolute Deviation)."""
    vals = residual[valid] if valid is not None else residual.ravel()
    if vals.size == 0:
        return 0.0
    med = np.median(vals)
    mad = np.median(np.abs(vals - med))
    return 1.4826 * mad  # fattore di conversione MAD -> sigma gaussiana


def detect_bad_pixels(data_chw, valid_mask=None, window=5, sigma_thresh=6.0, dilate=0):
    """
    Individua dead/hot pixel per canale.

    data_chw    : (C, H, W) float32
    valid_mask  : (H, W) bool opzionale, regione su cui stimare la sigma
                  e restringere il rilevamento (es. maschera di make_masks.py)
    window      : dimensione finestra filtro mediano locale
    sigma_thresh: soglia in sigma robusti (MAD) per flag hot/dead
    dilate      : iterazioni di dilatazione binaria sulla maschera risultante

    Ritorna:
        bad_mask   : (H, W) bool, unione dei canali
        bad_per_ch : (C, H, W) bool, maschera per canale
        medians    : (C, H, W) float32, valori mediani locali (riusati per la correzione)
    """
    C, H, W = data_chw.shape
    bad_per_ch = np.zeros((C, H, W), dtype=bool)
    medians = np.zeros_like(data_chw)

    for c in range(C):
        ch = data_chw[c]
        med = median_filter(ch, size=window)
        medians[c] = med
        residual = ch - med

        sigma = robust_sigma(residual, valid_mask)
        if sigma <= 0:
            # canale piatto o sigma non stimabile: nessun difetto rilevabile
            continue

        hot = residual > sigma_thresh * sigma
        dead = residual < -sigma_thresh * sigma
        bad = hot | dead

        if valid_mask is not None:
            bad &= valid_mask  # ignora bordi/padding già invalidi

        if dilate > 0:
            bad = binary_dilation(bad, iterations=dilate)

        bad_per_ch[c] = bad

    bad_mask = bad_per_ch.any(axis=0)
    return bad_mask, bad_per_ch, medians


def correct_pixels(data_chw, bad_per_ch, medians):
    """
    Sostituisce i pixel individuati con il valore del filtro mediano locale
    già calcolato in detect_bad_pixels (nessun ricalcolo).
    """
    corrected = data_chw.copy()
    for c in range(data_chw.shape[0]):
        m = bad_per_ch[c]
        corrected[c][m] = medians[c][m]
    return corrected

def save_as_tiff(image, path):
    """Salva un'immagine come TIFF RGB uint16 compatibile con StarNet2."""
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile non installato: pip install tifffile")

    img = image

    # Converti da CHW a HWC se necessario
    if img.ndim == 3 and img.shape[0] in (1, 3, 4) and img.shape[0] != img.shape[-1]:
        img = np.transpose(img, (1, 2, 0))

    # Float [0,1] -> uint16 [0,65535]
    img = np.clip(img, 0.0, 1.0)
    img = np.round(img * 65535.0).astype(np.uint16)

    tifffile.imwrite(str(path), img)


def save_mask_as_tiff(bad_mask, path):
    """Salva la maschera booleana dei pixel corretti come TIFF 8-bit (0/255)."""
    if not HAS_TIFFFILE:
        raise RuntimeError("tifffile non installato: pip install tifffile")
    tifffile.imwrite(str(path), (bad_mask.astype(np.uint8) * 255))


def process_file(npy_path, mask_dir, out_dir, tiff_dir, mask_tiff_dir,
                  window, sigma_thresh, dilate, save_tiff, save_mask_tiff):
    basename = npy_path.stem
    out_npy = out_dir / f"{basename}.npy"

    already_done = (
        out_npy.exists()
        and (not save_tiff or (tiff_dir / f"{basename}.tiff").exists())
        and (not save_mask_tiff or (mask_tiff_dir / f"{basename}.tiff").exists())
    )
    if already_done:
        print(f"[SKIP] {npy_path.name}: già processato")
        return

    data = np.load(npy_path).astype(np.float32)
    data_chw, was_hwc = to_chw(data)

    valid_mask = None
    if mask_dir is not None:
        mask_path = mask_dir / f"{basename}.npy"
        if mask_path.exists():
            valid_mask = np.load(mask_path)
        else:
            print(f"  [WARN] maschera di validità non trovata per {basename}: "
                  f"la stima della sigma userà l'intera immagine (incluso eventuale padding)")

    bad_mask, bad_per_ch, medians = detect_bad_pixels(
        data_chw, valid_mask=valid_mask, window=window,
        sigma_thresh=sigma_thresh, dilate=dilate,
    )

    n_bad = int(bad_mask.sum())
    pct = 100 * n_bad / bad_mask.size
    print(f"Processing {npy_path.name}: {n_bad} pixel difettosi ({pct:.4f}%)")

    corrected_chw = correct_pixels(data_chw, bad_per_ch, medians)
    corrected = from_chw(corrected_chw, was_hwc)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_npy, corrected.astype(np.float32))

    if save_tiff:
        tiff_dir.mkdir(parents=True, exist_ok=True)
        save_as_tiff(corrected, tiff_dir / f"{basename}.tiff")

    if save_mask_tiff:
        mask_tiff_dir.mkdir(parents=True, exist_ok=True)
        save_mask_as_tiff(bad_mask, mask_tiff_dir / f"{basename}.tiff")

    print(f"  [OK] -> {out_npy.name}")


def main():
    parser = argparse.ArgumentParser(
        description="Individua e corregge dead/hot pixel nelle immagini stretchate (npy)."
    )
    parser.add_argument("--input-dir", default="assets/outputs-npy",
                         help="Cartella npy stretchati in input (output di astro_stretch.py)")
    parser.add_argument("--mask-dir", default="assets/outputs-mask",
                         help="Cartella maschere di validità (make_masks.py); usata per stima sigma e regione di ricerca")
    parser.add_argument("--output-dir", default="assets/outputs-pixelfix",
                         help="Cartella npy corretti (target)")
    parser.add_argument("--tiff-dir", default="assets/outputs-pixelfix/tiff",
                         help="Cartella TIFF corretti (con --save-tiff)")
    parser.add_argument("--mask-tiff-dir", default="assets/outputs-pixelfix/mask-preview",
                         help="Cartella TIFF maschera difetti (con --save-mask-tiff)")
    parser.add_argument("--window", type=int, default=5,
                         help="Finestra filtro mediano locale in pixel (default: 5)")
    parser.add_argument("--sigma", type=float, default=6.0,
                         help="Soglia in sigma robusti (MAD) per flag hot/dead (default: 6.0)")
    parser.add_argument("--dilate", type=int, default=0,
                         help="Iterazioni di dilatazione binaria sulla maschera (default: 0)")
    parser.add_argument("--no-valid-mask", action="store_true",
                         help="Ignora le maschere di validità anche se presenti")
    parser.add_argument("--save-tiff", action="store_true",
                         help="Salva anche il TIFF dell'immagine corretta")
    parser.add_argument("--save-mask-tiff", action="store_true",
                         help="Salva un TIFF con la maschera dei pixel corretti")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mask_dir = None if args.no_valid_mask else Path(args.mask_dir)
    out_dir = Path(args.output_dir)
    tiff_dir = Path(args.tiff_dir)
    mask_tiff_dir = Path(args.mask_tiff_dir)

    npy_files = sorted(input_dir.glob("*.npy"))
    if not npy_files:
        print(f"Nessun file npy trovato in {input_dir}")
        return

    print(f"Trovati {len(npy_files)} file npy")
    print(f"window={args.window}px | sigma={args.sigma} | dilate={args.dilate} | "
          f"save_tiff={args.save_tiff} | save_mask_tiff={args.save_mask_tiff}\n")

    for f in npy_files:
        try:
            process_file(f, mask_dir, out_dir, tiff_dir, mask_tiff_dir,
                         args.window, args.sigma, args.dilate,
                         args.save_tiff, args.save_mask_tiff)
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    print("\nCompletato.")


if __name__ == "__main__":
    main()
