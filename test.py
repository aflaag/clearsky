"""
test_variance_mask.py

Testa la maschera basata su varianza locale su tutti i FITS,
salvando PNG di debug per trovare la soglia giusta.

Per ogni file salva 3 PNG con soglie diverse cosi puoi scegliere
visivamente quale funziona meglio.
"""

import numpy as np
from astropy.io import fits
from pathlib import Path
from PIL import Image
from scipy.ndimage import uniform_filter

INPUT_DIR = Path("assets/inputs")
DEBUG_DIR = Path("debug_variance")
DEBUG_DIR.mkdir(exist_ok=True)

# Soglie da testare
THRESHOLDS = [1e-5, 1e-4, 1e-3]
WINDOW = 16  # pixel, dimensione finestra varianza locale

fits_files = sorted(
    list(INPUT_DIR.glob("*.fits"))
    + list(INPUT_DIR.glob("*.fit"))
    + list(INPUT_DIR.glob("*.FITS"))
)

print(f"Trovati {len(fits_files)} file FITS | window={WINDOW}px\n")

for fits_path in fits_files:
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float32)

    if data.shape[-1] == 3:
        data = np.transpose(data, (2, 0, 1))

    C, H, W = data.shape

    # Usa canale verde (indice 1) per la varianza
    gray = data[1]

    # Gestisci NaN
    gray = np.nan_to_num(gray, nan=0.0)

    # Varianza locale tramite integral image del quadrato
    mean   = uniform_filter(gray,      size=WINDOW)
    mean_sq = uniform_filter(gray**2,  size=WINDOW)
    variance = np.clip(mean_sq - mean**2, 0, None)

    print(f"{fits_path.name} | Shape: {H}x{W}")
    print(f"  Variance: min={variance.min():.2e} max={variance.max():.2e} "
          f"p1={np.percentile(variance,1):.2e} p5={np.percentile(variance,5):.2e} "
          f"p50={np.percentile(variance,50):.2e}")

    # Preview in scala di grigi (stretch semplice per il debug)
    vmin, vmax = np.percentile(gray[gray > 0], [1, 99.5]) if (gray > 0).any() else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    gray_norm = np.clip((gray - vmin) / (vmax - vmin), 0, 1)

    # Scala a max 800px
    scale = min(800 / H, 800 / W, 1.0)
    th = int(H * scale)
    tw = int(W * scale)

    gray_8bit = (gray_norm * 255).astype(np.uint8)
    gray_img = Image.fromarray(gray_8bit).resize((tw, th), Image.BILINEAR)

    for thresh in THRESHOLDS:
        valid = variance > thresh
        pct = valid.mean() * 100

        print(f"  thresh={thresh:.0e} -> {pct:.1f}% validi")

        # Overlay rosso dove INVALIDO
        rgb_img = gray_img.convert("RGB")
        invalid_mask = Image.fromarray(((~valid).astype(np.uint8) * 255)).resize(
            (tw, th), Image.NEAREST
        )
        red = Image.new("RGB", (tw, th), (220, 50, 50))
        rgb_img.paste(red, mask=invalid_mask)

        out_path = DEBUG_DIR / f"{fits_path.stem}_thresh{thresh:.0e}.png"
        rgb_img.save(out_path)

    print()

print("Completato. Controlla i PNG in debug_variance/")
