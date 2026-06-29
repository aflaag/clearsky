import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile  # Sostituito PIL con tifffile per gestire i TIFF a 16-bit


def load_starless(path):
    """Carica l'immagine starless generata da StarNet2 (ora in formato TIFF).

    Gestisce sia TIFF a 16-bit che a 8-bit normalizzando i dati in [0, 1].
    """
    raw = tifffile.imread(path)

    if raw.dtype == np.uint16:
        arr = raw.astype(np.float32) / 65535.0
    elif raw.dtype == np.uint8:
        arr = raw.astype(np.float32) / 255.0
    else:
        raise RuntimeError(f"dtype non supportato: {raw.dtype}")

    return arr


def main():
    parser = argparse.ArgumentParser(
        description="Stima la coerenza di scala tra input e target usando i file TIFF a 16-bit."
    )
    parser.add_argument(
        "--npy", required=True, help="Path al file .npy originale con stelle"
    )
    parser.add_argument(
        "--starless",
        required=True,
        help="Path al file .tif starless generato da StarNet2",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Soglia sulla differenza media RGB.",
    )
    args = parser.parse_args()

    inp = np.load(args.npy).astype(np.float32)
    tgt = load_starless(args.starless)

    if inp.shape != tgt.shape:
        raise RuntimeError(
            f"Shape diverse:\n"
            f"input    : {inp.shape}\n"
            f"starless : {tgt.shape}"
        )

    diff = np.abs(inp - tgt).mean(axis=2)

    # pixel presumibilmente non toccati da StarNet2
    mask = diff < args.threshold

    n = mask.sum()
    total = mask.size

    print(f"\nPixel usati: {n:,} / {total:,} ({100*n/total:.2f}%)")

    if n < 1000:
        print("Troppi pochi pixel selezionati.")
        return

    x = inp[mask].ravel()
    y = tgt[mask].ravel()

    # evita il fondo esattamente a zero
    m = x > 1e-6
    x = x[m]
    y = y[m]

    print(f"Campioni finali: {len(x):,}")

    # y = a*x
    a = np.sum(x * y) / (np.sum(x * x) + 1e-12)

    # y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a2, b2 = np.linalg.lstsq(A, y, rcond=None)[0]

    corr = np.corrcoef(x, y)[0, 1]

    print("\n==============================")
    print("SCALA SU PIXEL NON MODIFICATI")
    print("==============================")
    print(f"y = a*x")
    print(f"a = {a:.6f}")

    print("\ny = a*x + b")
    print(f"a = {a2:.6f}")
    print(f"b = {b2:.6e}")

    print(f"\ncorrelation = {corr:.6f}")

    if abs(a - 1.0) < 0.02:
        print("\n✅ Scale sostanzialmente identiche.")
    elif abs(a - 1.0) < 0.05:
        print("\n⚠️ Piccola differenza di scala.")
    else:
        print("\n❌ Differenza di scala significativa.")

    # scatter
    nplot = min(len(x), 200000)
    idx = np.random.choice(len(x), nplot, replace=False)

    xs = x[idx]
    ys = y[idx]

    lim = max(xs.max(), ys.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, s=0.2, alpha=0.2)

    plt.plot([0, lim], [0, lim], "r--", linewidth=1, label="y=x")

    plt.plot([0, lim], [0, a * lim], "g-", linewidth=1, label=f"y={a:.3f}x")

    plt.xlabel("Input NPY")
    plt.ylabel("Starless")
    plt.title(f"Background scale factor = {a:.4f}")
    plt.legend()

    out = Path(args.npy).with_suffix("")
    out = str(out) + "_background_scale.png"

    plt.savefig(out, dpi=200)
    print(f"\nScatter plot salvato in:\n{out}")


if __name__ == "__main__":
    main()
