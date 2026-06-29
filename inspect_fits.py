"""
inspect_scale.py

Verifica se il target starless è stato riscalato rispetto all'input.

Uso:

python inspect_scale.py \
  --fits assets/inputs/file.fits \
  --npy assets/outputs-npy/file.npy \
  --starless assets/outputs-starless/file.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_starless(path):
    arr = np.array(Image.open(path))

    if arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"dtype non supportato: {arr.dtype}")

    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits", required=False)
    parser.add_argument("--npy", required=True)
    parser.add_argument("--starless", required=True)
    args = parser.parse_args()

    inp = np.load(args.npy).astype(np.float32)
    tgt = load_starless(args.starless)

    if inp.shape != tgt.shape:
        raise ValueError(
            f"Shape diverse:\n"
            f"input    {inp.shape}\n"
            f"starless {tgt.shape}"
        )

    # Escludi i pixel molto luminosi
    p99 = np.percentile(inp, 99)

    mask = (
        np.isfinite(inp)
        & np.isfinite(tgt)
        & (inp <= p99)
    )

    x = inp[mask]
    y = tgt[mask]

    print(f"Pixel usati: {len(x):,}")
    print(f"p99 input: {p99:.6f}")

    # regressione y = a*x (senza intercetta)
    a = np.sum(x * y) / (np.sum(x * x) + 1e-12)

    # regressione y = a*x + b
    A = np.vstack([x, np.ones_like(x)]).T
    a2, b2 = np.linalg.lstsq(A, y, rcond=None)[0]

    corr = np.corrcoef(x, y)[0, 1]

    print("\n==============================")
    print("SCALA GLOBALE")
    print("==============================")
    print(f"best scale factor (a):     {a:.6f}")
    print(f"linear fit y=a*x+b:")
    print(f"    a = {a2:.6f}")
    print(f"    b = {b2:.6e}")
    print(f"correlation:              {corr:.6f}")

    if abs(a - 1.0) < 0.05:
        print("\n✅ Nessuna evidenza di riscalatura globale.")
    else:
        print("\n⚠️ Possibile differenza di scala tra input e target.")

    # scatter
    n = min(len(x), 200000)
    idx = np.random.choice(len(x), n, replace=False)

    xs = x[idx]
    ys = y[idx]

    lim = max(xs.max(), ys.max())

    plt.figure(figsize=(7, 7))
    plt.scatter(xs, ys, s=0.2, alpha=0.2)
    plt.plot([0, lim], [0, lim], "r--", linewidth=1)
    plt.plot([0, lim], [0, a * lim], "g-", linewidth=1)

    plt.xlabel("Input NPY")
    plt.ylabel("Starless")
    plt.title(
        f"Scale factor={a:.4f}  corr={corr:.5f}"
    )

    out = Path(args.npy).with_suffix("")
    out = str(out) + "_scale_check.png"

    plt.savefig(out, dpi=200)
    print(f"\nScatter salvato in:\n{out}")


if __name__ == "__main__":
    main()
