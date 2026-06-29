"""
debug_pair.py

Diagnostica di una coppia:
    input.npy
    starless.png

Uso:

python debug_pair.py \
    --npy assets/outputs-npy/foo.npy \
    --starless assets/outputs-starless/foo.png
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def load_png(path):
    raw = np.array(Image.open(path))

    if raw.dtype == np.uint8:
        norm = raw.astype(np.float32) / 255.0
        divisor = 255
    elif raw.dtype == np.uint16:
        norm = raw.astype(np.float32) / 65535.0
        divisor = 65535
    else:
        raise RuntimeError(f"dtype non supportato: {raw.dtype}")

    return raw, norm, divisor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npy", required=True)
    parser.add_argument("--starless", required=True)
    args = parser.parse_args()

    inp = np.load(args.npy).astype(np.float32)
    raw, tgt, divisor = load_png(args.starless)

    print("\n==============================")
    print("INPUT")
    print("==============================")
    print("shape:", inp.shape)
    print("dtype:", inp.dtype)
    print("min:", inp.min())
    print("max:", inp.max())
    print("mean:", inp.mean())

    print("\n==============================")
    print("STARLESS")
    print("==============================")
    print("shape:", raw.shape)
    print("dtype:", raw.dtype)
    print("divisor:", divisor)
    print("raw min:", raw.min())
    print("raw max:", raw.max())
    print("unique levels:", np.unique(raw).size)
    print("norm min:", tgt.min())
    print("norm max:", tgt.max())
    print("norm mean:", tgt.mean())

    if inp.shape != tgt.shape:
        print("\nERRORE: shape diverse")
        print(inp.shape)
        print(tgt.shape)
        return

    diff = inp - tgt
    absdiff = np.abs(diff)

    print("\n==============================")
    print("DIFFERENZE")
    print("==============================")
    print("mean abs diff:", absdiff.mean())
    print("median abs diff:", np.median(absdiff))
    print("p95 abs diff:", np.percentile(absdiff, 95))
    print("p99 abs diff:", np.percentile(absdiff, 99))

    # heatmap differenze
    diff_img = absdiff.mean(axis=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(diff_img)
    plt.colorbar()
    plt.title("Mean absolute difference")
    out1 = Path(args.npy).with_suffix("")
    out1 = str(out1) + "_diffmap.png"
    plt.savefig(out1, dpi=200)
    plt.close()

    # patch centrale
    h, w = inp.shape[:2]
    s = min(512, h, w)

    y0 = h // 2 - s // 2
    x0 = w // 2 - s // 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(inp[y0:y0+s, x0:x0+s])
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(tgt[y0:y0+s, x0:x0+s])
    ax[1].set_title("Starless")
    ax[1].axis("off")

    ax[2].imshow(
        absdiff[y0:y0+s, x0:x0+s].mean(axis=2)
    )
    ax[2].set_title("Abs diff")
    ax[2].axis("off")

    plt.tight_layout()

    out2 = Path(args.npy).with_suffix("")
    out2 = str(out2) + "_patchcheck.png"

    plt.savefig(out2, dpi=200)
    plt.close()

    print("\nSalvati:")
    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
