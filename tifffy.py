import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def npy_to_png(npy_path: Path, png_path: Path):
    arr = np.load(npy_path).astype(np.float32)

    # Se i dati fossero accidentalmente in [0,255]
    if arr.max() > 1.0:
        arr /= 255.0

    arr = np.clip(arr, 0.0, 1.0)

    png_path.parent.mkdir(parents=True, exist_ok=True)

    if arr.ndim == 2:
        plt.imsave(png_path, arr, cmap="gray")
    else:
        plt.imsave(png_path, arr)

    print(f"[OK] {png_path}")


def main(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    npy_files = sorted(input_root.rglob("*.npy"))

    if not npy_files:
        print("Nessun file .npy trovato.")
        return

    print(f"Trovati {len(npy_files)} file.\n")

    for npy_path in npy_files:
        rel = npy_path.relative_to(input_root)
        png_path = output_root / rel.with_suffix(".png")

        try:
            npy_to_png(npy_path, png_path)
        except Exception as e:
            print(f"[ERROR] {npy_path}: {e}")

    print("\nCompletato.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte ricorsivamente un dataset di file .npy in PNG preservando la struttura delle cartelle."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Cartella radice contenente il dataset (.npy).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Cartella radice di destinazione (.png).",
    )

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
