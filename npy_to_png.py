import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".npy"):
            continue

        npy_path = os.path.join(input_dir, filename)

        try:
            arr = np.load(npy_path)

            # Normalizzazione opzionale
            arr = arr.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()

            output_name = os.path.splitext(filename)[0] + ".png"
            output_path = os.path.join(output_dir, output_name)

            if arr.ndim == 2:
                plt.imsave(output_path, arr, cmap="gray")
            else:
                plt.imsave(output_path, arr)

            print(f"Salvata: {output_path}")

        except Exception as e:
            print(f"Errore con {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converte tutti i file .npy di una cartella in immagini PNG."
    )
    parser.add_argument("--input_dir", help="Cartella contenente i file .npy")
    parser.add_argument("--output_dir", help="Cartella dove salvare le immagini")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
