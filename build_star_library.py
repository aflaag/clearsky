import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage


def load_mask_tiff(path):
    """Carica una star mask 16-bit salvata da StarNet2 (--starmask) e la normalizza in [0,1]."""
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        # se salvata come RGB/multi-canale, i canali sono identici: ne basta uno
        arr = arr[..., 0]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float32)
    else:
        raise ValueError(f"Dtype inatteso per starmask: {arr.dtype}")


def extract_stars_from_image(original, starmask, pixel_mask, min_area, mask_thresh, pad):
    """Estrae uno stamp (RGB + alpha) per ogni componente connessa nella starmask.

    Lo stamp contiene i pixel REALI della stella (dall'immagine con stelle),
    non la ground truth starless: e' quello che vogliamo poter "riappiccicare".
    """
    binary = starmask > mask_thresh
    labeled, _ = ndimage.label(binary)
    H, W = binary.shape
    objs = ndimage.find_objects(labeled)

    stars = []
    for i, sl in enumerate(objs, start=1):
        if sl is None:
            continue
        ys, xs = sl
        area = int((labeled[sl] == i).sum())
        if area < min_area:
            continue

        y0, y1 = max(ys.start - pad, 0), min(ys.stop + pad, H)
        x0, x1 = max(xs.start - pad, 0), min(xs.stop + pad, W)

        # scarta stelle che toccano zone della pixel_mask non valide (bordi, mosaic gaps)
        if pixel_mask is not None and not pixel_mask[y0:y1, x0:x1].all():
            continue

        rgb = original[y0:y1, x0:x1, :].astype(np.float32).copy()
        alpha = starmask[y0:y1, x0:x1].astype(np.float32).copy()

        # azzera il contributo di stelle vicine finite nello stesso crop
        local_labeled = labeled[y0:y1, x0:x1]
        alpha = np.where(local_labeled == i, alpha, 0.0).astype(np.float32)

        if alpha.max() <= 0:
            continue

        peak = float((rgb * alpha[..., None]).max())
        flux = float((rgb * alpha[..., None]).sum())
        h, w = rgb.shape[:2]

        stars.append({"rgb": rgb, "alpha": alpha, "h": h, "w": w, "peak": peak, "flux": flux})
    return stars


def main():
    parser = argparse.ArgumentParser(
        description="Estrae una libreria di 'star stamps' reali (pixel + alpha) da usare per "
        "sintetizzare campi stellari plausibili sopra le basi starless."
    )
    parser.add_argument("--npy-dir", default="assets/outputs-npy", help="NPY originali con stelle")
    parser.add_argument("--starmask-dir", default="assets/outputs-starmask", help="Starmask 16-bit da StarNet2 (--starmask)")
    parser.add_argument("--pixel-mask-dir", default="assets/outputs-mask", help="Maschere pixel validi da make_masks.py")
    parser.add_argument("--out-dir", default="assets/star_library")
    parser.add_argument("--mask-thresh", type=float, default=0.1, help="Soglia binarizzazione starmask")
    parser.add_argument("--min-area", type=int, default=4, help="Area minima (px) per considerare una componente una stella")
    parser.add_argument("--pad", type=int, default=2, help="Padding attorno al bounding box di ogni stella")
    args = parser.parse_args()

    npy_dir = Path(args.npy_dir)
    starmask_dir = Path(args.starmask_dir)
    pixel_mask_dir = Path(args.pixel_mask_dir)
    out_dir = Path(args.out_dir)
    stamps_dir = out_dir / "stamps"
    stamps_dir.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(npy_dir.glob("*.npy"))
    if not npy_files:
        print(f"Nessun file .npy trovato in {npy_dir}")
        return

    metadata = []
    densities = []
    star_id = 0

    for npy_path in npy_files:
        stem = npy_path.stem
        starmask_path = starmask_dir / f"{stem}.tif"
        pixel_mask_path = pixel_mask_dir / f"{stem}.npy"

        if not starmask_path.exists():
            print(f"[SKIP] {stem}: starmask non trovata (rilancia pipeline.sh con --starmask)")
            continue

        original = np.load(npy_path)
        starmask = load_mask_tiff(starmask_path)
        pixel_mask = np.load(pixel_mask_path) if pixel_mask_path.exists() else None

        if starmask.shape != original.shape[:2]:
            print(f"[SKIP] {stem}: shape starmask {starmask.shape} != originale {original.shape[:2]}")
            continue

        stars = extract_stars_from_image(
            original, starmask, pixel_mask,
            min_area=args.min_area, mask_thresh=args.mask_thresh, pad=args.pad,
        )

        valid_area = int(pixel_mask.sum()) if pixel_mask is not None else original.shape[0] * original.shape[1]
        if valid_area > 0:
            densities.append(len(stars) / valid_area)

        for s in stars:
            fname = f"star_{star_id:06d}.npz"
            np.savez_compressed(stamps_dir / fname, rgb=s["rgb"], alpha=s["alpha"])
            metadata.append({
                "file": fname, "h": s["h"], "w": s["w"],
                "peak": s["peak"], "flux": s["flux"], "source": stem,
            })
            star_id += 1

        print(f"{stem}: {len(stars)} stelle estratte (tot. {star_id})")

    if not metadata:
        print("Nessuna stella estratta. Verifica che pipeline.sh sia stato lanciato con --starmask.")
        return

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)
    with open(out_dir / "field_density.json", "w") as f:
        json.dump(densities, f)

    densities_arr = np.array(densities)
    print(f"\nLibreria completata: {star_id} stelle da {len(npy_files)} immagini")
    print(f"Densita' media: {densities_arr.mean():.6f} stelle/px^2 (min {densities_arr.min():.6f}, max {densities_arr.max():.6f})")


if __name__ == "__main__":
    main()
