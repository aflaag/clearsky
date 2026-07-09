"""
Super-Resolution (SU) Dataset Builder

Creates (degraded_input, high_resolution) training pairs for conditioned DDPM 
super-resolution training. It pairs degraded NPY inputs (which have already been 
upsampled to the HR resolution by degrade_images.py) with the original high-resolution 
TIFF ground truths. It uses an integral image to efficiently sample valid patches strictly 
within the provided pixel masks, ensuring it avoids completely black or featureless patches.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import tifffile


def build_candidate_mask(pixel_mask, patch_size):
    """Integral image (summed area table) on the pixel mask.

    Returns a boolean map (H, W) of valid top-left positions for patches.
    """
    H, W = pixel_mask.shape
    p = patch_size

    if H < p or W < p:
        return np.zeros((H, W), dtype=bool)

    pv = pixel_mask.astype(np.int32)
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(pv, axis=0), axis=1)

    y_max = H - p
    x_max = W - p

    Y, X = np.meshgrid(np.arange(y_max), np.arange(x_max), indexing="ij")

    patch_sums = (
        sat[Y + p, X + p]
        - sat[Y, X + p]
        - sat[Y + p, X]
        + sat[Y, X]
    )

    candidate_mask = np.zeros((H, W), dtype=bool)
    candidate_mask[:y_max, :x_max] = patch_sums == p * p
    return candidate_mask


def sample_position(candidate_mask):
    """Samples a random top-left position among the valid ones."""
    ys, xs = np.where(candidate_mask)
    if len(ys) == 0:
        return None, None
    idx = np.random.randint(len(ys))
    return int(ys[idx]), int(xs[idx])


def apply_augmentation(patch, flip_h, flip_v, rot_k):
    """Applies deterministic augmentation given the parameter set."""
    if flip_h:
        patch = np.fliplr(patch)
    if flip_v:
        patch = np.flipud(patch)
    if rot_k > 0:
        patch = np.rot90(patch, k=rot_k)
    return np.ascontiguousarray(patch)


def random_aug_params():
    """Samples random augmentation parameters."""
    return {
        "flip_h": np.random.rand() < 0.5,
        "flip_v": np.random.rand() < 0.5,
        "rot_k": int(np.random.randint(4)),
    }


def save_patch(patch, stem, npy_dir, png_dir):
    """Saves patch as float32 NPY and, if png_dir is not None, also uint8 PNG."""
    np.save(npy_dir / f"{stem}.npy", patch.astype(np.float32))
    if png_dir is not None:
        png = np.clip(patch * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(png, mode="RGB").save(png_dir / f"{stem}.png")


def load_hr_tiff(tiff_path):
    """Loads the HR TIFF (ground truth, output of astro_stretch.py).

    16-bit -> normalizes to [0, 1] by dividing by 65535.
    """
    arr = tifffile.imread(tiff_path)

    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    elif arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unexpected dtype for HR TIFF: {arr.dtype}")


def process_file(
    degraded_npy_path,
    hr_tiff_path,
    mask_path,
    input_npy_dir,
    input_png_dir,
    target_npy_dir,
    target_png_dir,
    patch_size,
    crops_per_image,
):
    print(f"Processing {degraded_npy_path.name}")

    # --- Load degraded input (already upsampled to HR resolution) ---
    degraded = np.load(degraded_npy_path)  # (H, W, 3) float32
    if degraded.ndim != 3 or degraded.shape[-1] != 3:
        print(f"  [SKIP] unexpected degraded NPY shape: {degraded.shape}")
        return 0

    H, W, _ = degraded.shape

    if H < patch_size or W < patch_size:
        print(
            f"  [SKIP] {H}x{W} too small for {patch_size}x{patch_size} patch"
        )
        return 0

    # --- Load HR (ground truth) ---
    if not hr_tiff_path.exists():
        print(f"  [SKIP] HR not found: {hr_tiff_path}")
        return 0

    hr = load_hr_tiff(hr_tiff_path)

    # --- Channel sanitization before cropping ---
    if hr.ndim == 2:
        hr = np.stack([hr, hr, hr], axis=-1)
    elif hr.ndim == 3 and hr.shape[-1] == 4:
        hr = hr[:, :, :3]

    if hr.shape != degraded.shape:
        print(f"  [SKIP] Unrecoverable mismatch: HR {hr.shape} vs degraded {degraded.shape}")
        return 0

    # --- Load mask (reused from make_masks.py, computed on the HR image) ---
    if not mask_path.exists():
        print(f"  [SKIP] mask not found: {mask_path}")
        return 0

    pixel_mask = np.load(mask_path)  # (H, W) bool
    if pixel_mask.shape != (H, W):
        print(f"  [SKIP] mask shape {pixel_mask.shape} != {H}x{W}")
        return 0

    # --- Calculate valid positions ---
    candidate_mask = build_candidate_mask(pixel_mask, patch_size)
    n_candidates = int(candidate_mask.sum())

    if n_candidates == 0:
        print(
            f"  [SKIP] no valid position for {patch_size}x{patch_size} patch"
        )
        return 0

    png_label = " + png" if input_png_dir is not None else ""
    print(
        f"  Shape: {H}x{W} | Candidates: {n_candidates} | Crops: {crops_per_image}{png_label}"
    )

    basename = degraded_npy_path.stem
    saved = 0

    # Thresholds (identical to make_dataset.py, to avoid black/flat patches)
    THRESH_BLACK = 0.05
    THRESH_VAR = 1e-5
    MAX_RETRIES = 10

    for i in range(crops_per_image):
        valid_patch_found = False

        for attempt in range(MAX_RETRIES):
            y, x = sample_position(candidate_mask)
            if y is None:
                break

            patch_input = degraded[y : y + patch_size, x : x + patch_size, :]

            patch_mean = float(patch_input.mean())
            patch_var = float(patch_input.var())

            if patch_mean >= THRESH_BLACK and patch_var >= THRESH_VAR:
                valid_patch_found = True
                break

        if not valid_patch_found:
            print(f"  [WARN] Impossible to find a non-empty patch for crop {i} after {MAX_RETRIES} attempts")
            continue

        # Same (y, x) coordinates on HR: degraded and HR have the same shape,
        # so the crop is perfectly aligned pixel by pixel.
        patch_target = hr[y : y + patch_size, x : x + patch_size, :]

        # Same augmentation for both
        aug = random_aug_params()
        patch_input = apply_augmentation(patch_input, **aug)
        patch_target = apply_augmentation(patch_target, **aug)

        stem = f"{basename}_{i:04d}"
        save_patch(patch_input, stem, input_npy_dir, input_png_dir)
        save_patch(patch_target, stem, target_npy_dir, target_png_dir)
        saved += 1

    print(f"  Saved {saved}/{crops_per_image} crops")
    return saved


def main():
    parser = argparse.ArgumentParser(
        description="Creates (degraded_input, HR) pairs for conditioned DDPM super-resolution training."
    )
    parser.add_argument(
        "--degraded-dir",
        default="assets/outputs-degraded/npy",
        help="Degraded NPYs generated by degrade_images.py",
    )
    parser.add_argument(
        "--hr-dir",
        default="assets/outputs-tiff",
        help="HR TIFFs (output of astro_stretch.py --save-tiff)",
    )
    parser.add_argument(
        "--mask-dir",
        default="assets/outputs-mask",
        help="Masks from make_masks.py",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset_su",
        help="Output folder (default: dataset_su)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size (default: 256)",
    )
    parser.add_argument(
        "--crops-per-image",
        type=int,
        default=50,
        help="Crops per image (default: 50)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed (default: 42)"
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Also save PNG patches (useful for visual debugging)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    degraded_dir = Path(args.degraded_dir)
    hr_dir = Path(args.hr_dir)
    mask_dir = Path(args.mask_dir)
    out = Path(args.out_dir)

    input_npy_dir = out / "input" / "npy"
    target_npy_dir = out / "target" / "npy"

    input_png_dir = out / "input" / "png" if args.save_png else None
    target_png_dir = out / "target" / "png" if args.save_png else None

    for d in [input_npy_dir, target_npy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        input_png_dir.mkdir(parents=True, exist_ok=True)
        target_png_dir.mkdir(parents=True, exist_ok=True)

    degraded_files = sorted(degraded_dir.glob("*.npy"))
    if not degraded_files:
        print(f"No .npy files found in {degraded_dir}")
        return

    print(f"Found {len(degraded_files)} NPY files (degraded)")
    print(f"Patch size : {args.patch_size}x{args.patch_size}")
    print(f"Crops/image: {args.crops_per_image}")
    print(f"Save PNG   : {args.save_png}")
    print(f"Output     : {out}\n")

    total = 0
    for degraded_path in degraded_files:
        stem = degraded_path.stem
        total += process_file(
            degraded_npy_path=degraded_path,
            hr_tiff_path=hr_dir / f"{stem}.tif",
            mask_path=mask_dir / f"{stem}.npy",
            input_npy_dir=input_npy_dir,
            input_png_dir=input_png_dir,
            target_npy_dir=target_npy_dir,
            target_png_dir=target_png_dir,
            patch_size=args.patch_size,
            crops_per_image=args.crops_per_image,
        )

    print(f"\nCompleted. Total pairs saved: {total}")


if __name__ == "__main__":
    main()
