"""
Image Restoration (IR) Dataset Builder

Creates (input_with_defects, corrected) training pairs for DDPM image restoration.
It specifically targets pixel-level defects (dead/hot pixels). To ensure the model 
learns effectively, it utilizes an integral image (summed area table) to actively 
sample patches containing a minimum number of defective pixels based on a specified ratio,
while still providing "negative" examples (clean patches) so the model learns not to 
alter healthy pixels.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def build_patch_sum_map(mask, patch_size):
    """Integral image (summed area table) on a binary mask.

    Returns (H, W) int64: sum of True pixels in each patch_size x patch_size patch
    with top-left at (y, x). Valid values only for y <= H-p, x <= W-p (0 elsewhere).
    """
    H, W = mask.shape
    p = patch_size

    if H < p or W < p:
        return np.zeros((H, W), dtype=np.int64)

    mv = mask.astype(np.int32)
    sat = np.zeros((H + 1, W + 1), dtype=np.int64)
    sat[1:, 1:] = np.cumsum(np.cumsum(mv, axis=0), axis=1)

    y_max = H - p
    x_max = W - p
    Y, X = np.meshgrid(np.arange(y_max), np.arange(x_max), indexing="ij")

    patch_sums = (
        sat[Y + p, X + p]
        - sat[Y, X + p]
        - sat[Y + p, X]
        + sat[Y, X]
    )

    sum_map = np.zeros((H, W), dtype=np.int64)
    sum_map[:y_max, :x_max] = patch_sums
    return sum_map


def build_candidate_mask(pixel_mask, patch_size):
    """Valid top-lefts: patches entirely within the valid region."""
    sum_map = build_patch_sum_map(pixel_mask, patch_size)
    return sum_map == patch_size * patch_size


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


def process_file(
    input_path,
    target_path,
    mask_path,
    input_npy_dir,
    input_png_dir,
    target_npy_dir,
    target_png_dir,
    patch_size,
    crops_per_image,
    defect_ratio,
    min_defect_pixels,
    defect_eps,
):
    print(f"Processing {input_path.name}")

    # --- Load input image (original, with dead/hot pixels) ---
    defective = np.load(input_path)  # (H, W, 3) float32
    if defective.ndim != 3 or defective.shape[-1] != 3:
        print(f"  [SKIP] unexpected NPY input shape: {defective.shape}")
        return 0

    H, W, _ = defective.shape

    if H < patch_size or W < patch_size:
        print(
            f"  [SKIP] {H}x{W} too small for {patch_size}x{patch_size} patch"
        )
        return 0

    # --- Load target (corrected by detect_pixel_defects.py) ---
    if not target_path.exists():
        print(f"  [SKIP] target not found: {target_path}")
        return 0

    target = np.load(target_path)

    if target.shape != defective.shape:
        print(f"  [SKIP] Shape mismatch: target {target.shape} vs input {defective.shape}")
        return 0

    # --- Load validity mask ---
    if not mask_path.exists():
        print(f"  [SKIP] mask not found: {mask_path}")
        return 0

    pixel_mask = np.load(mask_path)  # (H, W) bool
    if pixel_mask.shape != (H, W):
        print(f"  [SKIP] mask shape {pixel_mask.shape} != {H}x{W}")
        return 0

    # --- Defect map: where input and target differ ---
    # (equivalent to the bad_mask from detect_pixel_defects.py, obtained by subtraction,
    # as already intuited: where the correction acted, input != target)
    defect_mask = np.abs(defective - target).sum(axis=-1) > defect_eps

    # --- Candidate positions (patches entirely in the valid region) ---
    candidate_mask = build_candidate_mask(pixel_mask, patch_size)
    n_candidates = int(candidate_mask.sum())

    if n_candidates == 0:
        print(
            f"  [SKIP] no valid position for {patch_size}x{patch_size} patch"
        )
        return 0

    # --- Positions whose patch contains enough defective pixels ---
    defect_count_map = build_patch_sum_map(defect_mask, patch_size)
    defect_candidate_mask = candidate_mask & (defect_count_map >= min_defect_pixels)
    n_defect_candidates = int(defect_candidate_mask.sum())

    png_label = " + png" if input_png_dir is not None else ""
    print(
        f"  Shape: {H}x{W} | Candidates: {n_candidates} "
        f"(with defects: {n_defect_candidates}) | Crops: {crops_per_image}{png_label}"
    )

    if n_defect_candidates == 0:
        print(
            f"  [WARN] no patch contains >= {min_defect_pixels} defective pixels: "
            f"all crops will be sampled randomly (check detect_pixel_defects.py thresholds)"
        )

    basename = input_path.stem
    saved = 0

    # Thresholds (identical to make_dataset_sr.py)
    THRESH_BLACK = 0.05
    THRESH_VAR = 1e-5
    MAX_RETRIES = 10  # Avoid infinite loops if the image is entirely dark

    for i in range(crops_per_image):
        # With defect_ratio probability, prioritize patches with real defects;
        # the rest are "negative" examples (the model must also learn 
        # not to alter already clean pixels)
        use_defect_pool = n_defect_candidates > 0 and np.random.rand() < defect_ratio
        pool = defect_candidate_mask if use_defect_pool else candidate_mask

        valid_patch_found = False

        for attempt in range(MAX_RETRIES):
            y, x = sample_position(pool)
            if y is None:
                break  # No valid position in this pool

            patch_input = defective[y : y + patch_size, x : x + patch_size, :]

            patch_mean = float(patch_input.mean())
            patch_var = float(patch_input.var())

            # Validity condition: not too black AND not too flat
            if patch_mean >= THRESH_BLACK and patch_var >= THRESH_VAR:
                valid_patch_found = True
                break

        if not valid_patch_found:
            print(
                f"  [WARN] Impossible to find a valid patch for crop {i} "
                f"after {MAX_RETRIES} attempts"
            )
            continue

        patch_target = target[y : y + patch_size, x : x + patch_size, :]

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
        description="Creates (input_with_defects, corrected) pairs for DDPM image restoration training."
    )
    parser.add_argument(
        "--input-dir",
        default="assets/outputs-npy",
        help="Original stretched NPYs (with dead/hot pixels), output of astro_stretch.py",
    )
    parser.add_argument(
        "--target-dir",
        default="assets/outputs-pixelfix",
        help="Corrected NPYs, output of detect_pixel_defects.py",
    )
    parser.add_argument(
        "--mask-dir",
        default="assets/outputs-mask",
        help="Masks from make_masks.py",
    )
    parser.add_argument(
        "--out-dir",
        default="dataset_ir",
        help="Output folder (default: dataset_ir)",
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
        "--defect-ratio",
        type=float,
        default=0.8,
        help="Fraction of crops sampled preferentially from patches with defects (default: 0.8)",
    )
    parser.add_argument(
        "--min-defect-pixels",
        type=int,
        default=1,
        help="Minimum corrected pixels required in a 'defective' patch (default: 1)",
    )
    parser.add_argument(
        "--defect-eps",
        type=float,
        default=1e-6,
        help="Threshold on absolute input-target difference to count a pixel as defective (default: 1e-6)",
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

    input_dir = Path(args.input_dir)
    target_dir = Path(args.target_dir)
    mask_dir = Path(args.mask_dir)
    out = Path(args.out_dir)

    input_npy_dir = out / "input" / "npy"
    target_npy_dir = out / "target" / "npy"

    # PNG folders created only if requested
    input_png_dir = out / "input" / "png" if args.save_png else None
    target_png_dir = out / "target" / "png" if args.save_png else None

    for d in [input_npy_dir, target_npy_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if args.save_png:
        input_png_dir.mkdir(parents=True, exist_ok=True)
        target_png_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.npy"))
    if not input_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Found {len(input_files)} NPY files (input)")
    print(f"Patch size      : {args.patch_size}x{args.patch_size}")
    print(f"Crops/image     : {args.crops_per_image}")
    print(f"Defect ratio    : {args.defect_ratio}")
    print(f"Min defect px   : {args.min_defect_pixels}")
    print(f"Save PNG        : {args.save_png}")
    print(f"Output          : {out}\n")

    total = 0
    for input_file in input_files:
        stem = input_file.stem
        total += process_file(
            input_path=input_file,
            target_path=target_dir / f"{stem}.npy",
            mask_path=mask_dir / f"{stem}.npy",
            input_npy_dir=input_npy_dir,
            input_png_dir=input_png_dir,
            target_npy_dir=target_npy_dir,
            target_png_dir=target_png_dir,
            patch_size=args.patch_size,
            crops_per_image=args.crops_per_image,
            defect_ratio=args.defect_ratio,
            min_defect_pixels=args.min_defect_pixels,
            defect_eps=args.defect_eps,
        )

    print(f"\nCompleted. Total pairs saved: {total}")


if __name__ == "__main__":
    main()
