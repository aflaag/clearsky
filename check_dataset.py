#!/usr/bin/env python3
"""
check_dataset.py — Dataset quality inspector for clearsky DDPM pipeline.

Checks:
  • Structural integrity  (file counts, shape pairing, dtype)
  • Value ranges         (global min/max, saturated/black pixel %)
  • Distribution         (mean, std, percentiles per patch)
  • Input/Target delta   (residual = input - target, i.e. the "stars")
  • Cross-pair alignment (same spatial content after star removal?)
  • Outlier patches      (near-black, near-white, low-variance)

Usage:
    python check_dataset.py [--dataset-dir DATASET_DIR] [--sample N] [--verbose]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# ─── ANSI colours ────────────────────────────────────────────────────────────
R = "\033[91m"; Y = "\033[93m"; G = "\033[92m"; C = "\033[96m"; B = "\033[1m"; E = "\033[0m"

def ok(msg):   print(f"  {G}✓{E} {msg}")
def warn(msg): print(f"  {Y}⚠{E} {msg}")
def err(msg):  print(f"  {R}✗{E} {msg}")
def hdr(msg):  print(f"\n{B}{C}{'─'*60}\n  {msg}\n{'─'*60}{E}")

# ─── helpers ─────────────────────────────────────────────────────────────────

def load_npy(path: Path) -> np.ndarray:
    return np.load(str(path)).astype(np.float32)

def npy_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.npy"))

def summarise(arr: np.ndarray, label: str = "") -> dict:
    d = {
        "min":   float(arr.min()),
        "max":   float(arr.max()),
        "mean":  float(arr.mean()),
        "std":   float(arr.std()),
        "p01":   float(np.percentile(arr, 1)),
        "p50":   float(np.percentile(arr, 50)),
        "p99":   float(np.percentile(arr, 99)),
        "sat%":  float((arr >= 0.999).mean() * 100),
        "blk%":  float((arr <= 0.001).mean() * 100),
    }
    return d

def print_stats(s: dict, indent: int = 4):
    pad = " " * indent
    print(f"{pad}range  : [{s['min']:.4f}, {s['max']:.4f}]")
    print(f"{pad}mean   : {s['mean']:.4f}   std : {s['std']:.4f}")
    print(f"{pad}p01/p50/p99: {s['p01']:.4f} / {s['p50']:.4f} / {s['p99']:.4f}")
    print(f"{pad}sat≥0.999: {s['sat%']:.2f}%   blk≤0.001: {s['blk%']:.2f}%")

# ─── checks ──────────────────────────────────────────────────────────────────

def check_structure(inp_dir: Path, tgt_dir: Path) -> tuple[list, list]:
    hdr("1 · Structural integrity")

    inp_files = npy_files(inp_dir)
    tgt_files = npy_files(tgt_dir)

    print(f"  input  patches found : {len(inp_files)}")
    print(f"  target patches found : {len(tgt_files)}")

    if len(inp_files) == 0:
        err("No input .npy files found — wrong directory?"); sys.exit(1)
    if len(tgt_files) == 0:
        err("No target .npy files found — wrong directory?"); sys.exit(1)

    inp_stems = {p.stem for p in inp_files}
    tgt_stems = {p.stem for p in tgt_files}
    only_inp  = inp_stems - tgt_stems
    only_tgt  = tgt_stems - inp_stems

    if only_inp:
        warn(f"{len(only_inp)} input(s) have no paired target: {sorted(only_inp)[:5]}")
    if only_tgt:
        warn(f"{len(only_tgt)} target(s) have no paired input: {sorted(only_tgt)[:5]}")

    paired_stems = sorted(inp_stems & tgt_stems)
    if not paired_stems:
        err("No matched pairs found — filenames must match between input/ and target/."); sys.exit(1)

    ok(f"{len(paired_stems)} matched pairs")
    return (
        [inp_dir / (s + ".npy") for s in paired_stems],
        [tgt_dir / (s + ".npy") for s in paired_stems],
    )


def check_shapes_and_dtypes(inp_paths: list, tgt_paths: list, sample: int) -> bool:
    hdr("2 · Shape & dtype")

    indices = np.random.choice(len(inp_paths), min(sample, len(inp_paths)), replace=False)
    shapes_ok = True
    shape_set_inp = set(); shape_set_tgt = set()

    for i in indices:
        a = np.load(str(inp_paths[i]))
        b = np.load(str(tgt_paths[i]))

        shape_set_inp.add(a.shape)
        shape_set_tgt.add(b.shape)

        if a.shape != b.shape:
            err(f"Shape mismatch at {inp_paths[i].stem}: input={a.shape} target={b.shape}")
            shapes_ok = False
        if a.dtype not in (np.float32,):
            warn(f"Input dtype is {a.dtype} (expected float32): {inp_paths[i].name}")
        if b.dtype not in (np.float32,):
            warn(f"Target dtype is {b.dtype} (expected float32): {tgt_paths[i].name}")

    if shapes_ok:
        ok(f"All sampled pairs have matching shapes")
    print(f"    input  shapes seen : {shape_set_inp}")
    print(f"    target shapes seen : {shape_set_tgt}")

    if len(shape_set_inp) > 1:
        warn("Multiple distinct input shapes — is that intentional?")
    return shapes_ok


def check_value_ranges(inp_paths: list, tgt_paths: list, sample: int) -> tuple[dict, dict]:
    hdr("3 · Global value ranges (sampled)")

    indices = np.random.choice(len(inp_paths), min(sample, len(inp_paths)), replace=False)
    inp_data = np.concatenate([np.load(str(inp_paths[i])).ravel() for i in indices])
    tgt_data = np.concatenate([np.load(str(tgt_paths[i])).ravel() for i in indices])

    si = summarise(inp_data, "input")
    st = summarise(tgt_data, "target")

    print(f"  {B}Input  (with stars):{E}")
    print_stats(si)
    print(f"  {B}Target (starless):{E}")
    print_stats(st)

    if si["min"] < -0.01 or si["max"] > 1.01:
        warn(f"Input values outside [0,1]: [{si['min']:.4f}, {si['max']:.4f}]")
    else:
        ok("Input range inside [0, 1]")

    if st["min"] < -0.01 or st["max"] > 1.01:
        warn(f"Target values outside [0,1]: [{st['min']:.4f}, {st['max']:.4f}]")
    else:
        ok("Target range inside [0, 1]")

    if si["sat%"] > 5:
        warn(f"Input has {si['sat%']:.1f}% saturated pixels (≥0.999) — check stretch")
    if st["sat%"] > 5:
        warn(f"Target has {st['sat%']:.1f}% saturated pixels (≥0.999) — residual stars in GT?")

    return si, st


def check_residual(inp_paths: list, tgt_paths: list, sample: int):
    hdr("4 · Star residual (input − target)")

    indices = np.random.choice(len(inp_paths), min(sample, len(inp_paths)), replace=False)
    residuals = []
    for i in indices:
        a = np.load(str(inp_paths[i])).astype(np.float32)
        b = np.load(str(tgt_paths[i])).astype(np.float32)
        residuals.append((a - b).ravel())

    res = np.concatenate(residuals)
    sr  = summarise(res)
    print_stats(sr)

    # Fraction of pixels with meaningful star signal
    star_frac = float((res > 0.01).mean() * 100)
    neg_frac  = float((res < -0.01).mean() * 100)

    print(f"    pixels with res > +0.01 (star signal) : {star_frac:.2f}%")
    print(f"    pixels with res < -0.01 (negative leak): {neg_frac:.2f}%")

    if sr["mean"] < 0.001:
        warn("Mean residual ≈ 0 — are stars actually being injected?")
    else:
        ok(f"Mean residual {sr['mean']:.4f} — stars present in input vs target")

    if neg_frac > 2:
        warn(f"{neg_frac:.1f}% of pixels show target brighter than input — "
             "StarNet2 may be adding nebulosity or there's a normalisation mismatch")

    if sr["max"] < 0.05:
        warn("Max residual very low — synthetic stars may be too faint")
    elif sr["max"] > 0.95:
        warn("Max residual very high — some stars may be fully saturated / clipped")
    else:
        ok(f"Residual peak {sr['max']:.3f} looks plausible")


def check_per_patch_outliers(inp_paths: list, tgt_paths: list,
                              sample: int, verbose: bool):
    hdr("5 · Per-patch outlier detection")

    indices = np.random.choice(len(inp_paths), min(sample * 2, len(inp_paths)), replace=False)

    nearly_black_inp  = []
    nearly_white_inp  = []
    low_variance_inp  = []
    low_variance_tgt  = []
    large_mean_delta  = []

    THRESH_BLACK = 0.05   # mean < this → boring black patch
    THRESH_WHITE = 0.90   # mean > this → overexposed
    THRESH_VAR   = 1e-5   # variance < this → flat / mask artifact
    THRESH_DELTA = 0.15   # |mean_inp - mean_tgt| > this → suspicious

    for i in indices:
        a = np.load(str(inp_paths[i])).astype(np.float32)
        b = np.load(str(tgt_paths[i])).astype(np.float32)

        ma = float(a.mean()); va = float(a.var())
        mb = float(b.mean()); vb = float(b.var())

        stem = inp_paths[i].stem
        if ma < THRESH_BLACK: nearly_black_inp.append((stem, ma))
        if ma > THRESH_WHITE: nearly_white_inp.append((stem, ma))
        if va < THRESH_VAR:   low_variance_inp.append((stem, va))
        if vb < THRESH_VAR:   low_variance_tgt.append((stem, vb))
        if abs(ma - mb) > THRESH_DELTA: large_mean_delta.append((stem, ma - mb))

    def report(label, lst, fmt):
        if lst:
            warn(f"{len(lst)} patches {label}")
            if verbose:
                for item in lst[:10]:
                    print(f"      {item[0]} : {fmt.format(item[1])}")
        else:
            ok(f"No patches {label}")

    total = len(indices)
    print(f"  Checked {total} patches\n")
    report(f"nearly black (mean < {THRESH_BLACK})",   nearly_black_inp,  "{:.4f}")
    report(f"nearly white (mean > {THRESH_WHITE})",   nearly_white_inp,  "{:.4f}")
    report(f"flat input   (var  < {THRESH_VAR})",     low_variance_inp,  "{:.2e}")
    report(f"flat target  (var  < {THRESH_VAR})",     low_variance_tgt,  "{:.2e}")
    report(f"|Δmean| > {THRESH_DELTA} (normalisation mismatch?)", large_mean_delta, "{:+.4f}")


def check_channel_consistency(inp_paths: list, tgt_paths: list, sample: int):
    hdr("6 · Channel consistency (multi-channel patches)")

    idx = np.random.choice(len(inp_paths), min(sample, len(inp_paths)), replace=False)
    a0 = np.load(str(inp_paths[idx[0]]))

    if a0.ndim == 2:
        ok("Patches are single-channel (H×W) — nothing to check")
        return
    if a0.ndim != 3:
        warn(f"Unexpected ndim={a0.ndim}"); return

    C = a0.shape[0] if a0.shape[0] <= 8 else a0.shape[-1]  # CHW vs HWC heuristic
    layout = "CHW" if a0.shape[0] <= 8 else "HWC"
    print(f"  Layout heuristic: {layout},  channels: {C}")

    chan_means_inp = np.zeros((len(idx), C))
    chan_means_tgt = np.zeros((len(idx), C))

    for k, i in enumerate(idx):
        a = np.load(str(inp_paths[i])).astype(np.float32)
        b = np.load(str(tgt_paths[i])).astype(np.float32)
        if layout == "CHW":
            chan_means_inp[k] = a.reshape(C, -1).mean(axis=1)
            chan_means_tgt[k] = b.reshape(C, -1).mean(axis=1)
        else:
            chan_means_inp[k] = a.reshape(-1, C).mean(axis=0)
            chan_means_tgt[k] = b.reshape(-1, C).mean(axis=0)

    for c in range(C):
        m_i = chan_means_inp[:, c].mean()
        m_t = chan_means_tgt[:, c].mean()
        print(f"    Ch{c}  inp_mean={m_i:.4f}  tgt_mean={m_t:.4f}  Δ={m_i-m_t:+.4f}")


def check_dataset_size(inp_paths: list, patch_size: int = 256, batch_size: int = 8):
    hdr("7 · Dataset size sanity")

    n = len(inp_paths)
    print(f"  Total pairs : {n}")

    # Rule of thumb for DDPM: at least a few thousand patches
    if n < 500:
        warn(f"Only {n} patches — DDPM training may underfit; aim for ≥ 2000")
    elif n < 2000:
        warn(f"{n} patches — marginal; consider extracting more patches per image")
    else:
        ok(f"{n} patches — should be sufficient to start training")

    steps_per_epoch = n // batch_size
    print(f"  Steps/epoch @ batch={batch_size} : {steps_per_epoch}")
    if steps_per_epoch < 50:
        warn("Very few steps per epoch — consider larger batch or more data")

    # Rough disk footprint
    a = np.load(str(inp_paths[0]))
    bytes_per_patch = a.nbytes
    total_gb = (bytes_per_patch * 2 * n) / 1e9
    print(f"  Approx. dataset size : {total_gb:.2f} GB  (float32 × 2 arrays)")


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset health-check for clearsky DDPM")
    parser.add_argument("--dataset-dir", default="dataset",
                        help="Root of the dataset directory (default: ./dataset)")
    parser.add_argument("--sample", type=int, default=200,
                        help="Number of patches to sample for stats (default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print filenames of outlier patches")
    args = parser.parse_args()

    root     = Path(args.dataset_dir)
    inp_npy  = root / "input"  / "npy"
    tgt_npy  = root / "target" / "npy"

    print(f"\n{B}clearsky — dataset inspector{E}")
    print(f"  dataset dir : {root.resolve()}")
    print(f"  sample size : {args.sample}")

    for d in (inp_npy, tgt_npy):
        if not d.exists():
            err(f"Directory not found: {d}")
            sys.exit(1)

    np.random.seed(42)

    inp_paths, tgt_paths = check_structure(inp_npy, tgt_npy)
    check_shapes_and_dtypes(inp_paths, tgt_paths, args.sample)
    check_value_ranges(inp_paths, tgt_paths, args.sample)
    check_residual(inp_paths, tgt_paths, args.sample)
    check_per_patch_outliers(inp_paths, tgt_paths, args.sample, args.verbose)
    check_channel_consistency(inp_paths, tgt_paths, args.sample)
    check_dataset_size(inp_paths)

    print(f"\n{B}Done.{E}\n")


if __name__ == "__main__":
    main()
