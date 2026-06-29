#!/usr/bin/env python3
# check_background_gap.py
# Per ogni immagine originale, confronta il livello di fondo
# tra il .npy stretchato e il TIFF starless di StarNet2.

import numpy as np
import tifffile
from pathlib import Path

NPY_DIR      = Path("assets/outputs-npy")
STARLESS_DIR = Path("assets/outputs-starless")
PERCENTILE   = 10.0   # stima cielo

results = []

stems = sorted(p.stem for p in NPY_DIR.glob("*.npy"))

if not stems:
    print("Nessun .npy trovato in assets/outputs-npy")
    exit(1)

print(f"{'Immagine':<55} {'Ch':>2}  {'sky_inp':>8}  {'sky_tgt':>8}  {'delta':>8}")
print("─" * 90)

for stem in stems:
    npy_path  = NPY_DIR      / f"{stem}.npy"
    tiff_path = STARLESS_DIR / f"{stem}.tif"

    if not tiff_path.exists():
        print(f"{stem:<55}  [SKIP] tif mancante")
        continue

    inp = np.load(npy_path).astype(np.float32)          # HWC [0,1]
    tgt = tifffile.imread(tiff_path).astype(np.float32) / 65535.0  # HWC [0,1]

    if inp.shape != tgt.shape:
        print(f"{stem:<55}  [SKIP] shape mismatch {inp.shape} vs {tgt.shape}")
        continue

    C = inp.shape[-1]
    row_deltas = []

    for c in range(C):
        sky_inp = float(np.percentile(inp[..., c], PERCENTILE))
        sky_tgt = float(np.percentile(tgt[..., c], PERCENTILE))
        delta   = sky_inp - sky_tgt
        row_deltas.append(delta)

        label = stem[-40:] if len(stem) > 40 else stem
        print(f"{label:<55} {c:>2}  {sky_inp:>8.4f}  {sky_tgt:>8.4f}  {delta:>+8.4f}")

    results.append((stem, row_deltas))

# ── Riepilogo ────────────────────────────────────────────────────────────────
print("\n" + "─" * 90)
print("Riepilogo delta (inp_sky − tgt_sky) per immagine:\n")

all_deltas = []
for stem, deltas in results:
    mean_d = np.mean(deltas)
    all_deltas.extend(deltas)
    sign = "+" if mean_d > 0 else ""
    flag = "  ← target più luminoso del cielo input" if mean_d < -0.01 else \
           "  ← target più scuro   del cielo input" if mean_d >  0.01 else \
           "  ✓ allineato"
    print(f"  {stem[-50:]:<50}  Δmean={sign}{mean_d:.4f}{flag}")

print(f"\nDelta globale su tutte le coppie:")
print(f"  mean  = {np.mean(all_deltas):+.4f}")
print(f"  std   = {np.std(all_deltas):.4f}")
print(f"  min   = {np.min(all_deltas):+.4f}")
print(f"  max   = {np.max(all_deltas):+.4f}")

systematic = abs(np.mean(all_deltas)) > 0.01
consistent = np.std(all_deltas) < 0.01

if systematic and consistent:
    print("\n→ Offset SISTEMATICO e COSTANTE: fix con sottrazione scalare globale sufficiente.")
elif systematic and not consistent:
    print("\n→ Offset SISTEMATICO ma VARIABILE tra immagini: fix per-patch (match_background) necessario.")
else:
    print("\n→ Nessun offset sistematico rilevante.")
