#!/usr/bin/env python3
# debug_residual.py — trova e visualizza le patch con residuo negativo marcato

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path("dataset")
INP_DIR = DATASET / "input"  / "npy"
TGT_DIR = DATASET / "target" / "npy"
THRESHOLD = -0.05   # residuo < questo → "marcatamente negativo"
N_SHOW = 6          # quante patch mostrare

pairs = sorted(INP_DIR.glob("*.npy"))
results = []

for p in pairs:
    a = np.load(p).astype(np.float32)
    b = np.load(TGT_DIR / p.name).astype(np.float32)
    res = a - b
    neg_frac = (res < -0.01).mean()
    neg_min  = res.min()
    results.append((neg_frac, neg_min, p.stem, a, b, res))

results.sort(reverse=True)  # peggiori prima

fig, axes = plt.subplots(N_SHOW, 4, figsize=(16, N_SHOW * 4))
fig.suptitle("Residuo negativo: input | target | residuo | heatmap negativo", fontsize=12)

for row, (neg_frac, neg_min, stem, a, b, res) in enumerate(results[:N_SHOW]):
    # clip per visualizzazione (HWC → usa canale medio se multi-ch)
    disp = lambda x: np.clip(x, 0, 1) if x.ndim == 2 else np.clip(x, 0, 1)
    neg_map = np.clip(-res, 0, None)  # solo valori negativi, flippati in positivo
    if res.ndim == 3:
        neg_map = neg_map.mean(axis=-1)

    axes[row, 0].imshow(disp(a),   cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"{stem}\ninput")
    axes[row, 1].imshow(disp(b),   cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("target (starless)")
    axes[row, 2].imshow(res.mean(axis=-1) if res.ndim==3 else res,
                        cmap="RdBu", vmin=-0.3, vmax=0.3)
    axes[row, 2].set_title(f"residuo (R-B)\nneg={neg_frac*100:.1f}%  min={neg_min:.3f}")
    im = axes[row, 3].imshow(neg_map, cmap="hot", vmin=0, vmax=0.3)
    axes[row, 3].set_title("heatmap negativo")
    plt.colorbar(im, ax=axes[row, 3])

for ax in axes.ravel():
    ax.axis("off")

plt.tight_layout()
plt.savefig("debug_residual.png", dpi=150)
print("Salvato debug_residual.png")
