#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

dataset = Path("dataset")
inp_dir = dataset / "input" / "npy"
tgt_dir = dataset / "target" / "npy"

corrs = []

for inp_path in sorted(inp_dir.glob("*.npy")):
    tgt_path = tgt_dir / inp_path.name
    if not tgt_path.exists():
        continue

    inp = np.load(inp_path).astype(np.float32)
    tgt = np.load(tgt_path).astype(np.float32)

    c = np.corrcoef(inp.ravel(), tgt.ravel())[0, 1]
    corrs.append((c, inp_path.stem))

corrs.sort()

print("\nWorst patches:\n")
for c, name in corrs[:20]:
    print(f"{c:.4f}  {name}")

N_SHOW = min(20, len(corrs))

for c, name in corrs[:N_SHOW]:
    inp = np.load(inp_dir / f"{name}.npy").astype(np.float32)
    tgt = np.load(tgt_dir / f"{name}.npy").astype(np.float32)

    residual = np.clip(inp - tgt, 0, None)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(np.clip(inp, 0, 1))
    ax[0].set_title("Input")
    ax[0].axis("off")

    ax[1].imshow(np.clip(tgt, 0, 1))
    ax[1].set_title("Target")
    ax[1].axis("off")

    vmax = np.percentile(residual, 99.5)
    if vmax <= 0:
        vmax = 1.0

    ax[2].imshow(
        residual.mean(axis=-1) if residual.ndim == 3 else residual,
        cmap="inferno",
        vmin=0,
        vmax=vmax,
    )
    ax[2].set_title("Residual")
    ax[2].axis("off")

    fig.suptitle(f"{name}   corr={c:.4f}")
    plt.tight_layout()
    plt.show()
