#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image

dataset = Path("dataset")
inp_dir = dataset / "input" / "npy"
tgt_dir = dataset / "target" / "npy"

out_dir = Path("negative_residuals")
out_dir.mkdir(exist_ok=True)

for inp_path in sorted(inp_dir.glob("*.npy")):
    tgt_path = tgt_dir / inp_path.name
    if not tgt_path.exists():
        continue

    inp = np.load(inp_path).astype(np.float32)
    tgt = np.load(tgt_path).astype(np.float32)

    neg = (tgt > inp + 0.01)

    frac = neg.mean()

    if frac < 0.01:
        continue

    vis = (neg.astype(np.uint8) * 255)

    if vis.ndim == 3:
        vis = vis.any(axis=-1).astype(np.uint8) * 255

    Image.fromarray(vis).save(
        out_dir / f"{inp_path.stem}_neg_{frac*100:.2f}.png"
    )

    print(f"{inp_path.stem}: {frac*100:.2f}% negative pixels")
