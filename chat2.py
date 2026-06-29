#!/usr/bin/env python3
import numpy as np
from pathlib import Path
from PIL import Image

dataset = Path("dataset")
inp_dir = dataset / "input" / "npy"
tgt_dir = dataset / "target" / "npy"

out_dir = Path("residual_preview")
out_dir.mkdir(exist_ok=True)

for i, inp_path in enumerate(sorted(inp_dir.glob("*.npy"))[:50]):
    tgt_path = tgt_dir / inp_path.name
    if not tgt_path.exists():
        continue

    inp = np.load(inp_path).astype(np.float32)
    tgt = np.load(tgt_path).astype(np.float32)

    res = np.clip(inp - tgt, 0, None)

    p99 = np.percentile(res, 99.5)
    if p99 > 0:
        res = np.clip(res / p99, 0, 1)

    img = (res * 255).astype(np.uint8)
    Image.fromarray(img).save(out_dir / f"{inp_path.stem}.png")
