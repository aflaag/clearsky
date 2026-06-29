#!/usr/bin/env python3
import numpy as np
from pathlib import Path

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

    a = inp.ravel()
    b = tgt.ravel()

    c = np.corrcoef(a, b)[0, 1]
    corrs.append(c)

corrs = np.array(corrs)

print(f"mean corr : {corrs.mean():.5f}")
print(f"min  corr : {corrs.min():.5f}")
print(f"max  corr : {corrs.max():.5f}")
