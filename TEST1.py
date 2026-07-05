import numpy as np
from pathlib import Path

INPUT_DIR = Path("assets/clean/pixelfix-npy")

npy_files = sorted(INPUT_DIR.glob("*.npy"))

if not npy_files:
    print(f"Nessun file trovato in {INPUT_DIR}")
    exit()

print(f"Trovati {len(npy_files)} file.\n")

for f in npy_files:
    arr = np.load(f)

    print(f"{f.name}")
    print(f"  shape : {arr.shape}")
    print(f"  dtype : {arr.dtype}")
    print(f"  min   : {arr.min():.10f}")
    print(f"  max   : {arr.max():.10f}")
    print(f"  mean  : {arr.mean():.10f}")
    print(f"  std   : {arr.std():.10f}")

    if np.issubdtype(arr.dtype, np.floating):
        finite = np.isfinite(arr)
        print(f"  finite: {finite.all()}")
        if not finite.all():
            print(f"    NaN : {np.isnan(arr).sum()}")
            print(f"    Inf : {np.isinf(arr).sum()}")

    print()
