from astropy.io import fits
from pathlib import Path

input_dir = Path("./assets/inputs")

fits_files = sorted(
    list(input_dir.glob("*.fits"))
    + list(input_dir.glob("*.fit"))
    + list(input_dir.glob("*.FITS"))
)

for f in fits_files:
    with fits.open(f) as hdul:
        h = hdul[0].header
        orientat = h.get("ORIENTAT")
        cd1_1    = h.get("CD1_1")
        cd2_1    = h.get("CD2_1")
        shape    = hdul[0].data.shape if hdul[0].data is not None else "N/A"
    print(f"{f.name} | shape={shape} | ORIENTAT={orientat} | CD1_1={cd1_1} | CD2_1={cd2_1}")
