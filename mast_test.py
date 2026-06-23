from astroquery.mast import Observations
import numpy as np
from collections import defaultdict

# ─── Configurazione ───────────────────────────────────────────────────────────
REQUIRED_FILTERS = ["F435W", "F606W", "F814W"]
INSTRUMENTS = ["ACS/WFC", "WFC3/UVIS"]

# ─── 1. Diagnosi: vediamo cosa trova MAST per F606W senza filtro strumento ────
print("=== Diagnosi F606W ===")
obs_f606w_all = Observations.query_criteria(
    objectname="M42",
    dataproduct_type="image",
    filters="F606W",
    dataRights="PUBLIC",
    intentType="science",
)
print(f"F606W senza filtro strumento: {len(obs_f606w_all)} osservazioni")
if len(obs_f606w_all) > 0:
    # Vedi quali strumenti compaiono
    instruments_found = set(obs_f606w_all["instrument_name"])
    print(f"Strumenti trovati: {instruments_found}")
    # Vedi la colonna filters esatta
    filters_found = set(obs_f606w_all["filters"])
    print(f"Valori esatti del campo filters: {filters_found}")
