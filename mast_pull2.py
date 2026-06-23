from astroquery.mast import Observations
from astropy.table import vstack
import numpy as np

import logging
from astroquery import log

log.setLevel(logging.DEBUG)

# ─── 1. Query osservazioni HST imaging pubbliche ──────────────────────────────
target_name = "M42"

obs = Observations.query_criteria(
    objectname=target_name,
    obs_collection="HST",
    dataproduct_type="image",
    intentType="science", # Solo dati scientifici, niente calibrazioni
    calib_level=[2,3]
)

print(f"Trovate {len(obs)} osservazioni")

# ─── 2. Raggruppare per target e trovare quelli con ≥3 filtri ────────────────
from astropy.table import Table

# Ogni riga ha: target_name, filters, obsid
targets = {}
for row in obs:
    t = row["target_name"]
    f = row["filters"]
    oid = row["obsid"]
    
    # 1. Salta l'iterazione se il target o il filtro sono mancanti ("mascherati")
    if np.ma.is_masked(t) or np.ma.is_masked(f):
        continue
        
    # 2. Convertiamo esplicitamente in stringa per sicurezza
    t_str = str(t)
    f_str = str(f)
    
    if t_str not in targets:
        targets[t_str] = {"filters": set(), "obsids": []}
        
    targets[t_str]["filters"].add(f_str)
    targets[t_str]["obsids"].append(oid)

# Tieni solo i target con ≥3 bande distinte (= potenzialmente "colorabile")
multiband_targets = {
    t: v for t, v in targets.items()
    if len(v["filters"]) >= 3
}

print(f"Target con ≥3 bande: {len(multiband_targets)}")

# ─── 3. Selezionare le osservazioni corrispondenti ───────────────────────────
good_obsids = []
for t, v in multiband_targets.items():
    good_obsids.extend(v["obsids"])

# Filtra la tabella originale
mask = np.isin(obs["obsid"], good_obsids)
obs_multiband = obs[mask]

# ─── 4. Ottenere i prodotti DRZ (drizzled, calibrati) ───────────────────────
products = Observations.get_product_list(obs_multiband)

fits_science = Observations.filter_products(
    products,
    extension="fits",
    productType="SCIENCE",
    productSubGroupDescription=["DRZ", "DRC"],  # DRC = ACS distortion-corrected
    mrp_only=False,
)

print(f"Prodotti FITS DRZ/DRC: {len(fits_science)}")

# ─── 5. Download ─────────────────────────────────────────────────────────────
manifest = Observations.download_products(
    fits_science,
    download_dir="./hst_multiband",
)
