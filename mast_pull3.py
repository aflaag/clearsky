from astroquery.mast import Observations
from collections import defaultdict
from astropy.table import vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

INSTRUMENTS_OK = ["ACS/WFC", "ACS/HRC", "WFPC2", "WFPC2/WFC", "WFPC2/PC", "WFC3/UVIS"]
REQUIRED_FILTERS = ["F435W", "F606W", "F814W"]
MATCH_RADIUS_DEG = 0.01  # ~36 arcsec: osservazioni entro questa distanza = stesso field

def query_target_colocated(target_name):
    print(f"\n{'='*50}")
    print(f"Target: {target_name}")

    obs = Observations.query_criteria(
        obs_collection=["HST", "HLA"],
        dataproduct_type="image",
        objectname=target_name,
        radius="0.2 deg",
        dataRights="PUBLIC",
        calib_level=[2, 3],
    )

    if len(obs) == 0:
        print("  Nessuna osservazione")
        return None

    mask_instr = np.isin(obs["instrument_name"], INSTRUMENTS_OK)
    obs = obs[mask_instr]

    # ─── Raggruppa per filtro ─────────────────────────────────────────────────
    obs_by_filter = {}
    for filt in REQUIRED_FILTERS:
        mask = np.array([filt in str(f) for f in obs["filters"]])
        obs_by_filter[filt] = obs[mask]

    if any(len(v) == 0 for v in obs_by_filter.values()):
        missing = [f for f, v in obs_by_filter.items() if len(v) == 0]
        print(f"  ✗ Mancano bande: {missing}")
        return None

    # ─── Trova terzetti co-locati ─────────────────────────────────────────────
    # Prendi ogni osservazione del filtro "ancora" (F606W, di solito il più raro)
    # e cerca le corrispondenti negli altri due filtri entro MATCH_RADIUS_DEG

    anchor_filter = min(obs_by_filter, key=lambda f: len(obs_by_filter[f]))
    other_filters = [f for f in REQUIRED_FILTERS if f != anchor_filter]

    print(f"  Anchor filter: {anchor_filter} ({len(obs_by_filter[anchor_filter])} obs)")

    matched_triplets = []

    for anchor_row in obs_by_filter[anchor_filter]:
        ra0  = float(anchor_row["s_ra"])
        dec0 = float(anchor_row["s_dec"])
        coord0 = SkyCoord(ra=ra0*u.deg, dec=dec0*u.deg)

        triplet = {anchor_filter: anchor_row}
        ok = True

        for filt in other_filters:
            ras  = np.array(obs_by_filter[filt]["s_ra"],  dtype=float)
            decs = np.array(obs_by_filter[filt]["s_dec"], dtype=float)
            coords = SkyCoord(ra=ras*u.deg, dec=decs*u.deg)
            seps = coord0.separation(coords).deg

            best_idx = np.argmin(seps)
            if seps[best_idx] <= MATCH_RADIUS_DEG:
                triplet[filt] = obs_by_filter[filt][best_idx]
            else:
                ok = False
                break  # questo anchor non ha match in tutti i filtri

        if ok:
            matched_triplets.append(triplet)

    print(f"  Terzetti co-locati trovati: {len(matched_triplets)}")

    if not matched_triplets:
        return None

    # Stampa i terzetti trovati
    for i, triplet in enumerate(matched_triplets[:5]):
        print(f"\n  Terzetto {i+1}:")
        for filt, row in triplet.items():
            print(f"    {filt}: obsid={row['obsid']}  "
                  f"ra={float(row['s_ra']):.4f}  dec={float(row['s_dec']):.4f}  "
                  f"instr={row['instrument_name']}")

    return matched_triplets


# ─── Cerca e scarica ──────────────────────────────────────────────────────────
GOOD_TARGETS = ["M42"]#, "GOODS-N", "GOODS-S", "COSMOS", "UDF", "NGC-4321", "NGC-3370", "M51"]

found_triplets = {}
for target in GOOD_TARGETS:
    result = query_target_colocated(target)
    if result:
        found_triplets[target] = result
        break  # fermati al primo target che funziona

if not found_triplets:
    print("Nessun target con terzetto co-locato trovato")
else:
    CHOSEN = list(found_triplets.keys())[0]
    triplet = found_triplets[CHOSEN][0]  # primo terzetto trovato
    print(f"\nScarico terzetto da: {CHOSEN}")

    to_download = []
    for filt, obs_row in triplet.items():
        products = Observations.get_product_list(obs_row["obsid"])
        drz = Observations.filter_products(
            products,
            extension="fits",
            productType="SCIENCE",
            productSubGroupDescription=["DRZ", "DRC"],
        )
        if len(drz) == 0:
            drz = Observations.filter_products(
                products,
                extension="fits",
                productType="SCIENCE",
            )
        to_download.append(drz[:1])
        print(f"  {filt}: {obs_row['obsid']} → 1 file")

    all_products = vstack(to_download)
    manifest = Observations.download_products(
        all_products,
        download_dir=f"./hst_rgb/{CHOSEN}",
    )
    print(manifest["Local Path", "Status"])
