from astroquery.mast import Observations

# 1. Impostiamo il target (puoi ometterlo se cerchi per coordinate o altro)
target_name = "M42"

print(f"Cercando osservazioni per {target_name}...")

# 2. Facciamo la query con i filtri specifici
obs_table = Observations.query_criteria(
    objectname=target_name,
    obs_collection="HST",
    dataproduct_type="image",
    intentType="science" # Solo dati scientifici, niente calibrazioni
)

print(f"Trovate {len(obs_table)} osservazioni.")

# 3. Selezioniamo solo le prime 3 osservazioni come test
# In uno scenario reale, vorresti filtrare per la colonna 'filters' 
# per assicurarti di avere un filtro rosso, uno verde e uno blu.
test_obs = obs_table[0:3]

# 4. Otteniamo la lista dei prodotti (i file effettivi) per queste osservazioni
products = Observations.get_product_list(test_obs)

# 5. Filtriamo i prodotti per avere SOLO i file FITS scientifici 
# (escludendo file di testo, anteprime jpg, log, ecc.)
filtered_products = Observations.filter_products(
    products,
    productType="SCIENCE",
    extension="fits"
)

print(f"Trovati {len(filtered_products)} file FITS scaricabili.")

# 6. Scarica i file (decommenta la riga sotto per scaricare davvero)
manifest = Observations.download_products(filtered_products, download_dir="./hst_images")
print("Download completato!")
