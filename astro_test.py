from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch, PowerStretch

# 1. Carica i dati FITS
hdul = fits.open("../Download/color_hst_05461_01_wfpc2_f814w_f336w_wf_sci.fits")
data = hdul[0].data

# Gestione dei NaN sui dati grezzi
data = np.nan_to_num(data)

# 2. Normalizzazione e Stretch Canale per Canale (per preservare i colori)
data_normalized = np.zeros_like(data)

for i in range(3):
    # Applichiamo AsinhStretch. Il parametro 'a' controlla la non-linearità.
    # Valori più piccoli di 'a' rendono lo stretch più aggressivo sui dettagli deboli.
    norm = ImageNormalize(
        data[i],
        interval=ZScaleInterval(),
        # stretch=AsinhStretch(a=0.2) 
        stretch=PowerStretch(0.5)
    )
    # Normalizza il singolo canale e lo inserisce nella matrice finale
    data_normalized[i] = norm(data[i])

# 3. Taglia eventuali valori fuori dal range [0, 1] per sicurezza
data_normalized = np.clip(data_normalized, 0, 1)

# 4. Ruota lo shape da (3, 8500, 8500) a (8500, 8500, 3) per Matplotlib
data_rgb = np.transpose(data_normalized, (1, 2, 0))

# 5. Mostra il risultato
plt.figure(figsize=(12, 12))
plt.imshow(data_rgb, origin="lower")
plt.axis('off')
plt.show()
