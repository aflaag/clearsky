# Solo il FITS (per vedere i valori raw)
# python inspect_fits.py --fits assets/inputs/color_hst_10775_05_acs_wfc_f814w_f606w_sci.fits

# Con anche input NPY e starless PNG, per il confronto completo
# python inspect_fits.py \
#   --fits     assets/inputs/color_hst_10775_05_acs_wfc_f814w_f606w_sci.fits \
#   --npy      assets/outputs-npy/color_hst_10775_05_acs_wfc_f814w_f606w_sci.npy \
#   --starless assets/outputs-starless/color_hst_10775_05_acs_wfc_f814w_f606w_sci.png

# python debug_pair.py \
#   --npy      assets/outputs-npy/color_hst_10775_05_acs_wfc_f814w_f606w_sci.npy \
#   --starless assets/outputs-starless/color_hst_10775_05_acs_wfc_f814w_f606w_sci.png

python inspect_bg.py \
  --npy      assets/outputs-npy/color_hst_10775_05_acs_wfc_f814w_f606w_sci.npy \
  --starless assets/outputs-starless/color_hst_10775_05_acs_wfc_f814w_f606w_sci.tif
