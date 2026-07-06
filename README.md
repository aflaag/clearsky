# clearsky

## TODO

- [ ] levare file inutili ed organizzare cartella
- [ ] sistemare branches
- [ ] pulire `astro_stretch.py` e `make_dataset.py` (ed eventualmente altro) da rimanenze denoising
- [ ] pulire il codice da commenti + refactoring

## Roadmap

### DPPM

- [ ] Star Removal
  - [x] dataset
  - [ ] DPPM trained

- [ ] Image Restoration
  - [x] dataset
  - [ ] DDPM trained

- [ ] Super Resolution
  - [x] dataset
  - [ ] DDPM trained

### Model merging

- [ ] creare $\theta_0$ trainando un DDPM su `dataset_sr/target + datset_ir/target + dataset_su/target` con identità e prob $p$ che sporca la condizione
- [ ] fine-tuning DDPM SR trainando su `dataset_sr`
- [ ] fine-tuning DDPM IR trainando su `dataset_ir`
- [ ] fine-tuning DDPM SU trainando su `dataset_su`
- [ ] creazione modello con task arithmetic
- [ ] creazione modello TIES
- [ ] creazione modello DARE
- [ ] creazione modello DARE-TIES
- [ ] creazione modello RegMean

### Report

TODO: ?

- idee:
  - spiegare processo creazione dataset per ogni dataset
  - spiegare arch ddpm
  - forse ddpm star removal performa meglio di starnet!!!
