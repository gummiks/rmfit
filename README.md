# rmfit

Fitting Rossiter McLaughlin Data.

# Installation


## From pip

```
pip install rmfit
```

## From source

```
git clone git@github.com:gummiks/rmfit.git
cd rmfit
python setup.py install
```

# Dependencies

- pyde, either (pip install pyde) or install from here: https://github.com/hpparvi/PyDE This package needs numba (try 'conda install numba' if problems).
- batman (pip install batman-package)
- emcee (pip install emcee)
- radvel (pip install radvel)
- corner (pip install corner)

## Note on PyDE
If issues with PyDE, it is best to clone it and install directly from GitHub.

```
git clone https://github.com/hpparvi/PyDE.git
cd PyDE
python setup.py install
```


# Quick start
See example notebooks in the notebook folder, which gives examples on fitting as-observed data from the literature (see data/examples folder)

# Citation
If you use this code, please cite:
- <a href='https://ui.adsabs.harvard.edu/abs/2022ApJ...931L..15S/abstract'>Stefansson et al. 2022</a>
- <a href='https://ui.adsabs.harvard.edu/abs/2011ApJ...742...69H/abstract'>Hirano et al. 2011<a/>
