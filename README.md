# specFitMSA

Spectra fitting pipeline for fitting line and continuum emission in JWST NIRSpec/MSA spectra. Uses the
[msaexp](https://github.com/gbrammer/msaexp) package to extract MSA spectra from the
[Dawn JWST Archive](https://dawn-cph.github.io/dja/) accessed at: 

https://s3.amazonaws.com/msaexp-nirspec/extractions/.

## Implementations

There are two line fitting implementations:

* **`scipy.optimize`** — chi-square minimization with [jax](https://github.com/google/jax)-computed gradients.
    * Default: local search (`trust-constr`)
    * Optional: global search (`differential_evolution`) followed by local search

    ```bash
    python src/fit_jax.py all
    python src/fit_jax.py <obj_id_user>
    python src/fit_jax.py obj_ids.txt
    ./run_fit_parallel.sh <n_proc> <COVARIANCE_FLAG>
    ```

* **[PyMC](https://doi.org/10.5281/zenodo.4603970)** — Bayesian modelling.

    ```bash
    python src/fit_pymc.py <obj_id_user> <n_iter> <nsteps> <ntune> <ncores>
    python src/fit_pymc.py all <n_iter> <nsteps> <ntune> <ncores>
    ```

The fitting is configured in [`src/config.py`](src/config.py). The code takes input information about the spectroscopic sample from [`data/project_example/catalog-flux.csv`](data/project_example/catalog-flux.csv).

## Requirements

### Common

* [msaexp](https://github.com/gbrammer/msaexp)
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [matplotlib](https://matplotlib.org/)
* [pandas](https://pandas.pydata.org/)
* [astropy](https://www.astropy.org/)
* [corner](https://corner.readthedocs.io/)

### `scipy.optimize` / jax implementation ([`fit_jax.py`](src/fit_jax.py))

* [jax](https://github.com/google/jax)
* [tqdm](https://tqdm.github.io/)

### PyMC implementation ([`fit_pymc.py`](src/fit_pymc.py))

* [pymc](https://www.pymc.io/)
* [pytensor](https://pytensor.readthedocs.io/)
* [arviz](https://python.arviz.org/)
