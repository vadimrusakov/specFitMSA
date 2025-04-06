import os
import numpy as np

from grizli.utils import get_line_wavelengths

import routines_spec as rspec

BAD_VALUE = -999.0
gratings = rspec.GRATINGS_MHRES

# label of the current sample (used in filenames and paths)
version_DJA_spec = 'v3'
label_catalog = f'highz_n_emitters_mhres_{version_DJA_spec}'

# filenames
fname_sample = 'sample_spec.csv' # spec sample filename
fname_lines_sn = 'sample-nirspec_lines-*-sn.csv'
fname_lines_flux = 'catalog_line_fluxes_mean.csv'
fname_lines_flux_Avcorr = f"{fname_lines_flux.split('.')[0]}_Avcorr.csv"
fname_abundances = 'catalog_abundances.csv'

# filnames: data tables and catalogs
version_DJA = '2025_01_13'
#version_DJA = '2024_09_27'
fpath_nirspec = os.path.join(os.getenv('data'), 'spectra/data/catalogs',
                          f'nirspec_tables-{version_DJA}')
fname_redshifts_manual = f'nirspec_redshifts_manual-{version_DJA}.fits'
fname_msaexp = f'nirspec_redshifts-{version_DJA}.fits'
fname_extractions = 'nirspec_extractions.fits'
fpath_redshifts_manual = os.path.join(fpath_nirspec, fname_redshifts_manual)
fpath_msaexp = os.path.join(fpath_nirspec, fname_msaexp)
fpath_extractions = os.path.join(fpath_nirspec, fname_extractions)

# PyMC sampler
step_method = 'NUTS'

# location of line fitting outputs
fdir_outputs = f"pymc_outputs-step{step_method}"
