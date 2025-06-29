import os
import config_lines as cl
import numpy as np

BAD_VALUE = -999.0

# paths to spectra
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/' # DJA AWS
PATH_AWS = BASE_URL + '{root}/{file}'
PATH_LOCAL = '../data/dja_spectra/{file}'

#=== JWST/NIRSpec dispersers
GRATINGS_HRES = ['G140H', 'G235H', 'G395H']
GRATINGS_MRES = ['G140M', 'G235M', 'G395M']
GRATINGS_LRES = ['PRISM']
GRATINGS_ALL = GRATINGS_HRES + GRATINGS_MRES + GRATINGS_LRES
GRATINGS_MHRES = GRATINGS_MRES + GRATINGS_HRES

#=== User inputs ==========================================================
#=== fitting inputs
#label_project = 'example'
#label_project = 'broad_lines_rusakov25_mhres'
label_project = 'highz_n_emitters_v3'
date_DJA = '2025_03_31' # date of the local DJA spec catalog
gratings = GRATINGS_MHRES

#=== filenames
fname_spec = 'sample_spec.csv' # .csv file with a 'file' column
#fname_spec = 'sample-NIV_1483_NIV_1487_NIII_1750_emitters.csv'
fname_snr = 'SNR_lines.csv' # .csv file with line S/N columns
fname_flux = 'catalog-line_fluxes_all.csv' # line fluxes per spectrum

step_method = 'NUTS' # PyMC sampling method
sn_thresh = 3.0 # S/N threshold for line detection (either fit or skip)

# medium/high-res spec setup
line_range_kms = 5e3 # line fitting region, FWHM [km/s]
line_fwhm_kms = 400 # line velocity FWHM [km/s]

# low-res spec setup
#line_range_kms = 50e3 # low-res line fitting region, FWHM [km/s]
#line_fwhm_kms = 2e3 # line velocity FWHM [km/s]

# lines to fit
line_keys = ['OIII_4363']
#line_keys = cl.cols_tem_diag + cl.cols_den_diag + cl.cols_hydrogen
#line_keys = cl.cols_high_ion
line_keys = np.unique(line_keys)

#=== fitting outputs
save_trace = False # save PyMC posterior trace or not


#=== paths to be used by the code ========================================
fname_extractions = f'nirspec_extractions-{date_DJA}-groupid_final-latest_ver.fits'
fpath_nirspec = os.path.join(
    os.getenv('data'), 
    'spectra/data/catalogs', 
    f'nirspec_tables-{date_DJA}'
)

fdir_data = f'../data/project_{label_project}' # sample directory
fdir_outputs = os.path.join(fdir_data, f"pymc_outputs")
fdirs = [fdir_data, fdir_outputs]
for fdir in fdirs:
    if not os.path.exists(fdir):
        os.makedirs(fdir)

fpath_spec = os.path.join(fdir_data, fname_spec) # path to spec_list
fpath_lines_snr = os.path.join(fdir_data, fname_snr)
fpath_flux = os.path.join(fdir_data, fname_flux)
fpath_extractions = os.path.join(fpath_nirspec, fname_extractions)
