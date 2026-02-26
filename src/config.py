import os
import config_lines as cl
import numpy as np

BAD_VALUE = -999.0

version_DJA = 'v4'
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/' # DJA AWS
PATH_AWS = BASE_URL + '{root}/{file}'
PATH_LOCAL = os.path.join( # path to spec on local disk
    os.getenv('data'), f'dja_spectra_{version_DJA}', '{file}') 

#=== User inputs ==========================================================
label_project = 'example'

#=== filenames
fname_spec = 'sample_spec.csv' # .csv file with a 'file' column
fname_catalog_flux = 'catalog_flux.csv' # flux catalog with fitted lines
step_method = 'NUTS' # PyMC sampling method

# medium/high-res spec setup
line_range_kms = 5e3 # line fitting region, FWHM [km/s]
line_fwhm_kms = 400 # line velocity FWHM [km/s]

# low-res spec setup (NIRSpec/PRISM)
#line_range_kms = 10e3 # low-res line fitting region, FWHM [km/s]
#line_fwhm_kms = 2e3 # line velocity FWHM [km/s]

# Global optimization settings
use_global_optimization = True  # use diff. evolution before local optimization
de_maxiter = 500  # max iter. for diff. evolution

# lines to fit
#line_keys = cl.cols_tem_diag + cl.cols_den_diag + cl.cols_hydrogen
#line_keys = cl.cols_high_ion
#line_keys = cl.lines_MgS + cl.lines_Ar + cl.lines_Ne +\
#            cl.lines_cnohe + cl.cols_tem_diag + cl.cols_den_diag + cl.cols_hydrogen
line_keys = ['Ha_6565']
line_keys = np.unique(line_keys)

broad_lines = [
    #'Hb_4861',
    #'Ha_6565',
]

# Lines with absorption component (multiplicative Gaussian optical depth)
abs_lines = [
    #'Hb_4861',
    #'Ha_6565'
]

# Fixed line ratios: {line_name: (reference_line, ratio_value)}
# line_amplitude = ratio_value * reference_line_amplitude
line_ratios = {
    # Storey & Zeippen 2000 (atomic physics): 3.05; 
    # Dojcinovic+2023 (250 SDSS Type II AGN): 3.049 ± 0.021
    'NII_6549': ('NII_6584', 1.0/3.05), 
    #'OIII_4959': ('OIII_5007', 1.0/2.98),  # OIII_4959 = 0.335 * OIII_5007
}

#=== fitting outputs
save_trace = False # save PyMC posterior trace or not


#=== paths to be used by the code ========================================
fpath_project = f'../data/project_{label_project}' # sample directory
fpath_outputs = os.path.join(fpath_project, f"pymc_outputs")
fdirs = [fpath_project, fpath_outputs]
for fdir in fdirs:
    if not os.path.exists(fdir):
        os.makedirs(fdir)

fpath_spec = os.path.join(fpath_project, fname_spec) # path to spec_list
fpath_catalog_flux = os.path.join(fpath_project, fname_catalog_flux) # flux catalog
