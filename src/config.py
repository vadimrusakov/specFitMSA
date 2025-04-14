import os
import config_lines as cl
import numpy as np

BAD_VALUE = -999.0

# paths to spectra
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/' # DJA AWS
PATH_AWS = BASE_URL + '{root}/{file}'
PATH_LOCAL = '../data/dja_spectra/{file}'

#=== User inputs ==========================================================
#=== fitting inputs
label_project = 'example'
fname_spec = 'sample_spec.csv' # .csv file with a 'file' column

step_method = 'NUTS' # PyMC sampling method
#sn_thresh = 2.0 # S/N threshold for line detection (either fit or skip)

set_hwhm_kms = 3500 # fit window: lines +/- hwhm [km/s] (gratings)
fit_exclude_hwhm_kms = 100 # exclude window: unfitted lines +/-hwhm [km/s] (gratings)
#set_hwhm_kms = 20000 # (PRISM)
#fit_exclude_hwhm_kms = 5000 # (PRISM)

# lines to fit
#line_keys = ['SII_4070', 'SII_4078', 'Hd_4103']
line_keys = cl.cols_tem_diag + cl.cols_den_diag + cl.cols_hydrogen
line_keys = np.unique(line_keys)

#=== fitting outputs
save_trace = False # save PyMC posterior trace or not


#=== paths to be used by the code ========================================
fdir_sample = f'../data/project_{label_project}' # sample directory
fdir_outputs = os.path.join(fdir_sample, f"pymc_outputs-step{step_method}")
fdirs = [fdir_sample, fdir_outputs]
for fdir in fdirs:
    if not os.path.exists(fdir):
        os.makedirs(fdir)

fpath_spec = os.path.join(fdir_sample, fname_spec) # path to spec_list
