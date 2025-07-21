import os
import astropy.units as au
import numpy as np
import pandas as pd

from astropy.cosmology import Planck18
from scipy.integrate import simpson

import config as cfg
from spectrum import *

cosmo = Planck18

# user input
w1, w2 = 1350, 1800 # [Angstrom] UV window for M_UV
download_data = True # get DJA spectra from cfg.PATH_AWS or cfg.PATH_LOCAL?
verbose_download = False

# load prism catalog
fpath = os.path.join(cfg.fpath_project, 'sample_spec_prism.csv')
df_prism = pd.read_csv(fpath)

# load all DJA spectra in memory
sample_info = {
    'fnames': df_prism.file.values, 
    'roots': df_prism.root.values, 
    'idxs': df_prism.Index.values, 
    'obj_ids': df_prism.PROG_ID.values,
    'z_input': df_prism.z.values,
    'flux_norm': None,
}

specSample = SpectrumSampleFit(
    **sample_info, download_data=download_data, verbose=verbose_download
) # load all spectra into memory

#=== calculate AB magnitudes
# redshift and lum. distance
z = specSample.z_input
dL = cosmo.luminosity_distance(z).to(au.pc).value # luminosity distance [pc]

# set up the UV window for galaxies in [w1, w2]
specSample.update_fit_window([[w1, w2]])
is_UV = specSample.in_window.copy()

# integrate flux density f_lambda through a UV top-hat filter
flam_unit = au.erg / (au.cm**2 * au.s * au.AA)
equiv = au.spectral_density(0.5*(w1 + w2) * (1 + z) * au.AA)
Flam_UV = np.array([
    simpson(
        specSample.y[i][is_UV[i]], 
        x=specSample.x[i][is_UV[i]]) / ((w2 - w1) * (1 + z[i])) / specSample.flux_norm[i].item() \
            if is_UV[i].any() else np.nan\
            for i in range(specSample.n_obj)
]) # integrate F_lambda in the UV window
Fnu_UV = (Flam_UV * flam_unit).to(au.Jy, equivalencies=equiv).value

# calculate magnitudes using the integrated flux Fnu
K = -2.5 * np.log10((1 + z)) # K-correction
m_UV_AB = -2.5 * np.log10(Fnu_UV) + 8.90 # apparent mag
DM = 5.0 * (np.log10(dL) - 1.0) # distance modulus
M_UV_AB = m_UV_AB - DM - K # absolute mag

# calculate median SNR in the UV window
sn50_UV = np.array([
    np.median(specSample.y[i][is_UV[i]] / specSample.ye[i][is_UV[i]])\
              if is_UV[i].any() else np.nan\
              for i in range(specSample.n_obj)]
)

# table of magnitudes
data = np.c_[
    specSample.idxs, 
    specSample.obj_ids, 
    specSample.fnames, 
    specSample.z_input, 
    sn50_UV, 
    m_UV_AB, 
    M_UV_AB,
]
df_ABmag = pd.DataFrame(
    data, columns=['Index', 'PROG_ID', 'file', 'z', 'sn50_UV', 'm_UV', 'M_UV']
)
fpath = os.path.join(cfg.fpath_project, 'magnitudes_UV_AB.csv')
df_ABmag.to_csv(fpath, index=False)
