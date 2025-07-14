import glob, os
import warnings

import numpy as np
import pandas as pd

from astropy.table import Table
from IPython.display import display
from tqdm import tqdm

import config as cfg

warnings.simplefilter(action='ignore', category=FutureWarning)

#=== Combine individual spectrum outputs and stack them in one table ==========

# list all FITS files produced after fitting 
# and select columns to include for every entry in the table
fits_all = sorted(glob.glob(os.path.join(cfg.fpath_outputs, 
                                         "*-line_props.fits")))
cols_props = ['z', 'z_siglo', 'z_sigup',
              'fwhm', 'fwhm_siglo', 'fwhm_sigup',
              'redchisq']
cols_other = ['flux_norm', 'npar', 'nobs']
rows = []

# get spectra filenames and fitting iterations from the output filenames
spectra, iterations = [], []
for fits in fits_all:
    idx_srcid = np.argwhere([f.replace('b', '').isdigit() \
                             for f in fits.split('-')[1:]]).item() + 1
    idx_iter = np.argwhere(['iter' in f for f in fits.split('-')]).item()
    spec = '-'.join(fits.split('-')[idx_srcid+1:idx_iter-1]) + '-' +\
           fits.split('-')[idx_iter+1] # spec rootname + fitted lines
    spectra.append(spec)
    iterations.append(int(fits.split('-')[idx_iter].replace('iter', '')))

# pick the latest iteration of a fit for each fit (highest 'niter')
spectra_unique = np.unique(spectra)
n_ids = len(spectra_unique)
fit_iterations = dict(zip(spectra_unique, [[] for i in range(n_ids)]))
fit_files = dict(zip(spectra_unique, [[] for i in range(n_ids)]))
for i, spec in enumerate(spectra):
    fit_iterations[spec].append(iterations[i])
    fit_files[spec].append(fits_all[i])

idxs_latest_iter = [iters.index(np.max(iters)) \
                        for iters in list(fit_iterations.values())]
fits_latest = [files[idxs_latest_iter[i]] \
                for i, files in enumerate(list(fit_files.values()))]

df_spec = Table.read(cfg.fpath_spec)

# loop over the individual outputs per spectrum
for fpath in tqdm(fits_latest):
    columns, values = [], []
    fname = fpath.split('/')[-1]
    
    # get current object ID and spectrum grating from filenames
    idx_srcid = np.argwhere([f.replace('b', '').isdigit() \
                             for f in fname.split('-')[1:]]).item() + 1
    idx_iter = np.argwhere(['iter' in f for f in fname.split('-')]).item()
    spec_name = '-'.join(fname.split('-')[idx_srcid+1:idx_iter-1])
    
    prog_id = '-'.join(fname.split('-')[:idx_srcid+1])
    is_prog_id = [v == prog_id for v in df_spec['PROG_ID'].data]
    obj_idx = df_spec['Index'][is_prog_id][0]
    grating = fname.split('-')[idx_iter-1]
    
    columns += ['Index', 'PROG_ID', 'grating', 'file']
    values += [obj_idx, prog_id, grating, spec_name]
    
    # load a table
    tab = Table.read(fpath)
    
    # take quantities of interest
    cols_SN = [c for c in tab.colnames if 'sn' in c.lower()]
    cols_flux = [c for c in tab.colnames \
                 if ('flux' in c.lower()) & (not 'norm' in c.lower())]
    cols_ew = [c for c in tab.colnames \
                 if ('ew' in c.lower())]
    vals_SN = [float(tab[c].item()) for c in cols_SN]
    vals_flux = [float(tab[c].item()) for c in cols_flux]
    vals_ew = [float(tab[c].item()) for c in cols_ew]
    vals_other = [float(tab[c].item()) for c in cols_other]
    
    # add props columns if a line measurement is present
    if len(cols_SN) > 0:
        keys = [c.replace('_SN', '') for c in tab.colnames if 'sn' in c.lower()]
        key = '_'.join(keys)
        cols_props_mod = [f'{key}_{c}' for c in cols_props]
        vals_props = [float(tab[c].item()) for c in cols_props]
    else:
        cols_props_mod = []
        vals_props = []
    
    columns += cols_SN + cols_flux + cols_ew + cols_props_mod + cols_other
    values += vals_SN + vals_flux + vals_ew + vals_props + vals_other
    
    # save pandas DFs for each file with outputs
    row = pd.DataFrame(data=[values], columns=columns)
    rows.append(row)

df_allspec = pd.concat(rows, axis=0)

# normalize fluxes by the same factor (currently factors can vary)
df_allspec.reset_index(drop=True, inplace=True)

flux_norm = np.log10(df_allspec['flux_norm'].values)
flux_norm_max = np.max(flux_norm)
factors_renorm = flux_norm_max - flux_norm
cols_flux = [c for c in df_allspec.columns \
             if ('flux' in c.lower()) & (not 'norm' in c.lower())]
df_allspec[cols_flux] = df_allspec[cols_flux] * 10**factors_renorm[:,None]
df_allspec['flux_norm'] = flux_norm_max
df_allspec = df_allspec.copy().groupby('file', as_index=False).first()

#=== find unresolved lines, split them and assign all flux to the first line 
# (others lines are NaNs)
cols_flux = [c for c in df_allspec.columns if c.endswith('_flux')]
lines_unresolved = ['_'.join(c.split('_')[:-1]) \
                    for c in cols_flux if c.count('_') > 2]
cols_sets_unresolved = [[c for c in df_allspec.columns if c.startswith(l)] \
                        for l in lines_unresolved]

for cols_orig, line_str_orig in zip(cols_sets_unresolved, lines_unresolved):
    lines = ['_'.join(cols_orig[0].split('_')[i:i+2]) \
             for i in range(0, len(cols_orig[0].split('_')[:-1]), 2)]
    cols_new = [c.replace(line_str_orig, lines[0]) for c in cols_orig]
    
    # rename original combined SN, EW columns to the first line's columns
    df_allspec.rename(
        columns=dict(zip(cols_orig, cols_new)),
        inplace=True,
    )
    df_allspec.reset_index(drop=True, inplace=True)
    
    # Remove duplicate columns in df_allspec
    df_allspec = df_allspec.loc[:, ~df_allspec.columns.duplicated()]
    
    # add columns for the remaining originally combined lines and fill with NaNs
    cols_other_new = [[c.replace(line_str_orig, line) for c in cols_orig] \
                      for line in lines[1:]]
    col_index = df_allspec.columns.get_loc(cols_new[-1])
    
    for cols_line in cols_other_new:
        # Insert the new columns into df_allspec at the specified column index
        df_newcols = pd.DataFrame(np.full((df_allspec.shape[0], 
                                        len(cols_line)), np.nan), 
                                columns=cols_line)
        df_allspec = pd.concat(
            [df_allspec.iloc[:, :col_index+1], df_newcols, df_allspec.iloc[:, col_index+1:]],
            axis=1
        )
        col_index += 1

# Remove duplicate columns in df_allspec
df_allspec = df_allspec.loc[:, ~df_allspec.columns.duplicated()]
df_allspec['file'] = df_allspec['file'].apply(lambda row: row+'.spec.fits')

# save the results for every spectrum per line
df_allspec.to_csv(cfg.fpath_catalog_flux, index=False)
