import glob, os
import warnings

import numpy as np
import pandas as pd

from astropy.table import Table
from multiprocessing import get_context, cpu_count
from tqdm import tqdm

import config as cfg

warnings.simplefilter(action='ignore', category=FutureWarning)

n_procs = 32 # number of processes for multiprocessing (reduced due to low available memory)

cols_props = ['z', 'z_siglo', 'z_sigup',
              'fwhm', 'fwhm_siglo', 'fwhm_sigup',
              'redchisq']
props_perline = ['cont_wmean_SN']
cols_other = ['flux_norm', 'npar', 'nobs']

# properties for double-component (_2) line fits
props_2comp = ['center', 'center_sigup', 'center_siglo',
               'fwhm', 'fwhm_sigup', 'fwhm_siglo',
               'decay', 'decay_sigup', 'decay_siglo']

#=== Combine individual spectrum outputs and stack them in one table ==========
# list all FITS files produced after fitting 
# and select columns to include for every entry in the table
fits_all = sorted(glob.glob(os.path.join(cfg.fpath_outputs, "*-line_props.fits")))

df_sample = pd.read_csv(cfg.fpath_spec)
files_sample = df_sample.file.to_numpy()

fits_all = [
    f for f in fits_all 
    if any([f"-{fsamp.replace('.spec.fits', '')}-" in f for fsamp in files_sample])
]

# get spectra filenames and fitting iterations from the output filenames
spectra, iterations = [], []
for fits in fits_all:
    fits = fits.replace('_-', '_')
    idx_srcid = np.argwhere([f.replace('b', '').isdigit() \
                             for f in fits.split('-')[1:]])[0].item() + 1
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

files_latest = np.unique(
    ['-'.join(f.split('-')[:-1]) for f in fit_files.keys()])

print(f"Found {len(files_latest)} / {len(files_sample)} spectra (relative to the project sample).")

# Function to process a single file
def process_file(args):
    """Process a single FITS file and return a DataFrame row."""
    fpath, prog_id, groupid = args
    
    columns, values = [], []
    fname = fpath.split('/')[-1].replace('_-', '_')

    # get current object ID and spectrum grating from filenames
    idx_srcid = np.argwhere([f.replace('b', '').isdigit() \
                             for f in fname.split('-')[1:]])[0].item() + 1
    idx_iter = np.argwhere(['iter' in f for f in fname.split('-')]).item()
    spec_name = '-'.join(fname.split('-')[idx_srcid+1:idx_iter-1])

    grating = fname.split('-')[idx_iter-1]
    #prog_id = '-'.join(fname.split('-')[:idx_srcid+1])
    #try:
    #    is_prog_id = [v.decode() == prog_id for v in tab_ext_progid]
    #    groupid = tab_ext_data[is_prog_id][0]
    #except IndexError:
    #    print(f" -- Warning: PROG_ID {prog_id} not found in extractions table.")
    
    columns += ['Index', 'PROG_ID', 'grating', 'file']
    values += [groupid, prog_id, grating, spec_name+'.spec.fits']
    
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
        keys = [c.replace('_SN', '') for c in tab.colnames
                if ('sn' in c.lower()) & ('cont_wmean' not in c.lower())]
        key = '_'.join(keys)
        cols_props_mod = [f'{key}_{c}' for c in cols_props]
        vals_props = [float(tab[c].item()) for c in cols_props]
        cols_other_mod = [f"{key}_{c}" if c != 'flux_norm' 
                          else c for c in cols_other]
    else:
        cols_props_mod = []
        vals_props = []
        cols_other_mod = cols_other

    cols_props_per_line_all = [
        [c for c in tab.colnames if c.endswith(k)] for k in props_perline]
    cols_props_per_line_all = [
        item for sublist in cols_props_per_line_all for item in sublist]
    vals_props_per_line_all = [
        float(tab[c].item()) for c in cols_props_per_line_all]

    # extract double-component (_2) line properties if present
    cols_2comp = [c for c in tab.colnames
                  if '_2_' in c
                  and any(c.endswith(f'_{p}') for p in props_2comp)]
    vals_2comp = [float(tab[c].item()) for c in cols_2comp]

    columns += cols_SN + cols_flux + cols_ew + cols_props_mod +\
        cols_other_mod + cols_props_per_line_all + cols_2comp
    values += vals_SN + vals_flux + vals_ew + vals_props + vals_other +\
        vals_props_per_line_all + vals_2comp

    # return dictionary for this file (more efficient than DataFrame)
    return dict(zip(columns, values))

# Process files in parallel
n_processes = min(cpu_count(), n_procs)  # set max number of cores
print(f"Processing {len(fits_latest)} files using {n_processes} parallel processes...")

# arguments for multiprocessing (pass tab_ext to avoid serialization issues)
#args_list = [(fpath, tab_ext['GroupID'].data, tab_ext['PROG_ID'].data)
#             for fpath in fits_latest]
fnames = [fpath.split('/')[-1].replace('_-', '_') for fpath in fits_latest]

idxs_srcid = [
    np.argwhere([f.replace('b', '').isdigit() for f in fname.split('-')[1:]]
                )[0].item() + 1
    for fname in fnames]
idxs_iter = [
    np.argwhere(['iter' in f for f in fname.split('-')]).item()
    for fname in fnames]
spec_names = np.array([
    '-'.join(fname.split('-')[idx_srcid+1:idx_iter-1])+'.spec.fits'
    for fname, idx_srcid, idx_iter in zip(fnames, idxs_srcid, idxs_iter)])
prog_ids = np.array([
    '-'.join(fname.split('-')[:idx_srcid+1])
    for fname, idx_srcid in zip(fnames, idxs_srcid)])
df_files = pd.DataFrame({'file': spec_names, 'PROG_ID': prog_ids})
object_idxs = pd.merge(df_files, df_sample, 
                    on='file', how='left')['Index'].to_numpy()
args_list = list(zip(fits_latest, prog_ids, object_idxs))

#for args in args_list:
#    if args[1] == 'JADES-GN-3608':
#        print(args[0].split('/')[-1], args[1], args[2])

with get_context('fork').Pool(processes=n_processes) as pool: # parallel processing
    rows = list(tqdm(pool.imap(process_file, args_list), 
                     total=len(fits_latest))) 

#rows = [process_file(args) for args in tqdm(args_list)] # sequential

# create DataFrame from output dictionaries of columns and values
df_allspec = pd.DataFrame(rows)

# normalize fluxes by the same factor (currently factors can vary)
df_allspec.reset_index(drop=True, inplace=True)

flux_norm = np.asarray(df_allspec['flux_norm'])
flux_norm = np.log10(flux_norm) if np.any(flux_norm > 1e2) else flux_norm
flux_norm_max = 20  # np.max(flux_norm)
factors_renorm = flux_norm_max - flux_norm
cols_flux = [c for c in df_allspec.columns \
             if ('flux' in c.lower()) & (not 'norm' in c.lower())]
df_allspec[cols_flux] = df_allspec[cols_flux] * 10**factors_renorm[:,None]
df_allspec['flux_norm'] = flux_norm_max
df_allspec = df_allspec.copy().groupby('file', as_index=False).first()

# find columns for unresolved lines (e.g., OII_3727_OII_3829_flux) 
# and add individual columns, if they don't already exist 
# (e.g., OII_3727_flux, OII_3829_flux) filled with NaN values
cols_flux = [c for c in df_allspec.columns if c.endswith('_flux')]
lines_unresolved = ['_'.join(c.split('_')[:-1]) \
                    for c in cols_flux 
                    if (c.count('_') > 2) & ('cont_wmean' not in c.lower())]
cols_sets_unresolved = [[c for c in df_allspec.columns 
                         if c.startswith(l) & ('cont_wmean' not in c.lower())] \
                        for l in lines_unresolved]

for cols_orig, line_str_orig in zip(cols_sets_unresolved, lines_unresolved):
    # record individual line names, handling component indices (_2, _3, etc.)
    parts = cols_orig[0].split('_')[:-1]  # Remove the property suffix (e.g., 'flux')
    lines = []
    i = 0
    while i < len(parts):
        # Check if next part is a component index (numeric)
        if i + 2 < len(parts) and parts[i + 2].isdigit():
            lines.append('_'.join(parts[i:i+3]))  # e.g., 'Ha_6565_2'
            i += 3
        else:
            lines.append('_'.join(parts[i:i+2]))  # e.g., 'Ha_6565'
            i += 2

    # Skip if only one line parsed (not actually an unresolved blend,
    # e.g., single line with additional component index like Ha_6565_2)
    if len(lines) <= 1:
        continue

    df_allspec.reset_index(drop=True, inplace=True)

    # Create NaN columns for all individual lines in the blend
    cols_all_new = [[c.replace(line_str_orig, line) for c in cols_orig]
                    for line in lines]

    col_index = df_allspec.columns.get_loc(cols_orig[-1])

    for cols_line in cols_all_new:
        # Only add columns that don't already exist
        cols_to_add = [c for c in cols_line if c not in df_allspec.columns]
        if cols_to_add:
            df_newcols = pd.DataFrame(np.full((df_allspec.shape[0],
                                            len(cols_to_add)), np.nan),
                                    columns=cols_to_add)
            df_allspec = pd.concat(
                [df_allspec.iloc[:, :col_index+1], df_newcols, df_allspec.iloc[:, col_index+1:]],
                axis=1
            )
            col_index += len(cols_to_add)

# Remove duplicate columns in df_allspec
df_allspec = df_allspec.loc[:, ~df_allspec.columns.duplicated()]
df_allspec['file'] = df_allspec['file'].apply(lambda row: row+'.spec.fits')

# save the results for every spectrum per line
df_allspec.to_csv(cfg.fpath_catalog_flux, index=False)
