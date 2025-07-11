import glob, os, pickle, random, shutil, sys

import arviz as az
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing as mp
mp_start_method = 'spawn'
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from astropy.table import Table
from copy import deepcopy
from scipy.integrate import simpson

import routines_spec as rspec
import config as cfg
import config_lines as cl
from datetime import datetime

mpl.style.use('scientific')
#plt.rcParams['text.usetex'] = False

obj_id_user = sys.argv[1]
try:
    obj_id_user = float(obj_id_user)
except ValueError:
    pass
nsteps = int(sys.argv[2])
ntune = int(sys.argv[3])
ncores = int(sys.argv[4])

def get_line_wavel(w1, w2, lines_dict=[], exclude=[]):
    """
    Retrieve spectral line names and wavelengths within a specified range.
    
    Parameters:
    -----------
    w1 : float
        The lower bound of the wavelength range.
    w2 : float
        The upper bound of the wavelength range.
    lines_dict : dict, optional
        A dictionary where keys are line names (str) and values are wavelengths (float or array-like). Default is an empty list.
    exclude : list, optional
        A list of line names (str) to exclude from the results. Default is an empty list.
    Returns:
    --------
    names : list
        A list of line names (str) that fall within the specified wavelength range and are not in the exclusion list.
    waves : list
        A list of lists, where each inner list contains the wavelengths (float) of the corresponding line name that fall within the specified range.
    """
    
    names, waves = [], []
    for k, l in lines_dict.items():
        l = np.atleast_1d(l)
        cond = (l > w1) & (l < w2)
        if cond.any() & (k not in exclude):
            names.append(k)
            waves.append([*l[cond]])
    return names, waves

def mode_in_hdi(samples, bw_fct=2, hdi_prob=0.68, multimodal=False):
    """
    Calculate the mode of a distribution and determine its position within the 
    highest density interval (HDI).
    Parameters:
    -----------
    samples : array-like
        The samples from the distribution for which the mode and HDI are to be calculated.
    bw_fct : float, optional
        Bandwidth factor for the kernel density estimation (KDE). Default is 2.
    hdi_prob : float, optional
        The probability mass to be contained within the HDI. Default is 0.68.
    multimodal : bool, optional
        If True, allows for the possibility of a multimodal distribution and calculates 
        the HDI accordingly. Default is False.
    Returns:
    --------
    numpy.ndarray
        A 1D array containing:
        - The mode of the distribution.
        - The upper uncertainty (difference between the upper HDI bound and the mode).
        - The lower uncertainty (difference between the mode and the lower HDI bound).
    Notes:
    ------
    - If the mode is not within the HDI and `multimodal` is False, the function assumes 
      the presence of multiple peaks and recalculates the HDI using a multimodal approach.
    - If `multimodal` is True, the function identifies the HDI that contains the mode 
      or adjusts the HDI to include the mode.
    """
    
    # calculate the KDE and get the mode
    x_kde, y_kde = az.kde(samples, bw_fct=bw_fct)
    mode = x_kde[np.argmax(y_kde)]
    hdi = az.hdi(samples, hdi_prob=hdi_prob, multimodal=False)
    
    if multimodal:
        hdi_multi = az.hdi(samples, hdi_prob=0.68, multimodal=True)
        mode_in_hdi = [(mode > hdi_i[0]) & (mode < hdi_i[1])\
                       for hdi_i in hdi_multi]
        if np.any(mode_in_hdi):
            idx_hdi = np.argwhere(mode_in_hdi).item()
            hdi = hdi_multi[idx_hdi]
        else:
            #print(f'Mode {mode:.2e} not in HDI {hdi}, assuming multimodal distribution')
            idx_mode = np.where(np.isclose(abs(mode-hdi_multi), 0.0))[0].item()
            hdi = hdi_multi[idx_mode]
            #hdi[np.argmin(abs(mode - hdi))] = mode
    else:
        
        # check the mode is within the HDI
        # if not, assume more than one peak and use a multimodal hdi
        mode_in_hdi = (mode >= hdi[0]) & (mode <= hdi[1])
        if not mode_in_hdi:
            
            hdi_multi = az.hdi(samples, hdi_prob=0.68, multimodal=True)
            mode_in_hdi = [(mode > hdi_i[0]) & (mode < hdi_i[1])\
                        for hdi_i in hdi_multi]
            if np.any(mode_in_hdi):
                idx_hdi = np.argwhere(mode_in_hdi).item()
                hdi = hdi_multi[idx_hdi]
            else:
                #print(f'Mode {mode:.2e} not in HDI {hdi}, assuming multimodal distribution')
            
                hdi[np.argmin(abs(mode - hdi))] = mode
    
    # return mode with uncertainties
    return np.array([mode, hdi[1] - mode, mode - hdi[0]])


if __name__ == "__main__":
    
    mp.set_start_method(mp_start_method, force=True)
    if mp_start_method=='spawn':
        os.environ["PYTENSOR_FLAGS"] = f"compiledir=/tmp/pytensor_{os.getuid()}_{random.randint(0, 1000000)}"

    #=== user inputs ================================================
    #cfg.sn_thresh = 2.0 # S/N threshold for line detection (either fit or skip)
    #save_trace = False # save PyMC posterior trace or not
    #cfg.step_method = cfg.step_method # PyMC sampling method
    #
    ## fitting windows (M/H-Res Gratings)
    #cfg.line_range_kms = 3500 # fit window: lines +/- hwhm [km/s] (gratings)
    #cfg.fit_exclude_hwhm_kms = 100 # exclude window: unfitted lines +/-hwhm [km/s] (gratings)
    
    # fitting windows (L-Res PRISM)
    #cfg.line_range_kms = 20000 # (PRISM)
    #cfg.fit_exclude_hwhm_kms = 5000 # (PRISM)

    # random seed
    #RANDOM_SEED = 8927 # Initialize random number generator
    #rng = np.random.default_rng(RANDOM_SEED)
    #az.style.use("arviz-darkgrid")


    #=== set up paths ===============================================
    # directories: working dir, outputs
    if not os.path.exists(cfg.fpath_outputs):
        os.makedirs(cfg.fpath_outputs)
    
    #=== process user-selected lines ================================
    # sort the lines by wavelength
    line_keys = cfg.line_keys
    line_wavs = [cl.lines_dict[k][0][0] for k in line_keys]
    idxs_sorted = np.argsort(line_wavs)
    line_wavs = np.array(line_wavs)[idxs_sorted]
    line_keys = np.array(line_keys)[idxs_sorted]
    
    # do individ. lines have neighbours? make sets of nearby lines.
    # use lines defined in cl.lines_dict in addition to 
    # the user-provided lines.
    wav_edges = np.c_[[line_wavs - cfg.line_range_kms / 3e5 * line_wavs,
                       line_wavs + cfg.line_range_kms / 3e5 * line_wavs]].T
    line_sets = [get_line_wavel(
        w1, w2, 
        lines_dict=cl.lines_dict, 
        exclude=cl.lw_exclude_sets
    ) for w1, w2 in wav_edges]
    line_set_keys = []
    line_set_wavs = []
    for keys, wavs in line_sets:
        # sort the lines by wavelength in each set
        sorted_indices = np.argsort(np.array(wavs).flatten())
        line_set_keys.append([keys[i] for i in sorted_indices])
        line_set_wavs.append([wavs[i] for i in sorted_indices])
    
    # dictionary of all lines that have been identified at this point
    lines_all_dict = {}
    for set_keys, set_wavs in zip(line_set_keys, line_set_wavs):
        for key, wav in zip(set_keys, set_wavs):
            lines_all_dict[key] = [wav]
    
    # Remove duplicate lists from line_set_keys and preserve 
    # the ascending-wavelegth order 
    # (useful later for checking if nearby lines are resolved)
    seen = set()
    line_set_keys = [
        sublist for sublist in line_set_keys 
        if tuple(sublist) not in seen and not seen.add(tuple(sublist))
    ]
    
    #=== alternative code to make lines sets using only the user input 
    # line dictionary
    #line_dict = deepcopy(cl.lines_dict)
    #[line_dict.pop(k) for k in cl.lines_dict.keys() if not (k in line_keys)]

    # is a line close to its neighbour? can later combine close lines into a set
    #vel_fwhm_kms = (line_wavs[1:] - line_wavs[:-1]) / line_wavs[:-1]*3e5*2.355
    #is_set = vel_fwhm_kms < 2*cfg.line_range_kms

    # make sets
    #line_keys_sets = []
    #i = 0
    #line_keys_sets.append([line_keys[0]])
    #for _cond_cur in is_set:
    #    if _cond_cur:
    #        line_keys_sets[-1].extend([line_keys[i+1]])
    #    else:
    #        line_keys_sets.append([line_keys[i+1]])
    #    i += 1
    #
    #print(line_keys_sets)
    
    
    # list of lines excluded from fit
    lines_excl = []
    for k in cl.lw_exclude_lines:
        lines_excl.extend(cl.lw[k])

    #=== sample selection ================================================
    df_sample = pd.read_csv(cfg.fpath_spec, comment='#')
    print("data table size:", len(df_sample))

    # load a SNR catalog (for SN cuts of fitted lines)
    df_lines_sn = pd.read_csv(cfg.fpath_lines_snr)

    #=== Define the model, fit each spectrum and save the outputs 
    samplers = {'MH': pm.Metropolis,
                'DEMZ': pm.DEMetropolisZ,
                'HMC': pm.HamiltonianMC,
                'NUTS': pm.NUTS}

    i = 0
    skipped_file = {} # info about files skipped in the loop
    skipped_file['spec_v1'] = []
    skipped_file['nospec'] = []
    skipped_file['fitted'] = []
    skipped_file['nolines'] = []
    skipped_file['nospecrange'] = []

    fig_axis_labels = {
        'fwhm': r'FWHM (km s$^{-1}$)',
        'cont_a': r'$a$ (continuum)',
        'cont_b': r'$b$ (continuum)',
        'z': r'Redshift',   
    }

    # loop over individual spectra (order: iter, grating, PROG_ID) and fit
    n_obj = df_sample.shape[0]
    for grating, group_grat in df_sample.groupby('grating'):
        for obj_id, group_obj in group_grat.groupby('PROG_ID'):
            for idx_row, row in group_obj.iterrows():
                
                i += 1
                
                # fit either all, or a specific spectrum
                if obj_id_user == 'all':
                    pass
                elif obj_id != obj_id_user:
                    continue
                
                print(f'\n=== {grating} / ID {obj_id} === {i}/{n_obj} spec ===')
                
                fname_spec = row['file'].replace('.spec.fits', '')
                if 'v1' in fname_spec:
                    skipped_file['spec_v1'].append(fname_spec)
                    print("  - Skipping... It is a v1 spectrum.")
                    continue
                
                #=== load a spectrum
                try:
                    sample_info = rspec.get_sample_dict(
                        group_obj, verbose=False)
                    sampleFit = rspec.SpectrumSampleFit(
                        fit_window_bins=None, flux_norm=None,
                        download_data=False, verbose=False,
                        **sample_info)
                except:
                    try:
                        sample_info = rspec.get_sample_dict(
                            group_obj, verbose=False
                        )
                        sampleFit = rspec.SpectrumSampleFit(
                            fit_window_bins=None, flux_norm=None, 
                            download_data=True, verbose=False,
                            **sample_info
                        )
                    except:
                        skipped_file['nospec'].append(fname_spec)
                        print("  - Skipping... No spec could be loaded.")
                        continue
                
                valid = sampleFit.valid[0]
                X = sampleFit.x[0][valid]
                Y = sampleFit.y[0][valid]
                Yerr = sampleFit.ye[0][valid]
                z_guess = sampleFit.z_input[0]
                X_rf = X / (1 + z_guess)
                xmin_spec, xmax_spec = X_rf.min(), X_rf.max()
                
                #=== spectral fit windows
                # get emission lines in the spectral range
                line_keys_spec, line_wavs_cur = get_line_wavel(
                    xmin_spec, xmax_spec, 
                    lines_dict=lines_all_dict, 
                    exclude=cl.lw_exclude_sets
                )
                
                if len(line_keys_spec) == 0:
                    _line_keys = list(lines_all_dict.keys())
                    skipped_file['nolines'].append(fname_spec+f'\t({_line_keys})')
                    print(f"  - Skipping ... No lines in the spectral range!\n\t\t({_line_keys})")
                    continue
                
                # create fit windows around the fitted lines
                fit_range = [[xmin_spec, xmax_spec]]
                fit_edges = rspec.exclude_fit_windows(
                    fit_range, hwhm=cfg.line_range_kms, 
                    lines=np.concatenate(line_wavs_cur)
                )
                fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)
                
                # exclude lines that are not fitted
                #fit_edges = rspec.exclude_fit_windows(
                #    fit_edges, lines=lines_excl, hwhm=cfg.fit_exclude_hwhm_kms
                #)
                
                # update the selected spectral data
                sampleFit.update_fit_window(fit_edges)
                mask = sampleFit.in_window[0][valid]
                if not mask.any():
                    skipped_file['nospecrange'].append(fname_spec)
                    print("  - Skipping ... No spectral range available!")
                    continue
                
                X = X[mask]
                Y = Y[mask]
                Yerr = Yerr[mask]
                X_rf = X / (1 + z_guess)
                xmin_mask, xmax_mask = X_rf.min(), X_rf.max()
                
                # get emission lines in the spectral range
                line_keys_spec, line_wavs_cur = get_line_wavel(
                    xmin_mask, xmax_mask, 
                    lines_dict=lines_all_dict, 
                    exclude=cl.lw_exclude_sets
                )
                
                if len(line_keys_spec) == 0:
                    skipped_file['nolines'].append(fname_spec)
                    print("  - Skipping ... No lines in the spectral range!")
                    continue
                
                # keep the line sets that are observed 
                # in the current spectrum
                line_set_keys_spec = []
                for s in line_set_keys:
                    is_in_spec = [k in line_keys_spec for k in s]
                    spec_set = np.array(s)[is_in_spec]
                    if len(spec_set) > 0:
                        line_set_keys_spec.append(list(spec_set))
                
                # loop over individual lines or sets of lines
                n_sets = len(line_set_keys_spec)
                for j in range(n_sets):
                    
                    # iterate over the sub sets of lines
                    line_set_keys_j = line_set_keys_spec[j]
                    line_set_wavs_j = [lines_all_dict[k][0][0] \
                                        for k in line_set_keys_j]
                    
                    n_lines = len(line_set_keys_j)
                    
                    #=== check that the lines are detected in the spectrum
                    # if not, skip - no need to fit them
                    cols_sn_i = [f'sn_{l}' for l in line_set_keys_j]
                    print(f'  - spectrum: {sampleFit.fnames[0]}')

                    cols_df = df_lines_sn.columns
                    has_line = np.array([c in cols_df for c in cols_sn_i])
                    cols_sn_avail = list(np.array(cols_sn_i)[has_line])
                    
                    skip_set = False
                    mask_file = df_lines_sn.file.isin([sampleFit.fnames[0]])
                    if (len(cols_sn_avail) > 0) & mask_file.any():
                        mask_sn = [df_lines_sn.loc[mask_file,c].item() > cfg.sn_thresh\
                                    for c in cols_sn_avail]
                        skip_set = not np.any(mask_sn)
                        print(f'mask_sn: {mask_sn}')
                    elif (len(cols_sn_avail) > 0) & (not mask_file.any()):
                        # the file isn't in the df_lines_sn table, so don't skip it
                        skip_set = False
                    else:
                        skip_set = True
                    if skip_set:
                        print(f"  - fitting lines: {line_set_keys_j} ...")
                        print(f"  - Skipping... No lines with at least {cfg.sn_thresh}-sigma detection.")
                        continue
                    
                    # get the wavelength range containing the fitted lines
                    fit_edges = rspec.exclude_fit_windows(
                        fit_range, hwhm=cfg.line_range_kms, lines=line_set_wavs_j
                    )
                    fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)

                    # exclude lines that are not fitted
                    #fit_edges = rspec.exclude_fit_windows(
                    #    fit_edges, lines=lines_excl, hwhm=cfg.fit_exclude_hwhm_kms
                    #)
                    
                    # update the selected spectral data
                    sampleFit.update_fit_window(fit_edges)
                    mask = sampleFit.in_window[0]
                    if not (mask & valid).any():
                        skipped_file['nospecrange'].append(fname_spec)
                        print("  - Skipping ... No spectral range available!")
                        continue
                    
                    X_fit = sampleFit.x[0][mask & valid]
                    Y_fit = sampleFit.y[0][mask & valid]
                    Yerr_fit = sampleFit.ye[0][mask & valid]
                    nobs = len(X_fit)
                    
                    # starting guess
                    start_params = {
                        'cont_a': 1e-2, 
                        'cont_b': 1e-1, 
                        'z': z_guess,
                        'fwhm': 200,
                    }
                    amplitude_params = dict(zip([k+'_amplitude' \
                                                    for k in line_set_keys_j],
                                                [0.0 for k in line_set_keys_j]))
                    start_params |= amplitude_params
                    
                    # check if the spectrum has already been fitted and if so, 
                    # reuse it's best parameters as a current starting step
                    j_iter = 0
                    lines_str = '_'.join(line_set_keys_j)
                    model_label = f'{obj_id}-{fname_spec}-{grating}-iter{j_iter}-{lines_str}-step{cfg.step_method}'
                    model_label_any = f'{obj_id}-{fname_spec}-{grating}-iter*-{lines_str}-step{cfg.step_method}'
                    fpaths = os.path.join(cfg.fpath_outputs, f"{model_label_any}-line_props.fits")
                    
                    if len(glob.glob(fpaths)) > 0:
                        print(f"\n  - Iteration {j_iter}... Set {j+1}/{n_sets}...")
                        print(f"  - fitting lines: {line_set_keys_j} ...")
                        print(f"  - Skipping... Already fitted.")
                        continue # VR
                        
                        #model_label = f'{obj_id}-{fname_spec}-{grating}-iter*-{lines_str}-step{cfg.step_method}'
                        #fpaths = os.path.join(cfg.fpath_outputs, 
                        #                      f"{model_label}-line_props.fits")
                        
                        labels = glob.glob(fpaths)
                        labels_iters = [int(l.split('-iter')[-1].split('-')[0]) \
                                        for l in labels]
                        j_iter = np.max(labels_iters) + 1
                        
                        # current iteration label
                        model_label = f'{obj_id}-{fname_spec}-{grating}-iter{j_iter}-{lines_str}-step{cfg.step_method}'
                        fpaths = os.path.join(cfg.fpath_outputs, 
                                            f"{model_label}-line_props.fits")

                        # load results from prev iteration
                        model_label_prev = f'{obj_id}-{fname_spec}-{grating}-iter{j_iter-1}-{lines_str}-step{cfg.step_method}'
                        fpaths_prev = os.path.join(cfg.fpath_outputs, 
                                        f"{model_label_prev}-line_props.fits")
                        tab = Table.read(fpaths_prev)
                        tab_cols = tab.colnames
                        tab_vals = [tab[c].item() for c in tab.colnames]
                        tab_vals[5:] = [float(v) for v in tab_vals[5:]]
                        prev_params = dict(zip(tab_cols, tab_vals))
                        for key, val in prev_params.items():
                            if key in start_params.keys():
                                start_params[key] = val

                    print(f"\n  - Iteration {j_iter}... Set {j+1}/{n_sets}...")
                    print(f"  - fitting lines: {line_set_keys_j} ...")
                    
                    # pickle the fitted data
                    d = {'X': X_fit, 'Y': Y_fit, 'Yerr': Yerr_fit, 
                        'z_guess': z_guess, 
                        'flux_norm': sampleFit.flux_norm[0],
                    }
                    fname = f"{model_label}-data.pckl"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    with open(fpath, 'wb') as f:
                        pickle.dump(d, f)
                    
                    #=== construct a spectrum model
                    model = pm.Model()
                    with model:
                
                        # continuum component
                        prefix = 'cont_'
                        b_max = np.median(Y_fit)
                        a = pm.Uniform(f"{prefix}a", lower=-10, upper=10)
                        b = pm.Uniform(f"{prefix}b", 
                                    lower=-10-abs(b_max)*10, 
                                    upper=10+abs(b_max)*10)
                        #z = pm.Normal(f"z", initval=z_guess, mu=z_guess, 
                        #            sigma=1e-4*z_guess)
                        z = pm.Uniform("z", initval=z_guess, 
                                       lower=z_guess-0.003, upper=z_guess+0.003)
                        if j_iter == 0:
                            start_params['cont_b'] = b_max
                        
                        # x values for fitting the lines and continuum
                        x_rf_guess = X_fit / (1 + z_guess)
                        x_rf = X_fit / (1 + z)
                        x_cont = np.linspace(-1, 1, len(X_fit), 
                                                endpoint=True)
                        
                        # merge models of lines that are not resolved,
                        # the resolution is defined as sep<1.5*FWHM(instrum)
                        fwhm_intrum = []
                        for line_key, line_wav in zip(line_set_keys_j, 
                                                        line_set_wavs_j):
                            
                            # spec resolution at the wavel
                            sig_disp = sampleFit.get_dispersion(
                                [obj_id], line_wav, [z_guess]
                            )[0]
                            fwhm_disp_kms = sig_disp/line_wav*3e5*2.355 # km/s
                            fwhm_intrum.append(fwhm_disp_kms)
                        
                        line_set_wavs_j = np.array(line_set_wavs_j)
                        fwhm_intrum = np.array(fwhm_intrum)
                        sep_fwhm_kms = np.diff(line_set_wavs_j) /\
                                                line_set_wavs_j[:-1] * 3e5
                        unresolved = 1.5*fwhm_intrum[:-1] > sep_fwhm_kms
                        if unresolved.any():
                            # idx to remove
                            idx2remove = np.argwhere(unresolved).flatten() + 1
                            
                            # set one mean wavelength for the combined lines
                            for idx in idx2remove:
                                merged_wav = np.mean(line_set_wavs_j[idx-1:idx+1])
                                line_set_wavs_j[idx-1] = merged_wav
                            
                            line_set_wavs_j = np.delete(line_set_wavs_j, idx2remove)
                            
                            # replace line keys with a combined ine
                            lines_combined = [line_set_keys_j[i] \
                                for i in np.insert(idx2remove, 0,
                                                idx2remove[0]-1)]
                            merged_lines = '_'.join([line \
                                                    for line in lines_combined])
                            line_set_keys_j[idx2remove[0]-1] = merged_lines
                            line_set_keys_j = [key for key in line_set_keys_j \
                                                if not key in merged_lines[1:]]
                            start_params[merged_lines+'_amplitude'] =\
                                start_params[lines_combined[0]+'_amplitude']
                            [start_params.pop(line+'_amplitude') \
                                    for line in lines_combined]
                            print(f"  - unresolved lines: {lines_combined} -- fitting as {merged_lines}")
                            
                            #if j_iter > 0:
                            #    start_params[merged_lines+'_amplitude'] =\
                            #        prev_params[merged_lines+'_amplitude']
                        print(f"  - starting params: {start_params}")
                        
                        # model components
                        #Y_lines = 0
                        Y_lines_list = []
                        fwhm = pm.Uniform(f"fwhm", 
                                        lower=0.1, upper=2000.0) # [km/s]
                        
                        for line_key, line_wav in zip(line_set_keys_j, 
                                                    line_set_wavs_j):
                            
                            # spec resolution at the wavel
                            sig_disp = sampleFit.get_dispersion(
                                [obj_id], line_wav, [z_guess]
                            )[0]
                            
                            prefix = f'{line_key}_'
                            A = 10.**pm.Uniform(f"{prefix}amplitude", 
                                                lower=-3, upper=4)
                            
                            sigma = fwhm/2.355/3e5*line_wav # Gauss disp. [A]
                            Y_line = pm.Deterministic(
                                f'Y_{line_key}',
                                A * np.exp(-0.5 * (x_rf - line_wav)**2 /\
                                    (sigma**2 + sig_disp**2))
                            )
                            Y_lines_list.append(Y_line)
                            
                            # add a broad component?
                            if line_key in cfg.broad_lines:
                                # Ha broad component
                                prefix = f'{line_key}_2_'
                                A2 = 10**pm.Uniform(f"{prefix}amplitude",
                                                lower=-2, upper=2) # log10(A)
                                mu2 = pm.Uniform(f"{prefix}center",
                                            lower=line_wav-5, upper=line_wav+5)
                                
                                fwhm2 = pm.Uniform(f"{prefix}fwhm",
                                                lower=0.0, upper=2000)
                                sigma2 = fwhm2 / 2.355 / 3e5 * mu2
                                sigma2 = np.sqrt(sigma2**2 + sig_disp**2) # +instrum. disp.
                                
                                tau2 = pm.Uniform(f"{prefix}decay",
                                                lower=-2, upper=0) #[-2, 0]
                                v2 = 1.0 / (10.0**tau2)
                                
                                # Argument for the exponential and error functions in each term
                                x_zero_rf_var = x_rf - mu2
                                exparg1 = 0.5 * sigma2**2 * v2**(-2) - x_zero_rf_var/v2
                                exparg2 = 0.5 * sigma2**2 * v2**(-2) + x_zero_rf_var/v2
                                erfarg1 = (x_zero_rf_var/sigma2 - sigma2/v2)/(2**0.5)
                                erfarg2 = -(x_zero_rf_var/sigma2 + sigma2/v2)/(2**0.5)
                                
                                # Compute each part of the expression
                                term1 = pt.exp(exparg1) * 0.5 * (1 + pt.erf(erfarg1))
                                term2 = pt.exp(exparg2) * 0.5 * (1 + pt.erf(erfarg2))
                                
                                Y_line_b = pm.Deterministic(f'Y_line_b',
                                    A2 * (term1 + term2) / (1 + z))
                                
                                Y_lines_list.append(Y_line_b)
                            
                        
                        #Y_lines = pm.Deterministic("Y_lines", Y_lines)
                        Y_lines = pm.Deterministic("Y_lines", 
                                        pt.sum(pt.stack(Y_lines_list), axis=0))
                        Y_cont = pm.Deterministic("Y_cont", (a * x_cont + b))
                        Y_mu = Y_lines + Y_cont
                        
                        # Likelihood (sampling distribution) of observations
                        Y_pred = pm.Normal("Y_pred", mu=Y_mu, sigma=Yerr_fit, 
                                        observed=Y_fit)
                
                        #=== sample posterior space
                        var_names = [rv.name for rv in model.free_RVs]
                        npar = len(var_names)
                        nchains = int(npar * 4) if int(npar * 4) < 50 else 50
                        print(f"  - number of params: {npar}")
                        print(f"  - number of data: {nobs}")
                        print(f"  - number of steps: {nsteps}")
                        print(f"  - number of chains: {nchains}")
                        if nchains == 50:
                            print("    consider reducing number of parameters!")
                            
                        #=== find an MLE solution & use it to start the MCMC
                        try:
                            start_params_mle = deepcopy(start_params)
                            if j_iter == 0:
                                mle = pm.find_MAP(method='L-BFGS-B',#'powell',
                                                start=start_params)
                                par_mle = dict(
                                    zip(var_names, 
                                        [mle[key] for key in var_names])
                                )
                                isna = np.isnan(list(par_mle.values()))
                                if not isna.any():
                                    start_params_mle |= par_mle
                                start_params_mle['z'] = start_params['z'] # I don't trust LBFGS to change 'z'

                            # draw samples
                            trace = pm.sample(nsteps, tune=ntune, cores=ncores, 
                                            chains=nchains, 
                                            initvals=start_params_mle,
                                            step=samplers[cfg.step_method](),
                                            #nuts_sampler='blackjax'
                                            #nuts_sampler='pymc'
                                            )
                            used_mle_start = 1.0
                        except:
                            # draw samples
                            trace = pm.sample(nsteps, tune=ntune, cores=ncores, 
                                            chains=nchains, 
                                            initvals=start_params,
                                            step=samplers[cfg.step_method](),
                                            #nuts_sampler='blackjax'
                                            #nuts_sampler='pymc'
                                            )
                            used_mle_start = 0.0
                        posterior_pc = pm.sample_posterior_predictive(trace)
                    
                    nthin = (nsteps - ntune) // 2000
                    nthin = nthin if nthin > 0 else 1
                    trace = trace.sel(draw=slice(None, None, nthin))
                    
                    # save trace
                    if cfg.save_trace:
                        fname = f"{model_label}-trace.nc"
                        fpath = os.path.join(cfg.fpath_outputs, fname)
                        trace.to_netcdf(fpath, compress=True)
                    
                    #=== get median parameters
                    # best values: mode; uncertainty: 68% high-density interval
                    params_best_full = {
                        var: mode_in_hdi(trace.posterior[var].data.flatten(), 
                                        bw_fct=1.0) \
                                for var in var_names
                    }
                    params_best = {
                        key: val[0] for key, val in params_best_full.items()
                    }
                    
                    # calculate goodness of fit metrics
                    y_pred_chain=posterior_pc.posterior_predictive['Y_pred'].data.reshape(-1,nobs)
                    y_pred = np.array([mode_in_hdi(y_pred_chain[:, i]) \
                                        for i in range(nobs)]).T
                    
                    # residuals statistics
                    diff_chi = (Y_fit - y_pred[0]) / Yerr_fit
                    chisq = np.sum(diff_chi**2)
                    redchisq = np.sum(diff_chi**2) / (len(Y_fit)-len(var_names))
                    bic = chisq + npar * np.log(nobs)
                    aic = chisq + 2 * npar
                    
                    #============== tabulate & save outputs =================
                    model_dict = {}
                    model_dict['PROG_ID'] = obj_id
                    model_dict['model_label'] = model_label
                    model_dict['chisq'] = chisq
                    model_dict['redchisq'] = redchisq
                    model_dict['bic'] = bic
                    model_dict['aic'] = aic
                    model_dict['nobs'] = nobs
                    model_dict['npar'] = npar
                    model_dict['step_method'] = cfg.step_method
                    model_dict['nsteps'] = nsteps
                    model_dict['nthin'] = nthin
                    model_dict['nchains'] = nchains
                    model_dict['used_mle_start'] = used_mle_start
                    
                    columns = []
                    values = []

                    # model columns
                    cols_model = ['model_label', 'redchisq', 'nobs', 'npar',
                                'step_method', 'nsteps', 'ntune', 'nthin']
                    vals_model = [model_label, redchisq, nobs, npar, 
                                cfg.step_method, nsteps, ntune, nthin]
                    columns += cols_model
                    values += vals_model

                    # observations columns
                    cols_fit_edges = [f'fit_edge_{i}' \
                            for i in range(len(fit_edges.flatten()))]
                    vals_fit_edges = [edge for edge in fit_edges.flatten()]

                    flux_norm = sampleFit.flux_norm[0].item()

                    columns += cols_fit_edges + ['flux_norm']
                    values += vals_fit_edges + [flux_norm]

                    # best-fit params columns
                    params_names = list(params_best_full.keys())
                    params_sigup_names = [f"{p}_sigup" for p in params_names]
                    params_siglo_names = [f"{p}_siglo" for p in params_names]
                    cols_parnames = np.c_[params_names, 
                                        params_sigup_names, 
                                        params_siglo_names].flatten()
                    columns += list(cols_parnames)

                    params_values = [params_best_full[p][0] \
                                        for p in params_names]
                    params_sigup = [params_best_full[p][1] \
                                        for p in params_names]
                    params_siglo = [params_best_full[p][2] \
                                        for p in params_names]
                    vals_par = np.c_[params_values, 
                                    params_sigup, 
                                    params_siglo].flatten()
                    values += list(vals_par)
                    
                    # line SN and Flux
                    cols_line, vals_line = [], []
                    sig_integ = 7.0 # sigmas to integrate over for Flux, SN, EW
                    n_interp = 1000
                    X_rf = X_fit / (1 + params_best['z'])
                    print(line_set_keys_j, line_set_wavs_j)
                    for k, w in zip(line_set_keys_j, line_set_wavs_j):
                        
                        # spec resolution at the wavel
                        disp_sig_ang = sampleFit.get_dispersion(
                            [obj_id], w, [params_best['z']])
                        
                        #=== compute S/N(Line)
                        velocity_sig_ang = params_best['fwhm']/2.355/3e5*w
                        total_sig_ang = np.sqrt(disp_sig_ang**2 +\
                                                velocity_sig_ang**2)
                        is_line = (X_rf > (w - sig_integ*total_sig_ang)) &\
                                  (X_rf < (w + sig_integ*total_sig_ang))
                        n_obs_line = is_line.sum()
                        print(f'  - {k} n_data: {n_obs_line}')
                        if n_obs_line == 0:
                            print(f'  - {k}: no valid data points.')
                            continue
                        
                        # posterior flux of the continuum
                        y_cont_samples = trace.posterior['Y_cont'].data.reshape(-1,nobs)[:,is_line]
                        y_cont = np.array([mode_in_hdi(y_cont_samples[:,i])\
                                            for i in range(n_obs_line)]).T
                        ye_cont_flux = 0.5 * (y_cont[1] + y_cont[2])
                        
                        if len(line_set_keys_j) > 1:
                            # get posterior fluxes of all but the current 
                            # components
                            line_keys_sub = deepcopy(line_set_keys_j)
                            line_keys_sub.pop(line_keys_sub.index(k)) # other comps
                            y_sub_samples = np.array([
                                trace.posterior[f'Y_{k_comp}'].data.reshape(-1, nobs)[:,is_line]\
                                for k_comp in line_keys_sub]) # sub. comps
                            if y_sub_samples.ndim < 3:
                                y_sub_samples = np.atleast_3d(y_sub_samples).T
                            y_sub_samples = y_sub_samples.sum(axis=0) # sum comps.
                            
                            y_sub = []
                            for i in range(n_obs_line):
                                try:
                                    y_sub.append(
                                        mode_in_hdi(y_sub_samples[:, i]))
                                except:
                                    y_sub.append([0.0, 0.0, 0.0])
                            y_sub = np.array(y_sub).T # [[mode,up,lo], ndata]
                            ye_sub = 0.5*(y_sub[1] + y_sub[2])
                            
                            # calculating SNR(component): 
                            # subtract all components except the current one
                            # and propagate the uncertainty 
                            y_flux_line = (Y_fit[is_line] - y_sub[0])\
                                            - y_cont[0]
                            ye_flux_line = np.sqrt(
                                Yerr_fit[is_line]**2 + ye_cont_flux**2 + ye_sub**2
                            )
                        else:
                            y_flux_line = Y_fit[is_line] - y_cont[0]
                            ye_flux_line = np.sqrt(
                                Yerr_fit[is_line]**2 + ye_cont_flux**2
                            )
                            
                        sn_line = np.sum(y_flux_line) /\
                                  np.sqrt(np.sum(ye_flux_line**2))
                        
                        # compute line Flux
                        Y_line_samples = trace.posterior[f'Y_{k}'].data.reshape(-1,nobs)[:,is_line]
                        Y_flux_samples = simpson(
                            Y_line_samples, x=X_fit[is_line], axis=1)
                        Y_flux = mode_in_hdi(Y_flux_samples)
                        
                        # compute line EW
                        Y_eqw_samples = simpson(
                            Y_line_samples/y_cont_samples, 
                            x=X_fit[is_line] / (1 + params_best['z']), 
                            axis=1)
                        Y_eqw = mode_in_hdi(Y_eqw_samples)
                        
                        # append values
                        cols_line += [f'{k}_SN']
                        vals_line += [sn_line]
                        
                        cols_line += [f'{k}_flux', 
                                    f'{k}_flux_sigup', 
                                    f'{k}_flux_siglo']
                        vals_line += [Y_flux[0], Y_flux[1], Y_flux[2]]
                        
                        cols_line += [f'{k}_ew', 
                                    f'{k}_ew_sigup', 
                                    f'{k}_ew_siglo']
                        vals_line += [Y_eqw[0], Y_eqw[1], Y_eqw[2]]
                        
                        cols_line += [f'{k}_ndata']
                        vals_line += [n_obs_line]
                        
                    columns += cols_line
                    values += vals_line
                    
                    dtypes = []
                    for val in values:
                        if isinstance(val, str):
                            dtypes.append(f'<U{len(val)}')
                        elif isinstance(val, float) | isinstance(val, int):
                            dtypes.append('f4')
                        else:
                            dtypes.append('i4')
                    
                    # make a table and save it
                    tab_line_props = Table(data=np.array(values)[None, :], 
                                        dtype=dtypes, names=columns)
                    fname = f"{model_label}-line_props.fits"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    tab_line_props.write(fpath, overwrite=True)
                    
                    #============== make figures =================
                    nthin = nthin if nthin > 0 else 1
                    trace = trace.sel(draw=slice(None, None, nthin))
                    
                    #=== Figure 1: posterior traces and projections
                    axes = pm.plot_trace(trace, var_names=var_names, 
                                        compact=True, combined=True)
                    for param, ax in zip(params_best_full.keys(), 
                                        axes[:,0].flatten()):
                        value = params_best_full[param][0]
                        sigup = params_best_full[param][1]
                        siglo = params_best_full[param][2]
                        ax.axvline(value, color="C1", linestyle=":", lw=1)
                        ax.axvline(value+sigup, color="k", linestyle=":", lw=1)
                        ax.axvline(value-siglo, color="k", linestyle=":", lw=1)
                        #ax.set_xlim(axes_limits[param])

                    # save figure
                    fname = f"{model_label}-trace.png"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    plt.savefig(fpath, bbox_inches='tight', dpi=200)
                    plt.close()
                    
                    #=== Figure 2: corner plot
                    var_labels = [
                        fig_axis_labels[k] \
                        if k in fig_axis_labels.keys() else k \
                            for k in var_names
                    ]
                    
                    flatchain = np.array([trace.posterior[v].data.flatten()\
                                        for v in var_names]).T
                    thruths = np.array([params_best[v] for v in var_names])
                    thruths_sigup = np.array([params_best_full[v][1] \
                                            for v in var_names])
                    thruths_siglo = np.array([params_best_full[v][2] \
                                            for v in var_names])
                    thruths_up = thruths + thruths_sigup
                    thruths_lo = thruths - thruths_siglo
                    fig = corner.corner(flatchain, 
                                        labels=var_labels, 
                                        truths=thruths,
                                        #show_titles=True, 
                                        label_kwargs=dict(fontsize=14),
                                        use_math_text=True, 
                                        #title_quantiles=[thruths_lo, thruths, thruths_up],
                    )
                    corner.overplot_lines(fig, thruths_up, color="k", ls=':')
                    corner.overplot_lines(fig, thruths_lo, color="k", ls=':')
                    
                    fname = f"{model_label}-corner.png"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    plt.savefig(fpath, bbox_inches='tight', dpi=220)
                    plt.close()

                    #=== Figure 3: best-fit model
                    fig, (ax, axd) = plt.subplots(
                        2, 1, figsize=(4, 3), sharex=True,
                        gridspec_kw={'height_ratios': [1.5, 1],
                                    'hspace': 0.0},
                    )
                    xmin_fit, xmax_fit = X_rf.min(), X_rf.max()
                    ax.set_xlim(xmin_fit, xmax_fit)
                    ax.set_ylabel(r'$F_{\lambda}$')
                    axd.set_xlabel(r'Rest-frame Wavelength [${\rm \AA}$]')
                    axd.set_ylabel(r'$\chi$')

                    ax.step(X_rf, Y_fit, lw=0.5, color='k', where='mid')
                    ax.fill_between(X_rf, Y_fit-Yerr_fit, Y_fit+Yerr_fit, 
                                    lw=0, color='k', step='mid', alpha=0.3)

                    ax.step(X_rf, y_pred[0], color='C1', lw=1, zorder=10, 
                            where='mid', label='Median posterior')
                    ax.fill_between(X_rf, 
                                    y_pred[0]-y_pred[1], 
                                    y_pred[0]+y_pred[2], 
                                    step='mid', alpha=0.4, color='C1', lw=0, 
                                    zorder=9)
                    line_names = [v for v in trace.posterior.data_vars if 'Y_' in v]
                    [ax.plot(X_rf, 
                             trace.posterior[k].median(axis=(0,1)).data,
                             lw=1, label=k, ls='--') \
                        for k in line_names]
                    
                    axd.errorbar(X_rf, diff_chi, yerr=1.0, fmt='o', color='k', 
                                ms=1.5, elinewidth=1, zorder=10)
                    axd.text(0.02, 0.85, rf"$\chi^2_{{\nu}}: {redchisq:.2f}$",
                            transform=axd.transAxes, fontsize=7, color='k')
                    axd.set_ylim(-7, 10)
                    axd.set_yticks(np.arange(-5, 6, 5), minor=False)
                    axd.set_yticks(np.arange(-5, 6, 1), minor=True)
                    axd.grid(which='both', axis='y', lw=1, ls=':', color='k', 
                                alpha=0.3)
                    axd.axhline(0.0, ls='-', color='k', lw=0.5)

                    # annotate spectral lines
                    [ax.axvline(w, ls='--', lw=0.5, c='C4') for w in line_set_wavs_j]
                    title = ', '.join(line_set_keys_j)
                    ax.set_title(title, fontsize=8)
                    ax.legend(fontsize=8)
                    
                    # save figure
                    fname = f"{model_label}-best_fit.png"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    plt.savefig(fpath, bbox_inches='tight', dpi=200)
                    plt.close()

    # Define the path for the skipped file log
    skipped_file_log_path = os.path.join(cfg.fpath_outputs, 'skipped_files.txt')

    # Open the file in append mode
    with open(skipped_file_log_path, 'a') as log_file:
        # Write the current date and time
        log_file.write(f"\n=== Log Entry: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        
        # Write the contents of the skipped_file dictionary
        for key, values in skipped_file.items():
            log_file.write(f"{key}:\n")
            for value in values:
                log_file.write(f"  - {value}\n")
    
    # when done fitting, clean up the compiled files produced by the C-linker
    if "PYTENSOR_FLAGS" in os.environ:
        shutil.rmtree(os.environ["PYTENSOR_FLAGS"].split("=")[1], 
                      ignore_errors=True)
