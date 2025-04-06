import glob, grizli, importlib, os, pickle

import astropy.constants as ac
import astropy.units as au
import corner
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt

from astropy.cosmology import Planck18
from astropy.table import Table, vstack, hstack
from copy import deepcopy
from IPython.display import display
from scipy.integrate import simpson

import helper_module as hmod
import routines_spec as rspec
#import config_lres as cfg
import config_mhres as cfg

mpl.style.use('scientific')

cosmo = Planck18

#RANDOM_SEED = 8927 # Initialize random number generator
#rng = np.random.default_rng(RANDOM_SEED)
#az.style.use("arviz-darkgrid")

#=== set up the run ==========================================
fpath_runs = f'sample-{cfg.label_catalog}'
fpath_outputs = os.path.join(fpath_runs, cfg.fdir_outputs)

# user inputs
nsteps = 20000
ntune = 10000
ncores = 9
step_method = cfg.step_method
if not os.path.exists(fpath_outputs): 
    os.makedirs(fpath_outputs)

hwhm_kms = 3000 # gratings hwhm around the lines [km/s]
#hwhm_kms = 20000 # PRISM hwhm around the lines [km/s]

hwhm_exclude = 100 # gratings; exclude hwhm around unfitted lines [km/s]
#hwhm_exclude = 5000

sn_thresh = 2.0 # S/N threshold for line detection (either fit or not a line)

#n_lines_max = 6

# lines to fit
lines_neutral = [
    'OI_2972', 'OI_6364',
    'NI_3466', 'NI_5200',
]

lines_MgS = [
    'MgV_2783', 'MgV_1325', 
    'SIII_3722',
    #'SII_4069', 
    #'SIII_6312', 'SIII_9069', 'SIII_9532'
]
lines_Ar = [
    'ArIII_7136', 'ArIII_3005', 
    'ArIV_4740', 'ArIV_4711', 'ArIV_2689', 'ArIV_2854',
    'ArV_6134', 'ArV_2691'
]
lines_Ne = [
    'NeV_1575', 'NeIV_1602', 'NeIII_1815', 
    'NeIV_2422', 'NeV_3426', 'NeIII_3967'
]
lines_cnohe = [
    'CIV_1549', 'CIV_1551', 'CIII_1906', 'CIII_1908', # 'CII_4267' (not in pn),
    'NIV_1483', 'NIV_1487', 
    'NIII_1750', 'NII_3063', 'NII_5756', 
    'NII_6549', 'NII_6584',
    'OIII_1660', 'OIII_1665', 
    'OIII_2321', 'OII_2471', 
    'OII_3727', 'OII_3729', 
    'OIII_4363', 'OIII_4959', 'OIII_5007', 
    'OII_7323', 'OII_7332',
    'HeII_1640', 'HeII_4687',
]
cols_tem_diag = [
    'OIII_4363', 'OIII_4959', 'OIII_5007', 
    'NII_5756', 'NII_6549', 'NII_6584',
]
cols_den_diag = [
    'OII_3727', 'OII_3729',
    'SII_4070', 'SII_4078',
    'SII_6717', 'SII_6731'
]
cols_hydrogen = [
    'Hd_4103', 'Hg_4342', 'Hb_4861', 'Ha_6565'
]

#line_keys = cols_tem_diag + cols_den_diag + cols_hydrogen
line_keys = lines_MgS + lines_Ar + lines_Ne +\
            lines_cnohe + cols_tem_diag + cols_den_diag + cols_hydrogen
line_keys = np.unique(line_keys)

# line dictionary
lw = cfg.lw
line_wavs = [cfg.lines_dict[k][0][0] for k in line_keys]
idxs_sorted = np.argsort(line_wavs)
line_wavs = np.array(line_wavs)[idxs_sorted]
line_keys = np.array(line_keys)[idxs_sorted]

line_dict = dict(zip(line_keys, line_wavs))
line_dict = deepcopy(cfg.lines_dict)
[line_dict.pop(k) for k in cfg.lines_dict.keys() if not (k in line_keys)]

# is a line close to its neighbour?
set_separation_kms = 5000 # FWHM ([O III] 4959,5007 are fitted tigether)
#set_separation_kms = 20000 # FWHM
vel_fwhm_kms = (line_wavs[1:] - line_wavs[:-1]) / line_wavs[:-1] * 3e5 * 2.355
is_set = vel_fwhm_kms < set_separation_kms

# make sets of lines (at least one per set depending on proximity)
line_keys_sets = []
i = 0
line_keys_sets.append([line_keys[0]])
for _cond_cur in is_set:
    if _cond_cur:
        line_keys_sets[-1].extend([line_keys[i+1]])
    else:
        line_keys_sets.append([line_keys[i+1]])
    i += 1

# exclude these lines from fit range
lines_excl = []
for k in cfg.lw_exclude_lines:
    lines_excl.extend(lw[k])


#lw, lr = grizli.utils.get_line_wavelengths()

def get_line_wavel(w1, w2, lines_dict=lw, exclude=[]):
    names, waves = [], []
    for k, l in lines_dict.items():
        l = np.atleast_1d(l)
        cond = (l > w1) & (l < w2)
        if cond.any() & (k not in exclude):
            names.append(k)
            waves.append([*l[cond]])
    return names, waves

#=== sample selection ================================================
#fpath = os.path.join(fpath_runs, 
#                     'sample-NIII_1750_NIV_1483_NIV_1487_emitters_allspec.csv')
#fpath = os.path.join(fpath_runs, 'sample-NIV_emitters_allspec.csv')
fpath = os.path.join(fpath_runs, 'sample-non_N_emitters_allspec.csv')
#fpath = os.path.join(fpath_runs, 'sample-uncover_24175.csv')

df_sample = pd.read_csv(fpath, comment='#')
print("df_sample size:", len(df_sample))

#=== load a table with S/N values for all lines
fdir = f"sample-{cfg.label_catalog}"
fpath = os.path.join(fdir, 'sample-nirspec_lines-sn.csv')
df_lines_sn = pd.read_csv(fpath)

""" Define the model, load the spec sample, fit it and save the outputs """
#=== load data sample and sample info
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
models = {} # store models
n_obj = df_sample.shape[0]
for grating, group_grat in df_sample.groupby('grating'):
    for groupid, group_id in group_grat.groupby('GroupID'):
        for idx_row, row in group_id.iterrows():
            i += 1
            #if groupid not in [5275]:
            #    continue
            
            print(f'\n=== {grating} / ID {groupid} === {i}/{n_obj} spec ===')
            
            fname_spec = row['file'].replace('.spec.fits', '')
            if 'v1' in fname_spec:
                skipped_file['spec_v1'].append(fname_spec)
                print("  - Skipping... It is a v1 spectrum.")
                continue
            
            #=== load a spectrum
            try:
                sample_info = rspec.get_sample_dict(group_id, line_label=None,
                                                    verbose=False)
                sampleFit = rspec.SpectrumSampleFit(fit_window_bins=None, 
                                                    flux_norm=None,
                                                    download_data=False, 
                                                    verbose=False,
                                                    **sample_info)
            except:
                try:
                    sample_info = rspec.get_sample_dict(group_id, 
                                                        line_label=None,
                                                        verbose=False)
                    sampleFit = rspec.SpectrumSampleFit(fit_window_bins=None, 
                                                        flux_norm=None,
                                                        download_data=True, verbose=False,
                                                        **sample_info)
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
            #print(f'  z={z_guess:.4f}')
            
            #=== spectral fit windows
            # get emission lines in the spectral range
            line_keys_cur, line_wavs_cur = get_line_wavel(xmin_spec, xmax_spec, 
                                                    lines_dict=line_dict, 
                                                    exclude=cfg.lw_exclude_sets)
            if len(line_keys_cur) == 0:
                skipped_file['nolines'].append(fname_spec)
                print("  - Skipping ... No lines in the spectral range!")
                continue
            
            # create fit windows around the fitted lines
            fit_range = [[xmin_spec, xmax_spec]]
            fit_edges = rspec.exclude_fit_windows(fit_range, 
                                                  hwhm=hwhm_kms,
                                        lines=np.concatenate(line_wavs_cur))
            fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)
            #fit_edges = [[6200, xmax_spec]] # VR temp. patch for lines on edge
            
            # exclude lines that are not fitted
            fit_edges = rspec.exclude_fit_windows(fit_edges, lines=lines_excl, 
                                                  hwhm=hwhm_exclude)
            
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
            line_keys_cur, line_wavs_cur = get_line_wavel(xmin_mask, xmax_mask, 
                                                    lines_dict=line_dict, 
                                                    exclude=cfg.lw_exclude_sets)
            if len(line_keys_cur) == 0:
                skipped_file['nolines'].append(fname_spec)
                print("  - Skipping ... No lines in the spectral range!")
                continue
            
            # keep the line sets that are observed in the current spectrum
            line_keys_sets_cur = []
            for s in line_keys_sets:
                is_in_cur = [k in line_keys_cur for k in s]
                set_cur = np.array(s)[is_in_cur]
                if len(set_cur) > 0:
                    line_keys_sets_cur.append(list(set_cur))
            
            # loop over individual lines or sets of lines
            n_sets = len(line_keys_sets_cur)
            for j in range(n_sets):
                
                # iterate over the sub sets of lines (max n_lines_max at a time)
                line_keys_set = line_keys_sets_cur[j]
                line_wavs_set = [cfg.lines_dict[k][0][0] \
                                    for k in line_keys_set]
                
                n_lines = len(line_keys_set)
                
                #=== check that the lines are detected in the spectrum
                # if not, skip - no need to fit them
                cols_sn_cur = [f'sn_{l}' for l in line_keys_set]
                print(f'  - spectrum: {sampleFit.fnames[0]}')

                cols_df = df_lines_sn.columns
                has_line = np.array([c in cols_df for c in cols_sn_cur])
                cols_sn_avail = list(np.array(cols_sn_cur)[has_line])
                
                skip_set = False
                mask_file = df_lines_sn.file.isin([sampleFit.fnames[0]])
                if (len(cols_sn_avail) > 0) & mask_file.any():
                    mask_sn = [df_lines_sn.loc[mask_file,c].item() > sn_thresh\
                                for c in cols_sn_avail]
                    skip_set = not np.any(mask_sn)
                    print(f'mask_sn: {mask_sn}')
                elif (len(cols_sn_avail) > 0) & (~mask_file.any()):
                    # the file isn't in the df_lines_sn table, so don't skip it
                    skip_set = False
                else:
                    skip_set = True
                if skip_set:
                    print(f"  - fitting lines: {line_keys_set} ...")
                    print("  - Skipping... No lines with at least {sn_thresh}-sigma detection.")
                    continue
                
                # get the wavelength range containing the fitted lines
                fit_edges = rspec.exclude_fit_windows(fit_range, 
                                                      hwhm=hwhm_kms,
                                                      lines=line_wavs_set)
                fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)
                
                # exclude lines that are not fitted
                fit_edges = rspec.exclude_fit_windows(fit_edges, 
                                                      lines=lines_excl, 
                                                      hwhm=hwhm_exclude)
                
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
                start_params = {'cont_a': 1e-2, 
                                'cont_b': 1e-1, 
                                'z': z_guess,
                                'fwhm': 100}
                amplitude_params = dict(zip([k+'_amplitude' \
                                                for k in line_keys_set],
                                            [1.0 for k in line_keys_set]))
                start_params |= amplitude_params
                
                # check if the spectrum has already been fitted and if so, 
                # reuse it's best parameters as a current starting step
                j_iter = 0
                lines_str = '_'.join(line_keys_set)
                model_label = f'id{groupid}-iter{j_iter}-{grating}-{lines_str}-cont_linear-step{step_method}-{fname_spec}'
                model_label_any = f'id{groupid}-iter*-{grating}-{lines_str}-cont_linear-step{step_method}-{fname_spec}'
                fpaths = os.path.join(fpath_outputs, f"{model_label_any}-line_props.fits")
                
                if len(glob.glob(fpaths)) > 0:
                    print(f"\n  - Iteration {j_iter}... Set {j+1}/{n_sets}...")
                    print(f"  - fitting lines: {line_keys_set} ...")
                    print(f"  - Skipping... Already fitted.")
                    continue # VR
                    
                    #model_label = f'id{groupid}-iter*-{grating}-{lines_str}-cont_linear-step{step_method}-{fname_spec}'
                    #fpaths = os.path.join(fpath_outputs, 
                    #                      f"{model_label}-line_props.fits")
                    
                    labels = glob.glob(fpaths)
                    labels_iters = [int(l.split('-iter')[-1].split('-')[0]) \
                                    for l in labels]
                    j_iter = np.max(labels_iters) + 1
                    
                    # current iteration label
                    model_label = f'id{groupid}-iter{j_iter}-{grating}-{lines_str}-cont_linear-step{step_method}-{fname_spec}'
                    fpaths = os.path.join(fpath_outputs, f"{model_label}-line_props.fits")

                    # load results from prev iteration
                    model_label_prev = f'id{groupid}-iter{j_iter-1}-{grating}-{lines_str}-cont_linear-step{step_method}-{fname_spec}'
                    fpaths_prev = os.path.join(fpath_outputs, 
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
                print(f"  - fitting lines: {line_keys_set} ...")
                
                # pickle the fitted data
                d = {'X': X_fit, 'Y': Y_fit, 'Yerr': Yerr_fit, 'z_guess': z_guess,
                     'flux_norm': sampleFit.flux_norm[0],}
                fname = f"{model_label}-data.pckl"
                fpath = os.path.join(fpath_outputs, fname)
                with open(fpath, 'wb') as f:
                    pickle.dump(d, f)
                
                #=== construct a spectrum model
                model = pm.Model()
                with model:
            
                    # continuum component
                    prefix = 'cont_'
                    b_max = np.median(Y_fit)
                    a = pm.Uniform(f"{prefix}a", lower=-10, upper=10)
                    b = pm.Uniform(f"{prefix}b", lower=-10-abs(b_max)*10, 
                                   upper=10+abs(b_max)*10)
                    #a = pm.Normal(f"{prefix}a", mu=1e-2, sigma=5)
                    #b = pm.Normal(f"{prefix}b", mu=b_max, sigma=b_max)
                    z = pm.Normal(f"z", initval=z_guess, mu=z_guess, 
                                  sigma=0.0001*z_guess)
                    if j_iter == 0:
                        start_params['cont_b'] = b_max
                    
                    # Precompute constants
                    x_rf_guess = X_fit / (1 + z_guess)
                    x_rf = X_fit / (1 + z)
                    x_cont = np.linspace(-1, 1, len(X_fit), endpoint=True)
                    #x_zero_rf = (x_rf - 5000.0) # use some reference wavel
                    #x_zero_rf_um = x_zero_rf * 1e-4
                    
                    # model components
                    Y_lines = 0
                    fwhm = pm.Uniform(f"fwhm", initval=100.0,
                                      lower=0.1, upper=2000.0) # [km/s]
                    #fwhm = pm.Normal(f"fwhm", initval=100.0,
                    #                  mu=200, sigma=500) # [km/s]
                    for line_key, line_wav in zip(line_keys_set, 
                                                  line_wavs_set):
                        
                        # spec resolution at the wavel
                        sig_disp = sampleFit.get_dispersion([groupid], 
                                                            line_wav, 
                                                            [z_guess])
                        fwhm_disp_kms = sig_disp / line_wav * 3e5 * 2.355 # km/s
                        hwhm = 0.5 * fwhm_disp_kms / 3e5 * line_wav
                        is_line = (x_rf_guess > (line_wav - hwhm)) &\
                                  (x_rf_guess < (line_wav + hwhm))
                        
                        Y_peak = np.log10(np.max(Y_fit[is_line])) \
                                    if is_line.any() else 0.0
                        Y_peak = -2 if Y_peak <= 0.0 else Y_peak
                        prefix = f'{line_key}_'
                        A = 10.**pm.Uniform(f"{prefix}amplitude", 
                                            lower=-4, upper=3)
                        #A = 10.**pm.Normal(f"{prefix}amplitude", mu=0.5, 
                        # sigma=1)
                        
                        sigma = fwhm / 2.355 / 3e5 * line_wav # Gauss dispersion [A]
                        Y_line = pm.Deterministic(f'Y_{line_key}',
                                    A * np.exp(-0.5 * (x_rf - line_wav)**2 /\
                                    (sigma**2 + sig_disp**2)) / (1 + z) )
                        Y_lines += Y_line
                    Y_lines = pm.Deterministic("Y_lines", Y_lines)
                    Y_cont = pm.Deterministic("Y_cont", (a * x_cont + b) / (1 + z))
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
                        
                    #=== find an MLE solution first & use it to start the MCMC
                    used_mle_start = 0.0
                    #"""
                    try:
                        start_params_mle = deepcopy(start_params)
                        if j_iter == 0:
                            mle = pm.find_MAP(method='L-BFGS-B',#'powell',
                                              start=start_params)
                            par_mle = dict(zip(var_names, [mle[key] for key in var_names]))
                            isna = np.isnan(list(par_mle.values()))
                            #is_on_edge = (par_mle['Ha_6565n_fwhm'] == 10) |\
                            #             (par_mle['Ha_6565n_amplitude'] == 3)
                            #print(par_mle)
                            if not isna.any():
                                start_params_mle |= par_mle
                            start_params_mle['z'] = start_params['z'] # I don't trust LBFGS to change 'z'

                        # draw samples
                        trace = pm.sample(nsteps, tune=ntune, cores=ncores, 
                                        chains=nchains, 
                                        initvals=start_params_mle,
                                        #start_params=start_params,
                                        step=samplers[step_method](),
                                        #nuts_sampler='blackjax'
                                        #nuts_sampler='pymc'
                                        )
                        used_mle_start = 1.0
                    except:
                        # draw samples
                        trace = pm.sample(nsteps, tune=ntune, cores=ncores, 
                                        chains=nchains, 
                                        initvals=start_params,
                                        #start_params=start_params,
                                        step=samplers[step_method](),
                                        #nuts_sampler='blackjax'
                                        #nuts_sampler='pymc'
                                        )
                    posterior_pc = pm.sample_posterior_predictive(trace)

            
                nthin = (nsteps - ntune) // 2000
                nthin = nthin if nthin > 0 else 1
                trace = trace.sel(draw=slice(None, None, nthin))
                
                # save trace
                #fname = f"{model_label}-trace.nc"
                #fpath = os.path.join(fpath_outputs, fname)
                #trace.to_netcdf(fpath, compress=True)
                
                #=== get median parameters
                params_best_full = {key: 
                 np.array([trace.posterior[key].median().data,
                   *np.diff(trace.posterior[key].quantile([0.5, 0.84]).data),
                   *np.diff(trace.posterior[key].quantile([0.16, 0.5]).data)]) \
                                        for key in var_names}
                params_best = dict(zip(var_names, 
                                    [trace.posterior[key].median().data \
                                        for key in var_names]))
                
                # calculate goodness of fit metrics
                y_pred_chain=posterior_pc.posterior_predictive['Y_pred'].data.reshape(-1,nobs)
                y_pred = np.percentile(y_pred_chain, [16, 50, 84], axis=0)

                # residuals statistics
                diff_chi = (Y_fit - y_pred[1]) / Yerr_fit
                redchisq = np.sum(diff_chi**2) / (len(Y_fit) - len(var_names))
                
                #============== tabulate & save outputs =================
                model_dict = {}
                model_dict['model_label'] = model_label
                model_dict['redchisq'] = redchisq
                model_dict['nobs'] = nobs
                model_dict['npar'] = npar
                model_dict['step_method'] = step_method
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
                              step_method, nsteps, ntune, nthin]
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

                params_values = [params_best_full[p][0] for p in params_names]
                params_sigup = [params_best_full[p][1] for p in params_names]
                params_siglo = [params_best_full[p][2] for p in params_names]
                vals_par = np.c_[params_values, params_sigup, params_siglo].flatten()
                values += list(vals_par)
                
                # line SN and Flux
                cols_line, vals_line = [], []
                sig_integ = 7.0 # sigmas to integrate over for Flux, SN, EW
                n_interp = 1000
                X_rf = X_fit / (1 + params_best['z'])
                for k, w in zip(line_keys_set, line_wavs_set):
                    
                    # spec resolution at the wavel
                    disp_sig_ang = sampleFit.get_dispersion([groupid], w, 
                                                    [params_best['z']])
                    
                    # get posterior fluxes of model components
                    line_keys_sub = deepcopy(line_keys_set)
                    line_keys_sub.pop(line_keys_sub.index(k)) # all but cur comp
                    y_sub_comps = [trace.posterior[f'Y_{k_comp}'].data.reshape(-1,nobs)\
                                for k_comp in line_keys_sub] # comps to subtract
                    y_sub_comps = np.array([np.percentile(y_comp, 
                                                        [16, 50, 84], axis=0) \
                        for y_comp in y_sub_comps]) # [n_comps, ptiles, ndata]
                    
                    if len(y_sub_comps.shape) == 3:
                        ye_sub_comps = 0.5*(y_sub_comps[:,2] - 
                                            y_sub_comps[:,0]) # symm. 1-sigma
                    elif len(y_sub_comps.shape) == 2:
                        ye_sub_comps = 0.5*(y_sub_comps[2] - 
                                            y_sub_comps[0]) # symm. 1-sigma
                    
                    y_cont_chain = trace.posterior['Y_cont'].data.reshape(-1,
                                                                          nobs)
                    y_cont = np.percentile(y_cont_chain, [16, 50, 84], axis=0)
                    
                    #=== compute S/N(Line)
                    velocity_sig_ang = params_best['fwhm'] / 2.355 / 3e5 * w
                    total_sig_ang = np.sqrt(disp_sig_ang**2 +\
                                            velocity_sig_ang**2)
                    is_line = (X_rf > (w - sig_integ * total_sig_ang)) &\
                              (X_rf < (w + sig_integ * total_sig_ang))
                    n_data_line = is_line.sum()
                    print(f'  - {k} n_data: {n_data_line}')
                    if n_data_line == 0:
                        continue

                    # subtract from data all components except the current one
                    # and propagate the uncertainty 
                    # to compute SN(comp)
                    ye_cont_flux = 0.5*(y_cont[2][is_line] - y_cont[0][is_line])
                    y_flux_line = None
                    ye_flux_line = None
                    if y_sub_comps.ndim == 3:
                        y_flux_line = ((Y_fit - y_sub_comps[:,1].sum(axis=0)) - 
                                       y_cont[1])[is_line]
                        ye_flux_line = np.sqrt(Yerr_fit[is_line]**2 +
                                         ye_cont_flux**2 + 
                                         (ye_sub_comps**2).sum(axis=0)[is_line])
                    elif y_sub_comps.ndim == 2:
                        y_flux_line = ((Y_fit - y_sub_comps[1]) -
                                       y_cont[1])[is_line]
                        ye_flux_line = np.sqrt(Yerr_fit[is_line]**2 + 
                                               ye_cont_flux**2 + 
                                               (ye_sub_comps**2)[is_line])
                    elif y_sub_comps.ndim == 1:
                        y_flux_line = (Y_fit - y_cont[1])[is_line]
                        ye_flux_line = np.sqrt(Yerr_fit[is_line]**2 + 
                                               ye_cont_flux**2)

                    sn_line = np.sum(y_flux_line) /\
                              np.sqrt(np.sum(ye_flux_line**2))
                    
                    # compute line Flux
                    Y_line_chain = trace.posterior[f'Y_{k}'].data.reshape(-1,
                                                                        nobs)
                    Y_line = np.percentile(Y_line_chain[:,is_line], 
                                           [16, 50, 84], axis=0)
                    Y_line_flux50 = simpson(Y_line[1], x=X_fit[is_line])
                    Y_line_flux16 = simpson(Y_line[0], x=X_fit[is_line])
                    Y_line_flux84 = simpson(Y_line[2], x=X_fit[is_line])
                    Y_line_sigup = Y_line_flux84 - Y_line_flux50
                    Y_line_siglo = Y_line_flux50 - Y_line_flux16
                    
                    # compute line EW
                    Y_line_eqw_chain = simpson(Y_line_chain[:,is_line]/\
                                               y_cont_chain[:,is_line], 
                                               x=X_fit[is_line]) / (1 + params_best['z'])
                    Y_line_eqw50 = np.percentile(Y_line_eqw_chain, 50)
                    Y_line_eqw16 = np.percentile(Y_line_eqw_chain, 16)
                    Y_line_eqw84 = np.percentile(Y_line_eqw_chain, 84)
                    Y_line_eqw_sigup = Y_line_eqw84 - Y_line_eqw50
                    Y_line_eqw_siglo = Y_line_eqw50 - Y_line_eqw16
                    del Y_line_chain, Y_line_eqw_chain, y_cont_chain, y_sub_comps
                    
                    # append values
                    cols_line += [f'{k}_SN']
                    vals_line += [sn_line]
                    
                    cols_line += [f'{k}_flux', f'{k}_flux_sigup', f'{k}_flux_siglo']
                    vals_line += [Y_line_flux50, Y_line_sigup, Y_line_siglo]
                    
                    cols_line += [f'{k}_ew', f'{k}_ew_sigup', f'{k}_ew_siglo']
                    vals_line += [Y_line_eqw50, Y_line_eqw_sigup, 
                                  Y_line_eqw_siglo]
                    
                    cols_line += [f'{k}_ndata']
                    vals_line += [n_data_line]
                    
                columns += cols_line
                values += vals_line
                
                # make a table and save it
                tab_line_props = Table(data=np.array(values)[None, :], 
                                       names=columns)
                fname = f"{model_label}-line_props.fits"
                fpath = os.path.join(fpath_outputs, fname)
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
                fpath = os.path.join(fpath_outputs, fname)
                plt.savefig(fpath, bbox_inches='tight', dpi=200)
                plt.close()
                
                #=== Figure 2: corner plot
                var_posterior = dict(zip(var_names, 
                                        [trace.posterior[key] for key in var_names]))
                _ = corner.corner(var_posterior)

                fname = f"{model_label}-corner.png"
                fpath = os.path.join(fpath_outputs, fname)
                plt.savefig(fpath, bbox_inches='tight', dpi=220)
                plt.close()

                #=== Figure 3: best-fit model
                fig, (ax, axd) = plt.subplots(2, 1, figsize=(4, 3), sharex=True,
                                        gridspec_kw={'height_ratios': [1.5, 1],
                                                     'hspace': 0.0})
                xmin_fit, xmax_fit = X_rf.min(), X_rf.max()
                ax.set_xlim(xmin_fit, xmax_fit)
                ax.set_ylabel(r'$F_{\lambda}$')
                axd.set_xlabel(r'Rest-frame Wavelength [${\rm \AA}$]')
                axd.set_ylabel(r'$\chi$')

                ax.step(X_rf, Y_fit, lw=0.5, color='k', where='mid')
                ax.fill_between(X_rf, Y_fit-Yerr_fit, Y_fit+Yerr_fit, 
                                lw=0, color='k', step='mid', alpha=0.3)

                ax.step(X_rf, y_pred[1], color='C1', lw=1, zorder=10, 
                        where='mid', label='Median posterior')
                ax.fill_between(X_rf, y_pred[0], y_pred[2], step='mid',
                                alpha=0.4, color='C1', lw=0, zorder=9)
                [ax.plot(X_rf, 
                         trace.posterior[f'Y_{k}'].median(axis=(0,1)).data,
                         lw=1, label=k, ls='--') \
                    for k in line_keys_set]
                
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
                [ax.axvline(w, ls='--', lw=0.5, c='C4') for w in line_wavs_set]
                title = ', '.join(line_keys_set)
                ax.set_title(title, fontsize=8)
                ax.legend(fontsize=8)
                
                # save figure
                fname = f"{model_label}-best_fit.png"
                fpath = os.path.join(fpath_outputs, fname)
                plt.savefig(fpath, bbox_inches='tight', dpi=200)
                plt.close()
                
# save 'skipped' dictionary to a pickle file
#fpath = os.path.join(fpath_outputs, 'skipped_files.pckl')
#with open(fpath, 'wb') as f:
#    pickle.dump(skipped_file, f)
