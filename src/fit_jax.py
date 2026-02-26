# fit_jax.py
# This script performs line fitting using JAX and SciPy optimization.
# Example usage:
# python fit_jax.py all
# python fit_jax.py <PROG_ID>
# python fit_jax.py prog_ids.txt

import glob, os, pickle, random, shutil, sys
import corner, logging, warnings
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import minimize, differential_evolution

from astropy.table import Table
from copy import deepcopy
from scipy.integrate import simpson

import config as cfg
import config_lines as cl
from spectrum import *
from datetime import datetime

# Suppress scipy optimizer warning about flat gradients (normal near convergence)
warnings.filterwarnings("ignore", message="delta_grad == 0.0")

# Set JAX to use CPU
jax.config.update("jax_platform_name", "cpu")

# Verify device
#print(f"JAX default backend: {jax.default_backend()}")
#print(f"JAX devices: {jax.devices()}")

# Command line arguments
# Can be: 'all', a single PROG_ID, or path to a file with PROG_IDs (one per line)
obj_id_arg = sys.argv[1]
obj_id_list = None

if os.path.isfile(obj_id_arg):
    # Read list of PROG_IDs from file
    with open(obj_id_arg, 'r') as f:
        obj_id_list = []
        for line in f:
            line = line.strip()
            if line:
                try:
                    obj_id_list.append(float(line))
                except ValueError:
                    obj_id_list.append(line)
    obj_id_user = 'list'
elif obj_id_arg == 'all':
    obj_id_user = 'all'
else:
    # Single PROG_ID
    try:
        obj_id_user = float(obj_id_arg)
    except ValueError:
        obj_id_user = obj_id_arg

plot_cov_mat = sys.argv[2].lower() == 'cov' if len(sys.argv) > 2 else False

def get_line_wavel(w1, w2, lines_dict=[], exclude=[]):
    """
    Retrieve line names and wavelengths within a specified range from a given dict.

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

def spec_model_forward(params_dict, X_fit, z_guess, line_set_keys_j,
                           line_set_wavs_j, sig_disp_dict, broad_lines_config=None,
                           line_ratios_config=None, abs_lines_config=None,
                           return_components=False):
    """
    Forward model for spectrum line fitting (JAX-compatible).

    Parameters:
    -----------
    params_dict : dict
        Dictionary of model parameters
    X_fit : array
        Observed wavelengths
    z_guess : float
        Initial redshift guess (used as reference)
    line_set_keys_j : list
        Line names
    line_set_wavs_j : list
        Line rest-frame wavelengths
    sig_disp_dict : dict
        Instrumental dispersion for each line
    broad_lines_config : dict, optional
        Configuration for broad line components
    line_ratios_config : dict, optional
        Configuration for fixed line ratios {line_name: (ref_line, ratio)}
    abs_lines_config : dict, optional
        Configuration for absorption line components {line_name: True}
    return_components : bool, optional
        If True, return individual components in addition to total model

    Returns:
    --------
    Y_model : array
        Model flux
    components : dict (only if return_components=True)
        Dictionary of individual model components (continuum and lines)
    """

    # extract parameters
    cont_a = params_dict["cont_a"]
    cont_b = params_dict["cont_b"]
    z = params_dict["z"]
    fwhm = params_dict["fwhm"]

    # rest-frame wavelengths
    x_rf = X_fit / (1 + z)

    # continuum model
    x_cont = jnp.linspace(-1, 1, len(X_fit))
    Y_cont = cont_a * x_cont + cont_b

    # line components - track narrow and broad separately for absorption
    Y_lines_narrow = jnp.zeros_like(X_fit)
    Y_lines_broad = jnp.zeros_like(X_fit)

    # store individual components if requested
    if return_components:
        components = {'Y_cont': Y_cont, 'x_rf': x_rf}

    for line_key, line_wav in zip(line_set_keys_j, line_set_wavs_j):
        # line amplitude (log scale)
        # check if this line's amplitude is fixed by a ratio
        if line_ratios_config and line_key in line_ratios_config:
            ref_line, ratio = line_ratios_config[line_key]
            ref_amplitude_log = params_dict[f"{ref_line}_amplitude"]
            amplitude_log = ref_amplitude_log + jnp.log10(ratio)
        else:
            amplitude_log = params_dict[f"{line_key}_amplitude"]
        A = 10.0 ** amplitude_log

        # instrumental dispersion
        sig_disp = sig_disp_dict[line_key]

        # gaussian line profile
        sigma = fwhm / 2.355 / 3e5 * line_wav
        sigma_total = jnp.sqrt(sigma**2 + sig_disp**2)
        Y_line = A * jnp.exp(-0.5 * (x_rf - line_wav)**2 / sigma_total**2)

        Y_lines_narrow += Y_line

        if return_components:
            components[f'Y_{line_key}'] = Y_line

        # add broad component if configured
        if broad_lines_config and line_key in broad_lines_config:
            
            ## Broad Gaussian component
            #A2_log = params_dict[f"{line_key}_2_amplitude"]
            #A2 = 10.0 ** A2_log
            #
            #mu2 = params_dict[f"{line_key}_2_center"]
            #fwhm2 = params_dict[f"{line_key}_2_fwhm"]
            #sigma2 = fwhm2 / 2.355 / 3e5 * mu2
            #sigma2_total = jnp.sqrt(sigma2**2 + sig_disp**2) # add instrum. broadening
            #
            #Y_line_broad = A2 * jnp.exp(-0.5 * (x_rf-mu2)**2 / sigma2_total**2)
            #Y_lines_total += Y_line_broad
            
            # broad Exponentially modified Gaussian (Rusakov et al. 2026)
            A2_log = params_dict[f"{line_key}_2_amplitude"]
            A2 = 10.0 ** A2_log

            mu2 = params_dict[f"{line_key}_2_center"]
            fwhm2 = params_dict[f"{line_key}_2_fwhm"]
            sigma2 = fwhm2 / 2.355 / 3e5 * mu2
            tau2 = params_dict[f"{line_key}_2_decay"]
            v2 = 1.0 / (10.0 ** tau2)
            sigma2_total = jnp.sqrt(sigma2**2 + sig_disp**2) # add instrum. broadening
            
            x_zero_rf_var = x_rf - mu2
            exparg1 = 0.5 * sigma2_total**2 * v2**(-2) - x_zero_rf_var / v2
            exparg2 = 0.5 * sigma2_total**2 * v2**(-2) + x_zero_rf_var / v2
            erfarg1 = (x_zero_rf_var / sigma2_total - sigma2_total / v2) / jnp.sqrt(2)
            erfarg2 = -(x_zero_rf_var / sigma2_total + sigma2_total / v2) / jnp.sqrt(2)
            
            term1 = jnp.exp(exparg1) * 0.5 * (1 + jax.scipy.special.erf(erfarg1))
            term2 = jnp.exp(exparg2) * 0.5 * (1 + jax.scipy.special.erf(erfarg2))
            Y_line_broad = A2 * (term1 + term2)
            Y_lines_broad += Y_line_broad

            if return_components:
                components[f'Y_{line_key}_2'] = Y_line_broad

    # total line emission
    Y_lines_total = Y_lines_narrow + Y_lines_broad

    # total model (emission + continuum)
    Y_model = Y_lines_total + Y_cont

    # apply absorption if configured (multiplicative)
    # applied only to broad lines and continuum, not narrow lines
    if abs_lines_config:
        transmission = jnp.ones_like(X_fit)
        for line_key, line_wav in zip(line_set_keys_j, line_set_wavs_j):
            if line_key in abs_lines_config:
                # absorption parameters
                tau_abs = 10.0 ** params_dict[f"{line_key}_abs_tau"]
                mu_abs = params_dict[f"{line_key}_abs_center"]
                fwhm_abs = params_dict[f"{line_key}_abs_fwhm"]

                # instrumental dispersion
                sig_disp = sig_disp_dict[line_key]

                # gaussian optical depth profile
                sigma_abs = fwhm_abs / 2.355 / 3e5 * mu_abs
                sigma_abs_total = jnp.sqrt(sigma_abs**2 + sig_disp**2)
                tau_profile = tau_abs * jnp.exp(-0.5 * (x_rf - mu_abs)**2 / sigma_abs_total**2)

                # transmission = exp(-tau)
                line_transmission = jnp.exp(-tau_profile)
                transmission *= line_transmission

                if return_components:
                    components[f'Y_{line_key}_abs'] = line_transmission

        # apply absorption only to broad lines + continuum
        Y_model = Y_lines_narrow + (Y_lines_broad + Y_cont) * transmission

        if return_components:
            components['transmission'] = transmission

    if return_components:
        components['Y_lines'] = Y_lines_total
        components['Y_lines_narrow'] = Y_lines_narrow
        components['Y_lines_broad'] = Y_lines_broad
        return Y_model, components

    return Y_model


def chi_squared_loss(params_array, param_names, X_fit, Y_fit, Yerr_fit,
                     z_guess, line_set_keys_j, line_set_wavs_j,
                     sig_disp_dict, broad_lines_config, line_ratios_config,
                     abs_lines_config):
    """
    Chi-squared loss function for optimization.

    Parameters:
    -----------
    params_array : array
        Flattened array of parameters
    param_names : list
        Names of parameters (for unpacking)
    ... (other parameters same as spec_model_forward)

    Returns:
    --------
    chi2 : float
        Chi-squared value
    """

    # convert array to dictionary
    params_dict = dict(zip(param_names, params_array))

    # forward model
    Y_model = spec_model_forward(
        params_dict, X_fit, z_guess,
        line_set_keys_j, line_set_wavs_j,
        sig_disp_dict, broad_lines_config,
        line_ratios_config, abs_lines_config
    )

    # chi-squared
    chi2 = jnp.sum(((Y_fit - Y_model) / Yerr_fit) ** 2)

    return chi2


def get_initial_params(line_set_keys_j, line_set_wavs_j, z_guess, b_max,
                       broad_lines_config, line_ratios_config, abs_lines_config):
    """
    Create initial parameter dictionary with reasonable guesses.

    Returns:
    --------
    params_dict : dict
        Initial parameters
    param_names : list
        Ordered list of parameter names
    bounds : list of tuples
        Parameter bounds for optimization
    """

    params_dict = {}
    bounds = []

    # straight-line continuum parameters
    params_dict["cont_a"] = 0.0
    params_dict["cont_b"] = b_max
    bounds.append((-10, 10))  # cont_a
    bounds.append((-10, 10))  # cont_b

    # redshift
    params_dict["z"] = z_guess
    bounds.append((z_guess - 0.001, z_guess + 0.001))

    # fwhm (km/s)
    params_dict["fwhm"] = 200.0  # km/s
    bounds.append((10.0, 500.0))

    # line amplitudes (skip lines with fixed ratios - they're derived from reference lines)
    for line_key in line_set_keys_j:
        # only create amplitude parameter if not fixed by a ratio
        if not (line_ratios_config and line_key in line_ratios_config):
            params_dict[f"{line_key}_amplitude"] = 0.0  # log scale, so 10^0 = 1
            bounds.append((-3, 4))

        # broad component if configured (always create, even for ratio-constrained lines)
        if broad_lines_config and line_key in broad_lines_config:
            idx = line_set_keys_j.index(line_key)
            line_wav = line_set_wavs_j[idx]

            # exponentially modified gaussian (flexible, can reduce to a gaussian)
            params_dict[f"{line_key}_2_amplitude"] = -0.5
            params_dict[f"{line_key}_2_center"] = line_wav
            params_dict[f"{line_key}_2_fwhm"] = 200.0
            params_dict[f"{line_key}_2_decay"] = -1.0

            bounds.append((-3, 4))  # amplitude
            bounds.append((line_wav - 600/3e5*line_wav, 
                           line_wav + 600/3e5*line_wav))  # center
            bounds.append((100.0, 2000.0))  # fwhm
            bounds.append((-2, 0))  # decay

        # absorption component if configured (always create, even for ratio-constrained lines)
        if abs_lines_config and line_key in abs_lines_config:
            idx = line_set_keys_j.index(line_key)
            line_wav = line_set_wavs_j[idx]

            params_dict[f"{line_key}_abs_tau"] = -1.0  # log optical depth
            params_dict[f"{line_key}_abs_center"] = line_wav - 2.0  # slightly blueward
            params_dict[f"{line_key}_abs_fwhm"] = 150.0  # km/s

            bounds.append((-3, 2))  # tau (log scale): 0.001 to 100
            bounds.append((
                line_wav - 200/3e5*line_wav, 
                line_wav + 200/3e5*line_wav))  # center (allow shifts)
            bounds.append((50.0, 500.0))  # fwhm

    # create ordered parameter list
    param_names = list(params_dict.keys())

    return params_dict, param_names, bounds


def compute_hessian_uncertainties(loss_fn, optimal_params, gauss_newton_fallback_fn=None):
    """
    Compute parameter uncertainties from Hessian matrix.
    Falls back to Gauss-Newton approximation if Hessian is not positive-definite.

    Parameters:
    -----------
    loss_fn : callable
        Loss function (chi-squared)
    optimal_params : array
        Optimal parameter values
    gauss_newton_fallback_fn : callable, optional
        Function that returns Gauss-Newton covariance matrix if Hessian fails

    Returns:
    --------
    param_errors : array
        Parameter uncertainties (1-sigma)
    cov_matrix : array
        Covariance matrix
    used_gauss_newton : bool
        True if Gauss-Newton fallback was used
    """

    # compute Hessian using JAX
    hess_fn = jax.hessian(loss_fn)
    hess = hess_fn(optimal_params)

    # convert to numpy for inversion
    hess_np = np.array(hess)

    # regularize if needed (add small diagonal term if singular)
    used_gauss_newton = False
    try:
        cov_matrix = np.linalg.inv(hess_np / 2.0)  # factor of 2 for chi-squared
    except np.linalg.LinAlgError:
        if gauss_newton_fallback_fn is not None:
            cov_matrix = gauss_newton_fallback_fn()
            used_gauss_newton = True
        else:
            # old behavior: regularize
            reg = 1e-6 * np.eye(len(hess_np))
            cov_matrix = np.linalg.inv((hess_np + reg) / 2.0)

    # check if covariance is positive-definite
    if not used_gauss_newton:
        try:
            eigvals = np.linalg.eigvalsh(cov_matrix)
            hessian_ok = np.all(eigvals > 0)
        except np.linalg.LinAlgError:
            hessian_ok = False

        if not hessian_ok:
            if gauss_newton_fallback_fn is not None:
                cov_matrix = gauss_newton_fallback_fn()
                used_gauss_newton = True

    # extract uncertainties (diagonal of covariance matrix)
    param_errors = np.sqrt(np.abs(np.diag(cov_matrix)))

    return param_errors, cov_matrix, used_gauss_newton


def compute_gauss_newton_covariance(optimal_params, param_names, X_fit_jax,
                                     Yerr_fit_jax, z_guess, line_set_keys_j,
                                     line_set_wavs_j, sig_disp_dict, broad_lines_config,
                                     line_ratios_config, abs_lines_config):
    """
    Compute covariance matrix using Gauss-Newton approximation.
    More stable than full Hessian for low S/N cases.

    The Gauss-Newton approximation ignores second-order terms in the residuals,
    making it always positive semi-definite.

    Parameters:
    -----------
    optimal_params : array
        Optimal parameter values
    param_names : list
        Parameter names
    X_fit_jax : array
        Observed wavelengths (JAX array)
    Yerr_fit_jax : array
        Flux uncertainties (JAX array)
    z_guess : float
        Initial redshift guess
    line_set_keys_j : list
        Line names
    line_set_wavs_j : list
        Line wavelengths
    sig_disp_dict : dict
        Instrumental dispersion
    broad_lines_config : dict
        Broad line configuration
    line_ratios_config : dict
        Fixed line ratios
    abs_lines_config : dict
        Absorption line configuration

    Returns:
    --------
    cov_matrix : array
        Covariance matrix (Gauss-Newton approximation)
    """
    def model_fn(params):
        params_dict = dict(zip(param_names, params))
        return spec_model_forward(params_dict, X_fit_jax, z_guess,
                                      line_set_keys_j, line_set_wavs_j,
                                      sig_disp_dict, broad_lines_config,
                                      line_ratios_config, abs_lines_config)

    # jacobian: shape (n_obs, n_params)
    J = jax.jacfwd(model_fn)(optimal_params)
    J = np.array(J)

    # weight matrix (inverse variance)
    W = 1.0 / np.array(Yerr_fit_jax)**2

    # Gauss-Newton: Cov = (J^T W J)^-1
    JtWJ = J.T @ (W[:, None] * J)

    try:
        cov_matrix = np.linalg.inv(JtWJ)
    except np.linalg.LinAlgError:
        # add small regularization if singular
        reg = 1e-10 * np.eye(len(JtWJ))
        cov_matrix = np.linalg.inv(JtWJ + reg)

    return cov_matrix



def create_loss_and_grad(param_names, line_set_keys_j, line_set_wavs_j,
                         sig_disp_dict, broad_lines_config, line_ratios_config,
                         abs_lines_config):
    """
    Create a JIT-compiled loss and gradient function for a specific line configuration.

    JIT compilation provides significant speedup (~2-5x). Cache is cleared periodically
    to prevent memory buildup from accumulating compiled functions for different line
    configurations.

    Returns a function with signature: (params_array, X_fit, Y_fit, Yerr_fit, z_guess) -> (loss, grad)
    """
    @jax.jit
    def loss_and_grad(params_array, X_fit, Y_fit, Yerr_fit, z_guess):
        def loss_fn(p):
            return chi_squared_loss(p, param_names, X_fit, Y_fit, Yerr_fit,
                                   z_guess, line_set_keys_j, line_set_wavs_j,
                                   sig_disp_dict, broad_lines_config, line_ratios_config,
                                   abs_lines_config)
        return jax.value_and_grad(loss_fn)(params_array)

    return loss_and_grad


def compute_line_flux_error(optimal_params, param_names, X_fit_jax, z_guess,
                            line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                            broad_lines_config, line_ratios_config, abs_lines_config,
                            cov_matrix, line_key, is_line):
    """
    Compute flux and its uncertainty for a specific line using full error propagation.

    Parameters:
    -----------
    optimal_params : array
        Optimal parameter values
    param_names : list
        Parameter names
    X_fit_jax : array
        Observed wavelengths (JAX array)
    z_guess : float
        Initial redshift guess
    line_set_keys_j : list
        All line names in this fit
    line_set_wavs_j : list
        All line wavelengths in this fit
    sig_disp_dict : dict
        Instrumental dispersion
    broad_lines_config : dict
        Broad line configuration
    line_ratios_config : dict
        Fixed line ratios
    abs_lines_config : dict
        Absorption line configuration
    cov_matrix : array
        Covariance matrix
    line_key : str
        Name of the line to compute flux for
    is_line : array
        Boolean mask for pixels in line region

    Returns:
    --------
    flux_val : float
        Integrated line flux
    flux_err : float
        Flux uncertainty
    """

    def compute_line_flux(params_array_local):
        """Compute flux for a given parameter set"""
        params_dict_local = dict(zip(param_names, params_array_local))

        # forward model
        Y_model_local, components_local = spec_model_forward(
            params_dict_local, X_fit_jax, z_guess,
            line_set_keys_j, line_set_wavs_j,
            sig_disp_dict, broad_lines_config,
            line_ratios_config, abs_lines_config, return_components=True
        )

        # extract this specific line component
        Y_line_local = components_local[f'Y_{line_key}']

        # integrated flux
        flux = jnp.trapezoid(Y_line_local[is_line], x=X_fit_jax[is_line])
        return flux

    # compute flux at best-fit parameters
    flux_val = float(compute_line_flux(optimal_params))

    # compute gradient and error propagation
    flux_grad_fn = jax.grad(compute_line_flux)
    flux_gradient = np.array(flux_grad_fn(optimal_params))
    flux_var = flux_gradient @ cov_matrix @ flux_gradient
    flux_err = np.sqrt(np.abs(flux_var))

    return flux_val, flux_err


def compute_line_ew_error(optimal_params, param_names, X_fit_jax, z_guess,
                          line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                          broad_lines_config, line_ratios_config, abs_lines_config,
                          cov_matrix, line_key, is_line):
    """
    Compute equivalent width and its uncertainty for a specific line using full error propagation.

    Parameters:
    -----------
    (Same as compute_line_flux_error)

    Returns:
    --------
    eqw_val : float
        Equivalent width
    eqw_err : float
        EW uncertainty
    """

    def compute_line_ew(params_array_local):
        """Compute equivalent width for a given parameter set"""
        params_dict_local = dict(zip(param_names, params_array_local))

        # forward model
        Y_model_local, components_local = spec_model_forward(
            params_dict_local, X_fit_jax, z_guess,
            line_set_keys_j, line_set_wavs_j,
            sig_disp_dict, broad_lines_config,
            line_ratios_config, abs_lines_config, return_components=True
        )

        # extract this specific line component and continuum
        Y_line_local = components_local[f'Y_{line_key}']
        Y_cont_local = components_local['Y_cont']
        z_local = params_dict_local['z']

        # calcualte EW
        X_rest_local = X_fit_jax / (1 + z_local)
        ew_integrand = Y_line_local[is_line] / Y_cont_local[is_line]
        ew = jnp.trapezoid(ew_integrand, x=X_rest_local[is_line])
        return ew

    # calculate EW at best-fit parameters
    eqw_val = float(compute_line_ew(optimal_params))

    # calculate gradient and error propagation
    ew_grad_fn = jax.grad(compute_line_ew)
    ew_gradient = np.array(ew_grad_fn(optimal_params))
    ew_var = ew_gradient @ cov_matrix @ ew_gradient
    eqw_err = np.sqrt(np.abs(ew_var))

    return eqw_val, eqw_err


def compute_subtracted_lines_error(optimal_params, param_names, X_fit_jax, z_guess,
                                   line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                                   broad_lines_config, line_ratios_config, abs_lines_config,
                                   cov_matrix, line_keys_sub, is_line):
    """
    Compute uncertainty in subtracted line flux using full error propagation.

    Parameters:
    -----------
    optimal_params : array
        Optimal parameter values
    param_names : list
        Parameter names
    X_fit_jax : array
        Observed wavelengths (JAX array)
    z_guess : float
        Initial redshift guess
    line_set_keys_j : list
        All line names in this fit
    line_set_wavs_j : list
        All line wavelengths in this fit
    sig_disp_dict : dict
        Instrumental dispersion
    broad_lines_config : dict
        Broad line configuration
    line_ratios_config : dict
        Fixed line ratios
    abs_lines_config : dict
        Absorption line configuration
    cov_matrix : array
        Covariance matrix
    line_keys_sub : list
        Names of lines to subtract
    is_line : array
        Boolean mask for pixels in line region

    Returns:
    --------
    ye_sub : array
        Uncertainty in subtracted flux at each pixel
    """

    def compute_subtracted_flux(params_array_local):
        """Compute total flux from all other line components"""
        params_dict_local = dict(zip(param_names, params_array_local))

        # forward model
        Y_model_local, components_local = spec_model_forward(
            params_dict_local, X_fit_jax, z_guess,
            line_set_keys_j, line_set_wavs_j,
            sig_disp_dict, broad_lines_config,
            line_ratios_config, abs_lines_config, return_components=True
        )

        # sum flux from all other line components
        y_sub_local = jnp.zeros(len(X_fit_jax))
        for k_comp in line_keys_sub:
            if f'Y_{k_comp}' in components_local:
                y_sub_local += components_local[f'Y_{k_comp}']

        return y_sub_local[is_line]

    # compute Jacobian (gradient for each pixel)
    jacobian_fn = jax.jacfwd(compute_subtracted_flux)
    jacobian = np.array(jacobian_fn(optimal_params))  # Shape: (n_obs_line, npar)

    # error propagation: sigma_sub[i] = J[i,:] @ Cov @ J[i,:]^T for each pixel i
    ye_sub_sq = np.einsum('ip,pq,iq->i', jacobian, cov_matrix, jacobian)
    ye_sub = np.sqrt(np.abs(ye_sub_sq))

    return ye_sub


if __name__ == "__main__":

    # Set random seeds for reproducibility
    np.random.seed(42)
    rng_key = jax.random.PRNGKey(42)

    #=== set up paths ===============================================
    if not os.path.exists(cfg.fpath_outputs):
        os.makedirs(cfg.fpath_outputs)

    # set up logging
    log_file = os.path.join(cfg.fpath_outputs, 'fit_jax.log')

    # create logger - file only, no console output
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Don't propagate to root logger

    # file handler - logs everything (INFO, WARNING, ERROR)
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # add only file handler (no console output from logger)
    logger.addHandler(file_handler)

    # log session start
    logger.info("="*80)
    logger.info(f"Starting optimization session")
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info("="*80)

    #=== process user-selected lines ================================
    # sort the lines by wavelength
    line_keys = cfg.line_keys
    line_wavs = [cl.lines_dict[k][0][0] for k in line_keys]
    idxs_sorted = np.argsort(line_wavs)
    line_wavs = np.array(line_wavs)[idxs_sorted]
    line_keys = np.array(line_keys)[idxs_sorted]

    # do individ. lines have neighbours? make sets of nearby lines.
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

    # remove duplicate lists from line_set_keys
    seen = set()
    line_set_keys = [
        sublist for sublist in line_set_keys
        if tuple(sublist) not in seen and not seen.add(tuple(sublist))
    ]

    # list of lines excluded from fit
    lines_excl = []
    for k in cl.lw_exclude_lines:
        lines_excl.extend(cl.lw[k])

    #=== sample selection ================================================
    df_sample = pd.read_csv(cfg.fpath_spec, comment='#')
    logger.info(f"Data table size: {len(df_sample)}")

    print(f"Loaded {len(df_sample)} spectra from {cfg.fpath_spec}")
    print(f"Logging to: {log_file}")

    skipped_file = {}
    skipped_file['spec_v1'] = []
    skipped_file['nospec'] = []
    skipped_file['fitted'] = []
    skipped_file['nolines'] = []
    skipped_file['nospecrange'] = []
    skipped_file['failed'] = []

    # counter for successfully fitted spectra
    n_success = 0

    fig_axis_labels = {
        'fwhm': r'FWHM (km s$^{-1}$)',
        'cont_a': r'$a$ (continuum)',
        'cont_b': r'$b$ (continuum)',
        'decay': r'$\tau$ (decay)',
        'z': r'Redshift',
    }

    # initialize progress bar
    n_obj = df_sample.shape[0]
    pbar = tqdm(total=n_obj, desc="Optimizing spectra", unit="spectrum",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')

    # loop over individual spectra
    i_fit = 0
    for grating, group_grat in df_sample.groupby('grating'):
        for obj_id, group_obj in group_grat.groupby('PROG_ID'):
            for idx_row, row in group_obj.iterrows():

                i_fit += 1

                # fit either all, a list, or a specific spectrum
                if obj_id_user == 'all':
                    pass
                elif obj_id_user == 'list':
                    if obj_id not in obj_id_list:
                        continue
                elif obj_id != obj_id_user:
                    continue

                logger.info(f"Processing {grating} / ID {obj_id} ({i_fit}/{n_obj})")
                pbar.set_postfix_str(f"ID:{obj_id} {grating}")

                fname_spec = row['file'].replace('.spec.fits', '')

                #=== load a spectrum
                # sample_info contains arrays of sample attributes
                # (e.g. file names, root names, indices, etc.)
                sample_info = {
                    'fnames': np.array([row.file]),
                    'roots': np.array([row.root]),
                    'idxs': np.array([idx_row]),
                    'obj_ids':np.array([row.PROG_ID]),
                    'z_input': np.array([row.z]),
                }
                
                try:
                    sampleFit = SpectrumSampleFit(
                        fit_window_bins=None, flux_norm=None,
                        download_data=False, verbose=False,
                        **sample_info
                    )
                except:
                    try:
                        sampleFit = SpectrumSampleFit(
                            fit_window_bins=None, flux_norm=None,
                            download_data=True, verbose=False,
                            **sample_info
                        )
                    except:
                        skipped_file['nospec'].append(fname_spec)
                        logger.warning(f"Skipping {fname_spec}: No spec could be loaded")
                        pbar.update(1)
                        continue

                valid = sampleFit.valid[0]
                X = sampleFit.x[0][valid]
                Y = sampleFit.y[0][valid]
                Yerr = sampleFit.ye[0][valid]
                z_guess = sampleFit.z_input[0]
                X_rf = X / (1 + z_guess)
                xmin_spec, xmax_spec = X_rf.min(), X_rf.max()
                
                #=== spec fit windows
                line_keys_spec, line_wavs_cur = get_line_wavel(
                    xmin_spec, xmax_spec,
                    lines_dict=lines_all_dict,
                    exclude=cl.lw_exclude_sets
                )

                if len(line_keys_spec) == 0:
                    _line_keys = list(lines_all_dict.keys())
                    skipped_file['nolines'].append(fname_spec+f'\t({_line_keys})')
                    logger.warning(f"Skipping {fname_spec}: No lines in range ({_line_keys})")
                    pbar.update(1)
                    continue

                # create fit windows around the fitted lines
                fit_range = [[xmin_spec, xmax_spec]]
                fit_edges = exclude_fit_windows(
                    fit_range, hwhm=cfg.line_range_kms,
                    lines=np.concatenate(line_wavs_cur)
                )
                fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)

                # update the selected spec data
                sampleFit.update_fit_window(fit_edges)
                mask = sampleFit.in_window[0][valid]
                if not mask.any():
                    skipped_file['nospecrange'].append(fname_spec)
                    logger.warning(f"Skipping {fname_spec}: No spec range available")
                    pbar.update(1)
                    continue

                X = X[mask]
                X_rf = X / (1 + z_guess)
                xmin_mask, xmax_mask = X_rf.min(), X_rf.max()

                # get emission lines in the spec range
                line_keys_spec, line_wavs_cur = get_line_wavel(
                    xmin_mask, xmax_mask,
                    lines_dict=lines_all_dict,
                    exclude=cl.lw_exclude_sets
                )

                if len(line_keys_spec) == 0:
                    skipped_file['nolines'].append(fname_spec)
                    logger.warning(f"Skipping {fname_spec}: No lines in spec range")
                    pbar.update(1)
                    continue

                # keep the line sets that are observed
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

                    # get the wavelength range containing the fitted lines
                    fit_edges = exclude_fit_windows(
                        fit_range, hwhm=cfg.line_range_kms, lines=line_set_wavs_j
                    )
                    fit_edges = fit_edges.flatten()[1:-1].reshape(-1, 2)

                    # update the selected spec data
                    sampleFit.update_fit_window(fit_edges)
                    mask = sampleFit.in_window[0]
                    if not (mask & valid).any():
                        skipped_file['nospecrange'].append(fname_spec)
                        logger.warning(f"Skipping line set {j+1}/{n_sets}: No spec range available")
                        continue

                    X_fit = sampleFit.x[0][mask & valid]
                    Y_fit = sampleFit.y[0][mask & valid]
                    Yerr_fit = sampleFit.ye[0][mask & valid]
                    
                    nobs = len(X_fit)
                    if nobs < 5:
                        logger.warning(f"Skipping line set {j+1}/{n_sets}: Less than 5 data points (nobs={nobs})")
                        continue
                    
                    # scale uncertainties if they are under/over-estimated
                    # these estimates are based on a test in Rusakov et al. 2026b (in prep.)
                    #G140: median redchisq = 0.757, k=0.870 (1490 fits, 738 spectra)
                    #G235: median redchisq = 0.798, k=0.893 (3554 fits, 1261 spectra)
                    #G395: median redchisq = 0.839, k=0.916 (9007 fits, 3577 spectra)
                    #if 'g140' in grating.lower():
                    #    Yerr_fit *= 0.870
                    #elif 'g235' in grating.lower():
                    #    Yerr_fit *= 0.893
                    #elif 'g395' in grating.lower():
                    #    Yerr_fit *= 0.916
                    
                    # renormalize flux in the fit window (to order unity)
                    p = np.round(abs(np.log10(Y_fit.max())), 0)
                    flux_norm = 10**p
                    Y_fit *= flux_norm
                    Yerr_fit *= flux_norm
                    flux_norm = flux_norm * sampleFit.flux_norm[0].item()
                    
                    # check if already fitted
                    j_iter = 0
                    lines_str = '_'.join(line_set_keys_j)
                    model_label = f'{obj_id}-{fname_spec}-{grating}-iter{j_iter}-{lines_str}-optimize'
                    model_label_any = f'{obj_id}-{fname_spec}-{grating}-iter*-{lines_str}-optimize'
                    fpaths = os.path.join(cfg.fpath_outputs, f"{model_label_any}-line_props.fits")

                    if len(glob.glob(fpaths)) > 0:
                        logger.info(f"Iteration {j_iter}, Set {j+1}/{n_sets}, Lines: {line_set_keys_j} - Already fitted, skipping")
                        skipped_file['fitted'].append(fname_spec)
                        continue

                    logger.info(f"Iteration {j_iter}, Set {j+1}/{n_sets}, Optimizing lines: {line_set_keys_j}")
                    pbar.set_postfix_str(f"ID:{obj_id} {grating} Lines:{','.join(line_set_keys_j[:2])}")

                    # pickle the fitted data
                    d = {'X': X_fit, 'Y': Y_fit, 'Yerr': Yerr_fit,
                        'z_guess': z_guess,
                        'flux_norm': flux_norm,
                    }
                    fname = f"{model_label}-data.pckl"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    with open(fpath, 'wb') as f:
                        pickle.dump(d, f)

                    #=== handle unresolved lines
                    fwhm_intrum = []
                    for line_key, line_wav in zip(line_set_keys_j, line_set_wavs_j):
                        sig_disp = sampleFit.get_dispersion([obj_id], line_wav, [z_guess])[0]
                        fwhm_disp_kms = sig_disp / line_wav * 3e5 * 2.355
                        fwhm_intrum.append(fwhm_disp_kms)

                    line_set_wavs_j = np.array(line_set_wavs_j)
                    fwhm_intrum = np.array(fwhm_intrum)
                    sep_fwhm_kms = np.diff(line_set_wavs_j) / line_set_wavs_j[:-1] * 3e5
                    unresolved = 1.5 * fwhm_intrum[:-1] > sep_fwhm_kms

                    # if unresolved, fit lines as a single blended line at a mean wavel.
                    if unresolved.any():
                        idx2remove = np.argwhere(unresolved).flatten() + 1
                        for idx in idx2remove:
                            merged_wav = np.mean(line_set_wavs_j[idx-1:idx+1])
                            line_set_wavs_j[idx-1] = merged_wav
                        line_set_wavs_j = np.delete(line_set_wavs_j, idx2remove)

                        lines_combined = [line_set_keys_j[i] for i in np.insert(idx2remove, 0, idx2remove[0]-1)]
                        merged_lines = '_'.join(lines_combined)
                        line_set_keys_j[idx2remove[0]-1] = merged_lines
                        # Remove the lines that were merged (all except the first one which now holds the merged name)
                        line_set_keys_j = [key for i, key in enumerate(line_set_keys_j)
                                          if i not in idx2remove]
                        logger.info(f"Unresolved lines: {lines_combined} -- fitting as {merged_lines}")

                    line_set_wavs_j = list(line_set_wavs_j)

                    # prepare instrumental dispersion dictionary
                    sig_disp_dict = {}
                    for line_key, line_wav in zip(line_set_keys_j, line_set_wavs_j):
                        sig_disp = sampleFit.get_dispersion([obj_id], line_wav, [z_guess])[0]
                        sig_disp_dict[line_key] = sig_disp

                    # check for broad lines
                    broad_lines_config = {}
                    if hasattr(cfg, 'broad_lines'):
                        for line_key in line_set_keys_j:
                            if line_key in cfg.broad_lines:
                                broad_lines_config[line_key] = True

                    # check for fixed line ratios
                    line_ratios_config = {}
                    if hasattr(cfg, 'line_ratios'):
                        for line_key in line_set_keys_j:
                            if line_key in cfg.line_ratios:
                                ref_line, ratio = cfg.line_ratios[line_key]
                                # only apply if reference line is also in the line set
                                if ref_line in line_set_keys_j:
                                    line_ratios_config[line_key] = (ref_line, ratio)
                                    logger.info(f"Fixed line ratio: {line_key}/{ref_line} = {ratio:.3f}")

                    # enable absorption lines? check config
                    abs_lines_config = {}
                    if hasattr(cfg, 'abs_lines'):
                        for line_key in line_set_keys_j:
                            if line_key in cfg.abs_lines:
                                abs_lines_config[line_key] = True
                                logger.info(f"Absorption component enabled for: {line_key}")

                    b_max = np.median(Y_fit)

                    # convert fitted data to JAX arrays
                    X_fit_jax = jnp.array(X_fit)
                    Y_fit_jax = jnp.array(Y_fit)
                    Yerr_fit_jax = jnp.array(Yerr_fit)

                    #=== set up optimization
                    # get initial parameters and bounds
                    params_init_dict, param_names, bounds = get_initial_params(
                        line_set_keys_j, line_set_wavs_j, z_guess, b_max,
                        broad_lines_config, line_ratios_config, abs_lines_config
                    )

                    # convert to array
                    params_init_array = np.array([params_init_dict[k] for k in param_names])
                    npar = len(param_names)

                    logger.info(f"Optimization setup: nobs={nobs}, npar={npar}")

                    # create loss and gradient function for this line configuration
                    loss_grad_fn = create_loss_and_grad(
                        param_names, line_set_keys_j, line_set_wavs_j,
                        sig_disp_dict, broad_lines_config, line_ratios_config,
                        abs_lines_config
                    )

                    #=== optimization with scipy
                    # check if global optimization is enabled, i.e. a 2-step optimization:
                    # 1) differential evolution for global search (no gradient, more robust to local minima)
                    # 2) trust-constr for local polishing (with gradient, faster convergence)
                    # otherwise, use only trust-constr
                    use_global_opt = getattr(cfg, 'use_global_optimization', False)

                    try:
                        # step 1: global optimization with differential evolution (optional)
                        if use_global_opt:
                            logger.info("Running differential evolution for global search...")

                            # loss function for differential evolution (no gradient)
                            def loss_fn_de(p):
                                loss, _ = loss_grad_fn(p, X_fit_jax, Y_fit_jax, Yerr_fit_jax, z_guess)
                                loss_val = float(loss)
                                # Return large penalty for invalid parameters (NaN/Inf from EMG overflow)
                                if np.isnan(loss_val) or np.isinf(loss_val):
                                    return 1e30
                                return loss_val

                            result_global = differential_evolution(
                                func=loss_fn_de,
                                bounds=bounds,
                                maxiter=getattr(cfg, 'de_maxiter', 200),
                                seed=42,
                                workers=1,  # JAX handles parallelism internally
                                updating='deferred',
                                polish=False,  # we'll polish with trust-constr
                                tol=0.01,
                                atol=0.01,
                            )

                            x0_for_polish = result_global.x
                            logger.info(f"Differential evolution: chi2={result_global.fun:.2f}, nfev={result_global.nfev}")
                        else:
                            x0_for_polish = params_init_array

                        # step 2: local optimization (polishing) with trust-constr
                        method = 'trust-constr'
                        result = minimize(
                            fun=lambda p: loss_grad_fn(p, X_fit_jax, Y_fit_jax, Yerr_fit_jax, z_guess),
                            x0=x0_for_polish,
                            method=method,
                            bounds=bounds,
                            jac=True,  # We provide gradient
                            options={'maxiter': 1000}
                        )

                        if not result.success:
                            logger.warning(f"Optimization did not converge: {result.message}")

                        optimal_params = result.x
                        chi2_min = result.fun

                    except Exception as e:
                        logger.error(f"Optimization failed for {obj_id}: {e}")
                        skipped_file['failed'].append(f"{fname_spec} - {e}")
                        with open(os.path.join(cfg.fpath_outputs, 'skipped_files.txt'), 'a') as f:
                            f.write(f"{obj_id} - Optimization failed: {e}\n")
                        continue

                    #=== compute uncertainties from Hessian ===
                    try:
                        # create loss function for Hessian computation
                        def loss_fn_for_hessian(params):
                            loss, _ = loss_grad_fn(params, X_fit_jax, Y_fit_jax, Yerr_fit_jax, z_guess)
                            return loss

                        # create Gauss-Newton fallback function
                        def gauss_newton_fallback():
                            return compute_gauss_newton_covariance(
                                optimal_params, param_names, X_fit_jax,
                                Yerr_fit_jax, z_guess, line_set_keys_j,
                                line_set_wavs_j, sig_disp_dict, broad_lines_config,
                                line_ratios_config, abs_lines_config
                            )

                        param_errors, cov_matrix, used_gauss_newton = compute_hessian_uncertainties(
                            loss_fn_for_hessian, optimal_params, gauss_newton_fallback
                        )
                        if used_gauss_newton:
                            logger.info(f"Hessian not positive-definite, used Gauss-Newton covariance")
                        else:
                            logger.info(f"Computed Hessian uncertainties")
                    except Exception as e:
                        logger.warning(f"Failed to compute uncertainties: {e}, setting to NaN")
                        param_errors = np.full_like(optimal_params, np.nan)
                        cov_matrix = np.full((npar, npar), np.nan)

                    # convert optimal parameters to dictionary
                    params_best = dict(zip(param_names, optimal_params))
                    params_errors = dict(zip(param_names, param_errors))

                    # add fixed ratio line amplitudes (derived from reference lines)
                    if line_ratios_config:
                        for line_key, (ref_line, ratio) in line_ratios_config.items():
                            ref_amplitude_log = params_best[f"{ref_line}_amplitude"]
                            params_best[f"{line_key}_amplitude"] = ref_amplitude_log + np.log10(ratio)
                            # propagate errors from reference line
                            params_errors[f"{line_key}_amplitude"] = params_errors[f"{ref_line}_amplitude"]

                    # for compatibility, create params_best_full with symmetric errors
                    params_best_full = {}
                    for name in params_best.keys():
                        if name in params_errors:
                            params_best_full[name] = np.array([
                                params_best[name],       # value
                                params_errors[name],     # upper error
                                params_errors[name]      # lower error (symmetric)
                            ])
                        else:
                            # for derived parameters (shouldn't happen now, but safe)
                            params_best_full[name] = np.array([
                                params_best[name],
                                0.0,
                                0.0
                            ])

                    # calculate best-fit model spectrum with components
                    Y_model_best, components = spec_model_forward(
                        params_best, X_fit_jax, z_guess,
                        line_set_keys_j, line_set_wavs_j,
                        sig_disp_dict, broad_lines_config,
                        line_ratios_config, abs_lines_config, return_components=True
                    )
                    Y_model_best = np.array(Y_model_best)

                    # extract components (convert from JAX to numpy)
                    Y_cont_best = np.array(components['Y_cont'])
                    x_rf_best = np.array(components['x_rf'])
                    Y_components_best = {k: np.array(v) for k, v in components.items()
                                        if k.startswith('Y_') and k not in ['Y_cont', 'Y_lines']}
                    Y_lines_best = np.array(components['Y_lines'])

                    # residuals statistics
                    diff_chi = (Y_fit - Y_model_best) / Yerr_fit
                    chisq = np.sum(diff_chi**2)
                    redchisq = chisq / (nobs - npar)
                    bic = chisq + npar * np.log(nobs)
                    aic = chisq + 2 * npar

                    logger.info(f"Optimization complete: chi2={chisq:.2f}, redchisq={redchisq:.2f}")

                    #=== tabulate & save outputs
                    columns = []
                    values = []

                    # model columns
                    cols_model = ['PROG_ID', 'model_label', 
                                  'redchisq', 'nobs', 'npar',
                                  'step_method']
                    vals_model = [obj_id, model_label, 
                                  redchisq, nobs, npar, 
                                  method]
                    columns += cols_model
                    values += vals_model

                    # observations columns
                    cols_fit_edges = [f'fit_edge_{i}' for i in range(len(fit_edges.flatten()))]
                    vals_fit_edges = [edge for edge in fit_edges.flatten()]
                    flux_norm_val = flux_norm
                    columns += cols_fit_edges + ['flux_norm']
                    values += vals_fit_edges + [np.log10(flux_norm_val)]
                    
                    # best-fit params columns
                    params_names = list(params_best_full.keys())
                    params_sigup_names = [f"{p}_sigup" for p in params_names]
                    params_siglo_names = [f"{p}_siglo" for p in params_names]
                    cols_parnames = np.c_[params_names, params_sigup_names, params_siglo_names].flatten()
                    columns += list(cols_parnames)

                    params_values = [params_best_full[p][0] for p in params_names]
                    params_sigup = [params_best_full[p][1] for p in params_names]
                    params_siglo = [params_best_full[p][2] for p in params_names]
                    vals_par = np.c_[params_values, params_sigup, params_siglo].flatten()
                    values += list(vals_par)

                    # line S/N and Flux (simplified - no posterior samples)
                    cols_line, vals_line = [], []
                    sig_integ = 3.0
                    X_rf = x_rf_best

                    # get list of all line keys including broad components
                    all_line_keys = line_set_keys_j.copy()
                    all_line_wavs = line_set_wavs_j.copy()
                    if broad_lines_config:
                        for line_key in line_set_keys_j:
                            if line_key in broad_lines_config:
                                all_line_keys.append(f"{line_key}_2")
                                idx = line_set_keys_j.index(line_key)
                                all_line_wavs.append(line_set_wavs_j[idx])

                    # calculate line-specific quantities (SNR, flux, EW) 
                    # for each line or a blended line set
                    for k, w in zip(all_line_keys, all_line_wavs):
                        disp_sig_ang = sampleFit.get_dispersion([obj_id], w, [params_best['z']])

                        velocity_sig_ang = params_best['fwhm'] / 2.355 / 3e5 * w
                        total_sig_ang = np.sqrt(disp_sig_ang**2 + velocity_sig_ang**2)
                        is_line = (X_rf > (w - sig_integ * total_sig_ang)) & (X_rf < (w + sig_integ * total_sig_ang))
                        is_cont = ~is_line
                        n_obs_line = is_line.sum()

                        if n_obs_line >= 4:
                            logger.debug(f'Line {k}: n_data={n_obs_line}')

                            # continuum level and uncertainty
                            y_cont = Y_cont_best[is_line]

                            # calculate continuum variance at each pixel
                            # Y_cont = cont_a * x_cont + cont_b
                            x_cont_line = np.linspace(-1, 1, len(X_fit))[is_line]
                            idx_cont_a = param_names.index('cont_a')
                            idx_cont_b = param_names.index('cont_b')
                            var_cont = (x_cont_line**2 * param_errors[idx_cont_a]**2 +
                                       param_errors[idx_cont_b]**2 +
                                       2 * x_cont_line * cov_matrix[idx_cont_a, idx_cont_b])
                            ye_cont_flux = np.sqrt(np.abs(var_cont))

                            if len(all_line_keys) > 1:
                                # subtract other lines
                                line_keys_sub = [key for key in all_line_keys if key != k]
                                y_sub = np.zeros(n_obs_line)

                                # get subtracted flux at best-fit parameters
                                for k_comp in line_keys_sub:
                                    if f'Y_{k_comp}' in Y_components_best:
                                        y_sub += Y_components_best[f'Y_{k_comp}'][is_line]

                                # error propagation for subtracted lines
                                ye_sub = compute_subtracted_lines_error(
                                    optimal_params, param_names, X_fit_jax, z_guess,
                                    line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                                    broad_lines_config, line_ratios_config, abs_lines_config,
                                    cov_matrix, line_keys_sub, is_line
                                )

                                y_flux_line = (Y_fit[is_line] - y_sub) - y_cont
                                ye_flux_line = np.sqrt(Yerr_fit[is_line]**2 + ye_cont_flux**2 + ye_sub**2)
                            else:
                                y_flux_line = Y_fit[is_line] - y_cont
                                ye_flux_line = np.sqrt(Yerr_fit[is_line]**2 + ye_cont_flux**2)

                            # matched-filter (optimal) S/N calculation
                            # use the expected line profile from best-fit model as filter
                            T_line = Y_components_best[f'Y_{k}'][is_line]

                            # normalize filter to unit integral for proper weighting
                            T_integral = simpson(T_line, x=X_fit[is_line])
                            if T_integral > 0:
                                T_norm = T_line / T_integral
                            else:
                                T_norm = T_line  # fallback if integral is zero

                            # matched filter S/N: optimal weighting by filter shape and inverse variance
                            inv_var = 1.0 / ye_flux_line**2
                            numerator = np.sum(y_flux_line * T_norm * inv_var)
                            denominator = np.sqrt(np.sum(T_norm**2 * inv_var))

                            if denominator > 0:
                                sn_line = numerator / denominator
                            else:
                                # fallback to raw S/N if denominator is zero
                                sn_line = np.sum(y_flux_line) / np.sqrt(np.sum(ye_flux_line**2))

                            # compute integrated line flux and uncertainty
                            flux_val, flux_err = compute_line_flux_error(
                                optimal_params, param_names, X_fit_jax, z_guess,
                                line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                                broad_lines_config, line_ratios_config, abs_lines_config,
                                cov_matrix, k, is_line
                            )

                            # compute line EW and uncertainty
                            eqw_val, eqw_err = compute_line_ew_error(
                                optimal_params, param_names, X_fit_jax, z_guess,
                                line_set_keys_j, line_set_wavs_j, sig_disp_dict,
                                broad_lines_config, line_ratios_config, abs_lines_config,
                                cov_matrix, k, is_line
                            )
                        else:
                            sn_line = np.nan
                            flux_val, flux_err = np.nan, np.nan
                            eqw_val, eqw_err = np.nan, np.nan
                            logger.warning(f'Line {k}: only {n_obs_line} valid data points')
                            continue

                        # calculate S/N of the continuum
                        if is_cont.sum() > 0:
                            weights = 1.0 / Yerr_fit[is_cont]**2
                            wmean = (Y_fit[is_cont] * weights).sum() / weights.sum()
                            wsigma = np.sqrt(1 / weights.sum())
                            sn_cont = wmean / wsigma
                        else:
                            sn_cont = np.nan

                        # append values to tabulate
                        cols_line += [f'{k}_SN', f'{k}_cont_wmean_SN']
                        vals_line += [sn_line, sn_cont]

                        cols_line += [f'{k}_flux', f'{k}_flux_sigup', f'{k}_flux_siglo']
                        vals_line += [flux_val, flux_err, flux_err]

                        cols_line += [f'{k}_ew', f'{k}_ew_sigup', f'{k}_ew_siglo']
                        vals_line += [eqw_val, eqw_err, eqw_err]

                        cols_line += [f'{k}_ndata']
                        vals_line += [n_obs_line]

                        cols_line += [f'{k}_sig_disp']
                        vals_line += [float(np.atleast_1d(disp_sig_ang)[0])]

                    columns += cols_line
                    values += vals_line

                    dtypes = []
                    for val in values:
                        if isinstance(val, str):
                            dtypes.append(f'<U{len(val)}')
                        elif isinstance(val, (float, np.floating)):
                            dtypes.append('f8')
                        elif isinstance(val, (int, np.integer)):
                            dtypes.append('i8')
                        else:
                            # default to float for anything else (handles NaN, etc.)
                            dtypes.append('f8')

                    # make a table and save
                    tab_line_props = Table(data=np.array(values)[None, :],
                                        dtype=dtypes, names=columns)
                    fname = f"{model_label}-line_props.fits"
                    fpath = os.path.join(cfg.fpath_outputs, fname)

                    # if encounter astropy overwrite issue
                    if os.path.exists(fpath):
                        os.remove(fpath)

                    tab_line_props.write(fpath, overwrite=True)
                    logger.info(f"Saved results: {fname}")

                    # Increment success counter
                    n_success += 1

                    #============== Make figures =================

                    #=== Figure 1: Correlation matrix from covariance
                    if plot_cov_mat and (npar > 1):
                        # Compute correlation matrix
                        corr_matrix = cov_matrix / np.outer(param_errors, param_errors)

                        fig, ax = plt.subplots(1, 1, figsize=(8, 7))
                        im = ax.imshow(
                            corr_matrix, cmap='RdBu_r', 
                            vmin=-1, vmax=1, aspect='auto')
                        ax.set_xticks(np.arange(npar))
                        ax.set_yticks(np.arange(npar))
                        ax.set_xticklabels(
                            param_names, rotation=45, ha='right', fontsize=8)
                        ax.set_yticklabels(param_names, fontsize=8)
                        plt.colorbar(im, ax=ax, label='Correlation')
                        ax.set_title('Parameter Correlation Matrix')

                        plt.tight_layout()
                        fname = f"{model_label}-correlation.png"
                        fpath = os.path.join(cfg.fpath_outputs, fname)
                        plt.savefig(fpath, bbox_inches='tight', dpi=100)
                        plt.close()

                    #=== Figure 2: best-fit model
                    fig, (ax, axd) = plt.subplots(2, 1, figsize=(4, 3), sharex=True,
                                                 gridspec_kw={'height_ratios': [1.5, 1], 'hspace': 0.0})
                    xmin_fit, xmax_fit = X_rf.min(), X_rf.max()
                    ax.set_xlim(xmin_fit, xmax_fit)
                    ax.set_ylabel(r'$F_{\lambda}$')
                    axd.set_xlabel(r'Rest-frame Wavelength [${\rm \AA}$]')
                    axd.set_ylabel(r'$\chi$')

                    ax.step(X_rf, Y_fit, lw=0.5, color='k', where='mid')
                    ax.fill_between(X_rf, Y_fit-Yerr_fit, Y_fit+Yerr_fit,
                                   lw=0, color='k', step='mid', alpha=0.3)

                    ax.step(X_rf, Y_model_best, color='C1', lw=1, zorder=10,
                           where='mid', label='Best-fit model')

                    # Plot line components
                    for k, Y_comp_best in Y_components_best.items():
                        ax.plot(X_rf, Y_comp_best, lw=1, label=k.replace('Y_', ''), ls='--')

                    axd.errorbar(X_rf, diff_chi, yerr=1.0, fmt='o', color='k',
                               ms=1.5, elinewidth=1, zorder=10)
                    axd.text(0.02, 0.85, rf"$\chi^2_{{\nu}}: {redchisq:.2f}$",
                           transform=axd.transAxes, fontsize=7, color='k')
                    axd.set_ylim(-7, 10)
                    axd.set_yticks(np.arange(-5, 6, 5), minor=False)
                    axd.set_yticks(np.arange(-5, 6, 1), minor=True)
                    axd.grid(which='both', axis='y', lw=1, ls=':', color='k', alpha=0.3)
                    axd.axhline(0.0, ls='-', color='k', lw=0.5)

                    # Annotate spec lines
                    [ax.axvline(w, ls='--', lw=0.5, c='C4') for w in line_set_wavs_j]
                    title = ', '.join(line_set_keys_j)
                    ax.set_title(title, fontsize=8)
                    ax.legend(fontsize=8)

                    fname = f"{model_label}-best_fit.png"
                    fpath = os.path.join(cfg.fpath_outputs, fname)
                    plt.savefig(fpath, bbox_inches='tight', dpi=100)
                    plt.close()

                # clear JAX cache after each spectrum to prevent memory buildup
                # Each spectrum may have different line configurations requiring new compilations
                if i_fit % 10 == 0:  # Every N spectra
                    jax.clear_caches()

                # update progress bar after processing each spectrum
                pbar.update(1)

    # close progress bar
    pbar.close()

    # log skipped files
    logger.info("="*80)
    logger.info("Session summary:")
    for key, values in skipped_file.items():
        if len(values) > 0:
            logger.info(f"{key}: {len(values)} spectra")
            for value in values:
                logger.debug(f"  - {value}")

    skipped_file_log_path = os.path.join(cfg.fpath_outputs, 'skipped_files.txt')
    with open(skipped_file_log_path, 'a') as log_file:
        log_file.write(f"\n=== Log Entry: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        for key, values in skipped_file.items():
            log_file.write(f"{key}:\n")
            for value in values:
                log_file.write(f"  - {value}\n")

    # calculate summary statistics
    n_fitted_already = len(skipped_file['fitted'])
    n_failed = len(skipped_file['failed'])
    n_nolines = len(skipped_file['nolines'])
    n_nospecrange = len(skipped_file['nospecrange'])
    n_nospec = len(skipped_file['nospec'])
    n_skipped_total = sum(len(v) for v in skipped_file.values())

    logger.info(f"Optimization complete! Log saved to: {log_file.name}")
    logger.info(f"Successfully fitted: {n_success}/{i_fit} spectra")
    logger.info(f"Already fitted: {n_fitted_already}")
    logger.info(f"Failed: {n_failed}")
    logger.info(f"Skipped total: {n_skipped_total}")
    logger.info("="*80)

    # print summary to console
    print(f"\n{'='*60}")
    print(f"Optimization complete!")
    print(f"Processed {i_fit} spectra")
    print(f"  Successfully fitted: {n_success}/{i_fit} ({100*n_success/i_fit:.1f}%)" if i_fit > 0 else "  Successfully fitted: 0/0")
    print(f"  Already fitted: {n_fitted_already}")
    print(f"  Failed: {n_failed}")
    if n_nolines > 0:
        print(f"  No lines in range: {n_nolines}")
    if n_nospecrange > 0:
        print(f"  No spec range: {n_nospecrange}")
    if n_nospec > 0:
        print(f"  No spectrum: {n_nospec}")
    print(f"Results saved to: {cfg.fpath_outputs}")
    print(f"Log file: {log_file.name}")
    print(f"{'='*60}")
    print("FITTING DONE")
