import glob, importlib, lmfit, os, pickle

import astropy.constants as ac
import astropy.units as au
import matplotlib.pyplot as plt
import msaexp.spectrum
import numpy as np
import pandas as pd

from astropy.table import Table, vstack
from copy import deepcopy
from datetime import datetime
from grizli.utils import get_line_wavelengths
from lmfit import Parameter, Parameters, CompositeModel, Model
from lmfit.models import GaussianModel, gaussian
from sedpy.smoothing import smoothspec  

from tqdm import tqdm

# AWS paths to spectra
BASE_URL = 'https://s3.amazonaws.com/msaexp-nirspec/extractions/'
PATH_AWS = BASE_URL + '{root}/{file}'
PATH_LOCAL = '/Volumes/shield/data/dja_spectra/{file}'
#PATH_LOCAL = '~/Downloads/dja_spectra/{file}'

GRATINGS_HRES = ['G140H', 'G235H', 'G395H']
GRATINGS_MRES = ['G140M', 'G235M', 'G395M']
GRATINGS_LRES = ['PRISM']
GRATINGS_ALL = GRATINGS_HRES + GRATINGS_MRES + GRATINGS_LRES
GRATINGS_MHRES = GRATINGS_MRES + GRATINGS_HRES

lw = get_line_wavelengths()[0] # spec line list


class SpectralLine():
    """ Spectroscopic lines initialized with the parameters from 
    LMFIT package """
    def __init__(self, prefix, params, sig_integ=7, n_interp=1000,
                 line_model=GaussianModel, bkg_model=None):
        
        # init the model with the best-fit params
        self.name = prefix
        self.params = params
        
        if isinstance(line_model, type):
            self.model = line_model(prefix=self.name)
        elif line_model.prefix == '':
            self.model = line_model(prefix=self.name)
        else:
            self.model = line_model
        self.continuum = bkg_model
        
        # intialize wavelength range for the model
        is_gauss = isinstance(self.model, GaussianModel) |\
                   ('gauss' in self.model.func.__name__.lower())
        is_exp = self.model._name == 'exponential'
        is_gauss_x_exp = 'GaussXExp' in self.model.func.__name__
        if is_gauss:
            center = params[f'{self.name}center']
            sigma = params[f'{self.name}sigma']
        
            xmin = center - sig_integ * sigma
            xmax = center + sig_integ * sigma
            self.x = np.linspace(xmin, xmax, n_interp)
        elif is_exp:
            center = params[f'{self.name}center']
            decay = 10**params[f'{self.name}decay']
            sig_integ = -np.log(1e-11) / decay
            xmin = center - sig_integ
            xmax = center + sig_integ
            self.x = np.linspace(xmin, xmax, n_interp)
        elif is_gauss_x_exp:
            center = params[f'{self.name}center']
            decay = 10**params[f'{self.name}decay']
            sig_integ = -np.log(1e-11) / decay
            xmin = center - sig_integ
            xmax = center + sig_integ
            self.x = np.linspace(xmin, xmax, n_interp)
        else:
            raise ValueError(f"Model {self.model._name} not supported.")
        
    def __call__(self, x):
        return self.model.eval(params=self.params, x=x)
    
    def flux(self):
        from scipy.integrate import simpson
        line = self.model.eval(params=self.params, x=self.x)
        return simpson(line, x=self.x)
    
    def eqw(self):
        if self.continuum is None:
            print(f"No background model provided: cannot compute EW({self.name}).")
        else:
            from scipy.integrate import simpson
            """ 
            eqw = sum( ([line + cont] - cont) / cont * dx) = sum(line/cont*dx)
            """
            
            line = self.model.eval(params=self.params, x=self.x)
            cont = self.continuum.eval(params=self.params, x=self.x)
            eqw = simpson(line / cont, x=self.x) # p-ve for emission lines
            #eqw = simpson(line, x=self.x) / cont_peak # x in obs. frame
        return eqw
        


class SpectrumSample(object):
    def __init__(self, fnames, roots, idxs=None, obj_ids=None, z_input=None, 
                 flux_norm=None, verbose=True, download_data=True):
        
        self.samp, self.is_loaded = self.load_spectra(fnames, roots, 
                                                verbose=verbose, 
                                                download_data=download_data)
        
        if self.is_loaded.any() == False:
            raise ValueError("No spectra were loaded.")
        
        
        if idxs is None:
            self.idxs = np.arange(len(self.samp))
        else:
            self.idxs = np.atleast_1d(idxs)
            self.idxs = self.idxs[self.is_loaded]
        
        if obj_ids is None:
            self.obj_ids = self.idxs.copy()
        else:
            self.obj_ids = np.atleast_1d(obj_ids)
            self.obj_ids = self.obj_ids[self.is_loaded]

        if z_input is None:
            self.z_input = np.array([s.z for s in self.samp])
        else:
            self.z_input = np.atleast_1d(z_input)
            self.z_input = self.z_input[self.is_loaded]
        
        self.fnames = np.atleast_1d(fnames)[self.is_loaded]
        self.froots = np.array([f.split('.spec')[0] for f in self.fnames])
        self.roots = np.atleast_1d(roots)[self.is_loaded]
        
        # unpack spectra data into arrays
        self.flux_norm = flux_norm
        if flux_norm is None:
            self.flux_norm = 1.0
        
        self.x = np.array([s.spec['wave'].value * 1e4 for s in self.samp]) # [A]
        self.y = np.array([s.spec['flux'].value for s in self.samp])
        self.ye = np.array([s.spec['err'].value for s in self.samp])
        self.R = np.array([s.spec['R'].value for s in self.samp])
        self.valid = np.array([s.valid for s in self.samp])
        
        #valid = ~(np.isclose(self.y[0], 0.0) & np.isclose(self.ye[0], 0.0))
        #self.valid = self.valid & valid
        
        # the resolution of g395m was scaled using 
        # one object with both g395h and g395m, where a narrow line 
        # (OIII 4959) is resolved.
        #fpath_params = os.path.join(PATH_OUT, 
        #                'mcmc_result-O3_4959_5007_gauss_narrow_instrum_broadened_optimal-jades-gdn-v3_g395m-f290lp_1181_68797-grp12057.fits')
        #tab = Table.read(fpath_params)
        #scale_disp = tab['scale_disp'].data
        scale_disp = 0.5833351238506942 # g395m
        is_med_res = np.array([[g in f for g in ['g140m', 'g235m', 'g395m']] \
                                for f in self.fnames])
        is_med_res = is_med_res.any(axis=1)
        self.R[is_med_res] = self.R[is_med_res] / scale_disp
        
        # convert uJy to erg/s/cm^2/A
        flam_unit = au.erg / au.cm**2 / au.s / au.AA
        equiv = au.spectral_density(self.x * au.AA)
        self.y = (self.y * au.uJy).to(flam_unit, equivalencies=equiv).value
        self.ye = (self.ye * au.uJy).to(flam_unit, equivalencies=equiv).value
        
        # normalize the flux for better conditioning of the fit
        if np.isclose(self.flux_norm, 1.0):
            p = np.round(abs(np.log10(self.y.max(axis=1))), 0)
            self.flux_norm = 10**p[:, None]
        self.y *= self.flux_norm
        self.ye *= self.flux_norm

    def load_spectra(self, fnames, roots, local_path=None, verbose=True,
                     download_data=True):
        
        if download_data:
            self.urls = [PATH_AWS.format(root=root, file=fname) \
                         for root, fname in zip(roots, fnames)]
        else:
            self.urls = [PATH_LOCAL.format(file=fname) \
                         for fname in fnames]
        
        # load all spectra with msaexp
        samp = []
        is_loaded = np.ones_like(fnames).astype(bool)
        
        if verbose:
            for i, url in tqdm(enumerate(self.urls), total=len(self.urls)):
                try:
                    s = msaexp.spectrum.SpectrumSampler(url, verbose=0)
                    samp.append(s)
                except:
                    is_loaded[i] = False
        else:
            for i, url in enumerate(self.urls):
                try:
                    s = msaexp.spectrum.SpectrumSampler(url)
                    samp.append(s)
                except:
                    is_loaded[i] = False

        return samp, is_loaded
    """
    def get_dispersion(self, obj_id, wavel, z=None):
        #
        Get the instrumental dispersion width (one gaussian sigma) 
        at the requested wavelength.
        
        Inputs
        ------
        obj_id : int
            Object ID
        wavel : float
            Wavelength in angstrom
        z : float
            Redshift
        
        Outputs
        -------
        disp_ang_wavel : array(float)
            The dispersion sigma in angstrom at the requested wavelength, 
            in the rest-frame (i.e. divided by (1 + z)).
        #
        obj_id = np.atleast_1d(obj_id)
        idxs_obj = [np.argwhere(np.isin(self.obj_ids, i)).ravel() \
                    for i in obj_id]
        idxs_obj = np.concatenate(idxs_obj)

        if z is None:
            z = self.z_input[idxs_obj]
            x_rf = self.x[idxs_obj] / (1 + z)
        else:
            assert len(obj_id) == len(z), f"obj_id (length {len(obj_id)}) and z (length {len(z)}) must have same length."
            z = np.atleast_2d(z)
            x_rf = self.x[idxs_obj] / (1 + z)
        if (x_rf > wavel).any():
            iw = np.argwhere(x_rf > wavel)[0]
        elif (x_rf <= wavel).any():
            iw = np.argwhere(x_rf <= wavel)[-1]
        else:
            raise ValueError("No spectral data. Please check the input.")
        disp_sig_ang = x_rf / self.R[idxs_obj] / 2.355
        disp_ang_wavel = disp_sig_ang[iw[0], iw[1]]
        
        # check that the resolution is taken at the right wavelength
        disp_fwhm_kms = disp_sig_ang[iw[0], iw[1]]/x_rf[iw[0], iw[1]]*3e5*2.355
        separation_kms = (x_rf[iw[0], iw[1]] - wavel) /\
                         (0.5 * (x_rf[iw[0], iw[1]] + wavel)) * 3e5 * 2.355
        
        if separation_kms > 3*disp_fwhm_kms:
            # raise a warning
            print(f"Warning: the resolution is reported {separation_kms/disp_fwhm_kms:.1f} instrumental sigmas away from the wanted position ({wavel:.0f} A).")
        
        return disp_ang_wavel
    """
    
    def get_dispersion(self, obj_id, wavel, z=None):
        """ 
        Get the instrumental dispersion width (one gaussian sigma) 
        at the requested wavelength.
        
        Inputs
        ------
        obj_id : int
            Object ID
        wavel : float
            Wavelength in angstrom
        z : float
            Redshift
        
        Outputs
        -------
        disp_ang_wavel : array(float)
            The dispersion sigma in angstrom at the requested wavelength, 
            in the rest-frame (i.e. divided by (1 + z)).
        """
        obj_id = np.atleast_1d(obj_id)
        
        #idxs_obj = [np.argwhere(np.isin(self.obj_ids, i)).ravel() \
        #            for i in obj_id]
        idxs_obj = np.array([list(self.obj_ids).index(i) for i in obj_id])
        
        if z is None:
            z = self.z_input[idxs_obj][:, None]
            x_rf = self.x[idxs_obj] / (1 + z)
        else:
            assert len(obj_id) == len(z), f"obj_id (length {len(obj_id)}) and z (length {len(z)}) must have same length."
            z = np.atleast_2d(z)
            x_rf = self.x[idxs_obj] / (1 + z)
        
        disp_ang = x_rf / self.R[idxs_obj] / 2.355
        
        n_obj = len(obj_id)
        idx_wavel = np.argmin(np.abs(x_rf - wavel), axis=1)
        idx_obj = np.arange(n_obj)
        idxs = (idx_obj, idx_wavel)
        disp_wave_ang = disp_ang[idxs]
        wavels = x_rf[idxs]
        d_wavel = abs(wavels - wavel) / disp_wave_ang
        outlier = d_wavel > 3
        if outlier.any():
            # raise a warning
            print(f"Warning: the resolution is reported > 3 instrumental sigmas away from the wanted position in {outlier.sum()} objects ({wavel:.0f} A).")
        return disp_wave_ang

    def get_spec(self, obj_id=None, obj_idx=None, masked=True):
        
        if obj_id is not None:
            idx = list(self.obj_ids).index(obj_id)
        elif obj_idx is not None:
            idx = obj_idx
        else:
            raise ValueError("Please provide either 'obj_id' or 'obj_idx'.")
        
        if masked:
            if hasattr(self, 'in_window'):
                mask = self.in_window[idx] & self.valid[idx]
            else:
                mask = self.valid[idx]
        else:
            mask = np.ones_like(self.x[idx], dtype=bool)
        
        x = self.x[idx][mask]
        y = self.y[idx][mask]
        ye = self.ye[idx][mask]
        
        return np.c_[x, y, ye]
        

class SpectrumModel(object):
    """ Spectrum model that uses LMFit functionality """
    
    def __init__(self):
        from eazy import igm
        #=== init model params
        # add parameters for a composite model
        self.params = Parameters()
        self.model = None
        
        # dust and igm laws
        from dust_extinction.averages import G03_SMCBar
        self.DUST_LAW = G03_SMCBar()
        self.IGM_OBJECT = igm.Inoue14() # IGM absorption law
        
    def redden(self, x, flux, **kwargs):
        # apply an attenuation law
        # x : [um] rest-frame wavelength
        
        #from eazy.templates import Redden
        #ext = Redden(model='smc', Av=1)
        #_ext = ext(_wave)
        
        # extinction limits
        x1, x2 = self.DUST_LAW.x_range
        x_ext = 1.0 / x
        ext_sel = (x_ext >= x1) & (x_ext < x2)
        
        ext = np.empty_like(x_ext)
        ext[ext_sel] = 10.**(-0.4 * self.DUST_LAW(x_ext[ext_sel] / au.um) * \
                        kwargs['Av'])
        ext[~ext_sel] = 1
        return flux * ext #* (1 + kwargs['z'])**(-1)

    def deredden(self, x, flux, **kwargs):
        # apply an attenuation law
        # x : [um] rest-frame wavelength
        
        #from eazy.templates import Redden
        #ext = Redden(model='smc', Av=1)
        #_ext = ext(_wave)
        
        # extinction limits
        x1, x2 = self.DUST_LAW.x_range
        x_ext = 1.0 / x
        ext_sel = (x_ext >= x1) & (x_ext < x2)
        
        ext = np.empty_like(x_ext)
        ext[ext_sel] = 10.**(-0.4 * self.DUST_LAW(x_ext[ext_sel] / au.um) * \
                        kwargs['Av'])
        ext[~ext_sel] = 1
        return flux / ext #* (1 + kwargs['z'])**(-1)

    def model_continuum(self, x_rest_um, **kwargs):    
        #is_uv = x_rest < 3000
        #y_cont = np.where(is_uv, 
        #  kwargs['cont_uv_amp'] * x_rest_um**kwargs['cont_uv_exp'],
        #  kwargs['cont_op_amp'] * x_rest_um**kwargs['cont_op_exp'])
        y_cont = kwargs['cont_amp'] * x_rest_um**kwargs['cont_exp']
        return y_cont
    
    def dla(self, x, NH):
        """ A Voigt-Hjerting model of Lya continuum absorption by the neutral 
        medium.  Used here to model the DLA feature (Garcia+2006,2007) """
        
        # TODO: sanity check params: T, v (neutral gas)
        
        # assume x is in Angstrom
        if not hasattr(x, 'unit'):
            x = (x * au.AA).to(au.cm)
            
        if not hasattr(NH, 'unit'):
            NH = NH * au.cm**(-2)

        lam = (1215.6701 * au.AA).cgs  # [A] H I resonant wavel
        T = 1e3 * au.K  # kinetic temperature
        kB = ac.k_B.cgs
        mp = ac.m_p.cgs
        me = ac.m_e.cgs
        vc = ac.c.cgs
        e = ac.e.esu
        
        fi =  0.416400  # oscillator strength
        Gi = 6.265e8 * au.s**(-1)  # damping constant
        
        C = 4 * np.sqrt(np.pi**3) * e**2 / (me * vc) * \
            fi / Gi  # const for the ith transition after photon absorption
        b = np.sqrt(2 * kB * T / mp)  # Doppler parameter
        dlam_doppler = b / vc * lam  # Doppler broadening, i.e. Doppler unit
        a = lam**2 * Gi / (4 * np.pi * vc * dlam_doppler) 
        
        x_D = (x - lam) / dlam_doppler # [Doppler] wavelegth difference
        
        H0 = np.exp(-x_D**2)
        H1 = -1.0 / (x_D**2 * np.sqrt(np.pi)) *\
            (H0**2 * (4*x_D**4 + 7*x_D**2 + 4 + 1.5*x_D**(-2)) - 1.5*x_D**(-2) - 1)
        H = H0 + a * H1
        
        tau = C * a * H * NH
        tau[x < lam] = 0
        return np.exp(-tau).value
    
    def eval(self, x_eval=None):
        
        """ Evaluate the sample models at the best-fit parameters """
        
        if len(self.params) == 0:
            raise ValueError("Parameters are missing: fit a model or load parameters")
        
        if x_eval is None:
            x_eval = self.x
        
        return self.model.eval(params=self.params, x=x_eval)
    
    def set_params(self, params=None):
        
        if type(params) != Parameters:
            raise ValueError("Parameters must be of type lmfit.parameter.Parameters")
        
        # fix the old params
        for p in self.params:
            self.params[p].vary = False
        
        self.params.update(params) # update params
    
    #def lmfit(self, func):
    #    ''' LMFit Model decorator for class models '''
    #    
    #    def wrapper(*args, **kwargs):
    #        return Model(func, independent_vars=['x'])#.eval(params=kwargs,x=x)
    #    return wrapper
    
    #@lmfit
    def model_uvcont(self, x, **kwargs):
        
        # shift from rest-frame to the observed frame
        # 'params' --> 'kwargs', as 'params' is reserved for Models in lmfit
        if 'z' in kwargs:
            x_rest = x * (1 + kwargs['z'])**(-1) # [A] rest-frame wavelength
        else:
            x_rest = x
        
        x_rest_um = x_rest * 1e-4
        
        y = self.model_continuum(x_rest_um, **kwargs) # add model continuum
        y = self.redden(x_rest_um, y, **kwargs) # redden the continuum
        
        # apply DLA feature
        #nh = 1e22 # [cm-2] column density
        dla_x = np.ones_like(x_rest)
        is_dla = x_rest < 1500
        dla_x[is_dla] = self.dla(x_rest[is_dla], 10**kwargs['NH'])
        
        # apply IGM absorption
        igmz = np.ones_like(y)
        lyman = x_rest < 1300
        igmz[lyman] = self.IGM_OBJECT.full_IGM(kwargs['z'], x[lyman])
        
        # remove flux below 900
        y[x_rest < 1000] = 0
        
        return y * igmz * dla_x
    
    def model_opcont(self, x, **kwargs):
        
        # shift from rest-frame to the observed frame
        # 'params' --> 'kwargs', as 'params' is reserved for Models in lmfit
        if 'z' in kwargs:
            x_rest = x * (1 + kwargs['z'])**(-1) # [A] rest-frame wavelength
        else:
            x_rest = x
        
        x_rest_um = x_rest * 1e-4
        
        y = self.model_continuum(x_rest_um, **kwargs) # add model continuum
        #y = self.redden(x_rest_um, y, **kwargs) # redden the continuum
        
        return y
    
    def model_composite(self, x, **kwargs):
        # shift from rest-frame to the observed frame
        # 'params' --> 'p', as 'params' is reserved for Models in lmfit
        redshift = 0.0
        if 'z' in kwargs:
            redshift = kwargs['z']
            x_rest = x * (1 + redshift)**(-1) # [A] rest-frame wavelength
        else:
            x_rest = x * (1 + redshift)**(-1)
        
        # model lines
        params = Parameters()
        for k, v in kwargs.items():
            params[k] = Parameter(name=k, value=v)
        
        y = self.model_components.eval(params=params, x=x_rest)
        
        #x_rest_um = x_rest * 1e-4
        #return self.redden(x_rest_um, y, **kwargs) # redden the continuum
        return y
    
    def set_model_components(self, model_components=None):
        err = "Please provide lmfit.Model(model_components) or lmfit.model.CompositeModel(model_components)."
        success = (not model_components is None) |\
                  (type(model_components) == Model) |\
                  (type(model_components) == CompositeModel)
        
        if not success:
            raise ValueError(err)
        self.model_components = model_components
    
    def set_model_uvcont(self):
        self.model = Model(self.model_uvcont, independent_vars=['x'])
        
    def set_model_opcont(self):
        self.model = Model(self.model_opcont, independent_vars=['x'])
    
    def set_model(self):
        self.model = Model(self.model_composite, independent_vars=['x'])
    
    def add_smoothing(self, fftsmooth=True):
    #def add_smoothing(self, fftsmooth=True, scale_disp=1.):
        
        if self.model is None:
            raise ValueError("Please set up the model first.")
        
        self.model_non_smooth = deepcopy(self.model)
        
        def smooth_model(x, **kwargs):
            # the spectrum is smoothed with the wavel-dependent dispersion (LSF)
            # R=l/dl * 2.355
            # official prism R is lower than in practice (* 0.8)

            params = Parameters()
            for k, v in kwargs.items():
                params[k] = Parameter(name=k, value=v)
            
            # Line Spread Function
            # x: [A] rest-frame    
            flux_mod = self.model_non_smooth.eval(x=x, params=params)
            wave_ang = self.x[0] * 1.
            disp_ang = self.x[0] / self.R[0] / 2.355 # dispersion [A/pix]
            
            if 'scale_disp' not in kwargs:
                kwargs['scale_disp'] = 1.
            disp = np.interp(x, wave_ang, disp_ang, 
                             left=disp_ang[0], right=disp_ang[-1]) \
                   * kwargs['scale_disp']
            
            if np.allclose(disp, 0.):
                flux_smooth = flux_mod * 1.
            else:
                x = x.squeeze()
                flux_mod = flux_mod.squeeze()
                disp = disp.squeeze()
                flux_smooth = smoothspec(x, flux_mod, outwave=x,
                                         resolution=disp, smoothtype='lsf',
                                         fftsmooth=fftsmooth)
            return flux_smooth
        
        if self.model.func.__name__ == 'smooth_model':
            print("Model smoothing is already active.")
        else:
            self.model = Model(smooth_model, independent_vars=['x'])
    
class SpectrumSampleFit(SpectrumSample,SpectrumModel):
    
    def __init__(self, fnames, roots, idxs=None, obj_ids=None, 
                 fit_window_bins=None, z_input=None, flux_norm=1, message=None, 
                 verbose=True, download_data=True, **kwargs):
        
        """
        Set up the fit window of the spectrum
        
        fit_window : list
            Wwavelength window for spectra in the rest frame.  For example, 
            fit_window = [[wav1, wav2], [wav3, wav4], ...].
        """
        
        self.params_sample = []
        self.output = {}
        self.logs = ['']
        
        if z_input is None:
            self.z_input = np.zeros(len(fnames))
        else:
            self.z_input = z_input
        
        # init base class and derived class attributes
        #super().__init__(fnames, roots, idxs=idxs, z_input=z_input, 
        #                 flux_norm=flux_norm)
        SpectrumSample.__init__(self, fnames, roots, idxs=idxs, obj_ids=obj_ids,
                                z_input=z_input, flux_norm=flux_norm, 
                                verbose=verbose, download_data=download_data)
        SpectrumModel.__init__(self)
        self.n_obj = self.x.shape[0]
        
        self.fit_window = np.full((self.n_obj, 1, 2), np.nan)
        self.in_window = np.ones_like(self.x, dtype=bool)
        self.update_fit_window(fit_window_bins)
        
        # TODO: add an attribute: mask with objects with x (in)outside of 
        # fit_window
        
        if message is None:
            message = f"Init sample with {self.n_obj} spectra"
        self.update_log(message=message)
    
    def update_fit_window(self, fit_window_bins=None):
        
        # check keyword arguments
        if fit_window_bins is None:
            #fit_window_bins = [[[self.x[i].min(), self.x[i].max()]] \
            #                   for i in range(len(self.obj_ids))]
            x_rf = self.x / (1 + self.z_input[:, None])
            fit_window_bins = [[x_rf.min(), x_rf.max()]]
        
        # get a fit wavelength mask
        fit_window_bins = np.array(fit_window_bins).T
        _nbound, nwin = fit_window_bins.shape[0], fit_window_bins.shape[1]
        z_tile = np.tile(self.z_input[:, None, None], 
                         (1, *fit_window_bins.shape))
        self.fit_window = fit_window_bins * (1 + z_tile)
        
        above = np.tile(self.x[:, None, :],(1, nwin, 1)) > \
                self.fit_window[:, 0][:, :, None]
        below = np.tile(self.x[:, None, :],(1, nwin, 1)) < \
                self.fit_window[:, 1][:, :, None]
        self.in_window = (above & below).sum(axis=1).astype(bool) # fit mask

    def fit(self, fit_redshift=True, idxs=None, message=None, 
            disable_tqdm=True):
        """ Fit spectra using the current model using
        least_squares and store the best fit and model properties """
        
        # create output dictionary with fit params and props
        param_keys = list(self.params.keys())
        other_keys = ['idx', 'status', 'chi2red']
        all_keys = other_keys + param_keys
        
        for k in all_keys:
            self.output[k] = np.full((self.n_obj, 1), np.nan, dtype=float)
        
        for k in param_keys:
            self.output[k+'_err'] = np.full((self.n_obj, 1), np.nan, 
                                            dtype=float)
            
        if idxs is None:
            idxs = np.arange(len(self.idxs))

        # loop over all spectra
        self.params_sample.clear()
        
        for i, idx in tqdm(enumerate(idxs), total=len(idxs), 
                           disable=disable_tqdm):
            
            # get current iteration spectrum
            in_xrange = self.in_window[idx] & self.valid[idx]
            x_fit = self.x[idx][in_xrange]
            y_fit = self.y[idx][in_xrange]
            ye_fit = self.ye[idx][in_xrange]
            
            # fix the input redshifts
            # TODO: only set from outside or make optional
            # as e.g., this is not needed for empirical continuum fits
            if fit_redshift:
                self.params['z'].set(value=self.z_input[idx], 
                                     min=0.85*self.z_input[idx]-1e-6, 
                                     max=1.15*self.z_input[idx]+1e-6, 
                                     vary=True)

            result = self.model.fit(x=x_fit, data=y_fit, 
                                    params=self.params.copy(), 
                                    weights=1/ye_fit,
                                    method='least_squares', 
                                    nan_policy='propagate')
            self.params_sample.append(result.params.copy())
            
            y_mod = self.model.eval(params=result.params, x=x_fit)
            
            #plt.step(x_fit / (1 + self.z_input[idx]), y_fit)
            #plt.step(x_fit / (1 + self.z_input[idx]), y_mod)
            #plt.show()
            
            # compute chi2 with the +ve residuals only
            y_diff = (y_fit - y_mod) / ye_fit
            
            # get line fluxes and EWs
            #lines = [SpectralLine(name, result.params) for name in line_names]
            #flux_i, eqw_i = np.array([(line.flux, line.eqw) for line in lines]).T
            
            # store line properties
            for k in param_keys:
                self.output[k][i] = result.params[k].value
                self.output[k+'_err'][i] = result.params[k].stderr
            
            # check and record fit status -- success or failed 
            # this is an empirical check (works for least_squares, not for mcmc)
            if type(result.covar) is type(None):
                status = 0 # failed
            else:
                status = 1 # success
            
            self.output['idx'][i] = self.idxs[idx]
            self.output['status'][i] = status
            self.output['chi2red'][i] = result.redchi
            
            if message is None:
                message = f"Fitted the model {type(self.model)}"
            self.update_log(message=message)
            
        return self.tabulate_output()

    def subtract_model(self, model=None, message=None):
        
        if (model is None) & (self.model is None):
            raise ValueError("'model' is None and 'self.Model' is None. Please provide a model.")
        elif model is None:
            model = self.model
        
        #if len(self.output) == 0:
        if len(self.params_sample) == 0:
            raise ValueError("Best-fit parameters are missing: no model has been fitted.")
        
        y_sub = np.empty_like(self.y)
        for i, params in enumerate(self.params_sample):
            y_model = model.eval(params=params, x=self.x[i])
            y_sub[i] = self.y[i] - y_model
        
        if message is None:
            message = f"Subtracted the model {type(model)}"
        self.update_log(message=message)
        
        return y_sub
    
    def eval_sample(self, x_eval=None):
        
        """ Evaluate the sample models at the best-fit parameters """
        
        if len(self.params_sample) == 0:
            raise ValueError("Best-fit parameters are missing: no model has been fitted.")
        
        if x_eval is None:
            x_eval = self.x
        
        y_model = np.empty(x_eval.shape)
        for i, params in enumerate(self.params_sample):
            y_model[i] = self.model.eval(params=params, x=x_eval[i])
        return y_model
    
    def deredden_sample(self):
        x_rest_um = self.x * 1e-4 / (1 + self.z_input[:, None])
        return self.deredden(x_rest_um, self.y, **self.params)
        
    def set_flux(self, flux):
        self.y = flux
        
    def update_log(self, message=''):
        
        stamp = datetime.today().strftime('%Y-%m-%d-%H:%M:%S')
        
        self.logs += [stamp + '\t' + message]
    
    def log(self):
        out = '\n'.join(self.logs)
        return out[2:]
    
    def tabulate_output(self, data=None, columns=None, dtypes=None):
        
        all_columns = list(self.output.keys()) # column names
        if not (columns is None):
            columns = list(np.atleast_1d(columns))
            all_columns.extend(columns)
            
        data_types = [np.int32]*2 + [float]*(len(all_columns)-2) # data types
        if not (dtypes is None):
            dtypes = list(np.atleast_1d(dtypes))
            data_types.extend(dtypes)

        values = [] # column values
        for k in self.output.keys():
            values.append(self.output[k])
        values = np.hstack(values)
        if not (data is None):
            values = np.concatenate([values, data])

        tab = Table(data=values, names=all_columns, dtype=data_types)
        return tab
    
    def save_output(self, fpath='.', fname='table.fits', message=None):
        
        tab = self.tabulate_output()
        PATH_OUT = os.path.join(fpath, fname)
        tab.write(PATH_OUT, overwrite=True)
        
        if message is None:
            message = f"Saved the output to: {PATH_OUT}"
        self.update_log(message=message)
    
    def dump_sample_params(self, fpaths=None, message=None):
        """ Pickle LMFIT parameters """
        import pickle
        
        if fpaths is None:
            fpaths = [f'param-{i}.pkl' for i in range(self.n_obj)]
        
        for i, p in enumerate(self.params_sample):
            with open(fpaths[i], 'wb') as f:
                pickle.dump(p.dumps(), f)
        
            if message is None:
                message = f"Saved the output to: {fpaths[i]}"
            self.update_log(message=message)

    #def load_params(self, fpath=None, message=None):
    
    def plot_fit(self, tab_params, obj_id=None, obj_idx=None):
        
        if (obj_id is None) & (obj_idx is None):
            raise ValueError("Please provide either object ID or Index.")
        elif obj_idx is None:
            obj_idx = list(self.obj_ids).index(obj_id)
        else:
            obj_id = self.obj_ids[obj_idx]

        # load the fit window edges
        print('obj_id:', obj_id)
        print('idx:', obj_idx)
        fit_edges = self.fit_window[obj_idx].reshape(-1, 2)
        
        # load a model from file
        components_all = self.model_components.components
        components = [c for c in components_all if not ('cont' in c._name)]
        components = [c for c in components if not ('cygni' in c.func.__name__)]

        # here, we want to plot the Gaussian when we hit 
        # the convolution model params, so sub in the GaussModel
        idx_GaussXExp = [i for i, c in enumerate(components) \
                            if c.func.__name__ == 'GaussXExp']
        if len(idx_GaussXExp) > 0:
            components[idx_GaussXExp[0]] = GaussianModel(prefix='Ha_6565b_')

        # load best-fit line params
        params_best = self.params.copy()
        mask = self.in_window[obj_idx] & self.valid[obj_idx]
        x = np.ma.masked_where(~mask, self.x[obj_idx])
        y = np.ma.masked_where(~mask, self.y[obj_idx])
        ye = np.ma.masked_where(~mask, self.ye[obj_idx])
        zbest = params_best['z'].value
        x_rf = x / (1 + zbest)
        
        # evaluate model at an observed x-axis
        x_eval = x.compressed()#.reshape(1, -1)
        ymod_obsrange = np.full_like(y, fill_value=np.nan)
        ymod_obsrange[mask] = self.eval(x_eval=x_eval)
        ymod_obsrange = np.ma.masked_where(~mask, ymod_obsrange)
        
        # evaluate model at a finer x-axis
        x_eval = np.linspace(x.min(), x.max(), 1000)
        x_eval_rf = x_eval / (1 + zbest)
        ymod = self.eval(x_eval=x_eval)

        # subtract best-fit continuum
        model_cont = [comp for comp in components_all \
                      if 'cont' in comp._name][obj_idx]
        y_cont_obsx = model_cont.eval(params=params_best, x=x / (1 + zbest))
        ymod_obsrange -= y_cont_obsx # remove continuum
        y -= y_cont_obsx
        
        y_cont = model_cont.eval(params=params_best, x=x_eval_rf)
        ymod -= y_cont
        
        # evaluate model components
        y_comps = [components[i].eval(params=params_best, x=x_eval_rf) \
                   for i in range(len(components))]
        
        diff_chi = (y - ymod_obsrange) / ye
        n_data = tab_params['ndata'].item()
        n_varys = tab_params['nvarys'].item()
        n_dof = n_data - n_varys
        chisq = np.sum(diff_chi**2)
        redchisq = chisq / n_dof
        
        
        #=== make a figure
        data = [[x_rf, y, ye],]
        model = [[x_eval_rf, ymod],
                 *[[x_eval_rf, y_comps[i]] for i in range(len(y_comps))],]
        fig, (ax, axr) = self.plot_data_arrays(data=data, model=model)
        axr.errorbar(x_rf, diff_chi, yerr=1, ls='none', c='gray', 
                     elinewidth=0.5, marker='.', ms=1.5)
        
        text = f"ID {obj_id}"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=8,
                va='top', ha='left')
        
        text = rf"$\chi^2 = {chisq:.2f}$" + '\n'
        text += rf"$\chi^2_{{\nu}} = {redchisq:.2f}$" + '\n'
        text += rf"ndata = {n_data}" + '\n'
        text += rf"npar = {n_varys}"
        ax.text(0.72, 0.95, text, transform=ax.transAxes, fontsize=8,
                va='top', ha='left')
        
        return fig, (ax, axr)
        
    def plot_data_arrays(self, data, model,
                         subplot_kw=None, gridspec_kw=None, fig_kw=None):
        
        # set up the keyword arguments
        _fig_kw = {'figsize': (4, 4), 'dpi': 150,}
        if not (fig_kw is None): # update figure kws with any user inputs
            _fig_kw = _fig_kw | fig_kw
        
        _subplot_kw = {}
        if not (subplot_kw is None):
            _subplot_kw = _subplot_kw | subplot_kw
        
        _gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.0}
        if not (gridspec_kw is None):
            _gridspec_kw = _gridspec_kw | gridspec_kw
        
        # create a figure
        fig, (ax, axr) = plt.subplots(2, 1, sharex=True,
                            subplot_kw=_subplot_kw,
                            gridspec_kw=_gridspec_kw,
                            **_fig_kw)
        
        axr.set_ylim(-10, 10)
        axr.set_yticks([-5, 0, 5])
        axr.set_yticks(np.arange(-5, 6, 1), minor=True)
        axr.grid(True, axis='y', which='both', ls=':')
        axr.set_xlabel(r'Wavelength [${\rm \AA}$]')
        axr.set_ylabel(r'$\chi$')
        ax.set_ylabel(r'$f_{\lambda}$ [erg s$^{-1}$ cm$^{-2}~{\rm \AA^{-1}}$]')
        
        axr.axhline(0.0, ls='-', c='gray', lw=0.4)
        
        for i, (x, y, ye) in enumerate(data):
            ax.step(x, y, lw=0.5, where='mid', color=f'C{i}')
            ax.fill_between(x, y - ye, y + ye, alpha=0.2, 
                            step='mid', color=f'C{i}')
        
        for i, (x, y) in enumerate(model):
            if i == 0:
                ax.plot(x, y, lw=0.5, color=f'C{i+1}', ls='-')
            else:
                ax.plot(x, y, lw=0.5, color=f'C{i+1}', ls='--')
        
        ymin = data[0][1].min() - 0.05 * (data[0][1].max() - data[0][1].min())
        ymax = data[0][1].max() + 0.1 * (data[0][1].max() - data[0][1].min())
        ax.set_ylim(ymin, ymax)
            
        #plt.show()
        return fig, (ax, axr)
        
        

def logLik(p, x, y, ye):
    """ log likelihood function for MCMC """
    y_pred = sampleFit.model.eval(params=p, x=x)
    #y_pred = model.eval(params=p, x=x)
    
    #return -0.5 * np.sum( ((y - y_pred) / ye)**2 )
    print("here is main", os.getpid())
    return np.abs( ((y - y_pred) / ye) )
    #return (y - y_pred) / ye

def load_tables(use_local_spectra=False, save_tables=False, verbose=True,
                version_DJA='2025_01_13'):
    """
    Load tables from AWS database or local disk
    
    Parameters
    ----------
    use_local_spectra : bool, optional
        If True, load tables from local disk. If False, load from AWS database.
        Default is False.
    save_tables : bool, optional
        If True, save tables to disk. Default is False.
    verbose : bool, optional
        If True, print out the length of the tables. Default is True.
    """
    # catalog and data paths
    fpath_data = os.path.join(os.getenv('data'), 'spectra/data')
    fpath_cat = os.path.join(fpath_data, 'catalogs')
    #fpath_spec = os.path.join(fpath_data, 'dja_spectra')
    
    fdir_path = ''
    #version_DJA = '2024_09_27'
    #version_DJA = '2025_01_09'
    #version_DJA = '2025_01_13'
    #version_DJA = '2025_03_01'
    fdir_path = f'nirspec_tables-{version_DJA}'
    fpath_redshifts = os.path.join(fpath_cat, fdir_path,
                                   f'nirspec_redshifts-{version_DJA}.fits')
    fpath_extractions = os.path.join(
        fpath_cat, fdir_path,
        f'nirspec_extractions-{version_DJA}-groupid_final.fits')
    #fpath_extractions = os.path.join(
    #    fpath_cat, fdir_path,
    #    f'nirspec_extractions-{version_DJA}-groupid_final-latest_ver.fits')
    
    
    if use_local_spectra:
        
        # load catalogs
        df_msaexp = Table.read(fpath_redshifts).to_pandas()
        df_ext = Table.read(fpath_extractions).to_pandas()
        
        # decode any dtype=bytes columns
        for col in df_msaexp.columns:
            isbytes = isinstance(df_msaexp[col].iloc[0], bytes)
            if isbytes:
                mask_notna = df_msaexp[col].notna()
                f = lambda x: x.decode('ascii')
                df_msaexp.loc[mask_notna, col] = df_msaexp.loc[mask_notna, 
                                                         col].apply(f)
                df_msaexp[col] = df_msaexp[col].astype('string')
        
        for col in df_ext.columns:
            isbytes = isinstance(df_ext[col].iloc[0], bytes)
            if isbytes:
                mask_notna = df_ext[col].notna()
                f = lambda x: x.decode('ascii')
                df_ext.loc[mask_notna, col] = df_ext.loc[mask_notna, 
                                                         col].apply(f)
                df_ext[col] = df_ext[col].astype('string')

        # the table is larger than the msaexp fits table
        df_ext_fit = pd.merge(df_msaexp['file'], df_ext, how='left', on='file')
        has_fit = df_ext['file'].isin(df_ext_fit['file'])
        df_ext_nofit = df_ext[~has_fit]
        if verbose:
            print(f"len nirspec_redshifts table: {len(df_ext_fit)}")
            print(f"len nirspec_extractions table: {len(df_ext)}")
        
    else:
        
        from grizli.aws import db

        # load msaexp catalog
        # WHERE ne.root LIKE 'gds%%'
        df_msaexp = db.SQL("""
                        SELECT *
                        FROM nirspec_redshifts ne
                        """).to_pandas()

        df_ext = db.SQL("""
                        SELECT *
                        FROM nirspec_extractions ne
                        """).to_pandas()
        # save tables to disk
        if save_tables:
            fpath_redshifts = os.path.join(fpath_cat, 
                                'nirspec_redshifts.fits')
            fpath_extractions = os.path.join(fpath_cat, 
                                'nirspec_extractions.fits')
            Table.from_pandas(df_ext).write(fpath_extractions, overwrite=True)
            Table.from_pandas(df_msaexp).write(fpath_redshifts, overwrite=True)
            
            print("Tables saved to: ")
            print(f"\t {fpath_redshifts}")
            print(f"\t {fpath_extractions}")

        if verbose:
            print(f"len nirspec_redshifts table: {len(df_msaexp)}")
            print(f"len nirspec_extractions table: {len(df_ext)}\n")

    return df_ext, df_msaexp


#=== helper functions ================================
def get_mcmc_estimates(labels, samples):
    """ Compute and return median +/- one sigma uncertainties from
    MCMC samples"""
    # print solutions
    ndim = len(labels)
    results_dict = {}
    theta_best = np.empty(shape=(ndim, 1))
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        results_dict[labels[i]] = np.array([mcmc[1], q[0], q[1]])
        theta_best[i] = mcmc[1]
    return results_dict, theta_best

def print_mcmc_estimates(labels, samples):
    """ Compute and print median +/- one sigma uncertainties from
    MCMC samples"""
    # print solutions
    ndim = len(labels)
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

def group_outputs(fnames_dja=None, src_ids=None, line_label=''):
    """ Load model fitting results for requested objects """
    fpaths = np.array(glob.glob(os.path.join(PATH_OUT, 
                                             f'mcmc_result-{line_label}*.fits')))
    fnames = np.array([f.split('/')[-1] for f in fpaths])
    froots = np.array([f.split('.fits')[0] for f in fnames])
    groupids = np.array([int(f.split('-')[-1].replace('grp', '')) \
                         for f in froots])
    model_labels = np.array([f.split('-')[1] for f in froots])
    fnames_spec = np.array(['-'.join(f.split('-')[2:-1]) + '.spec.fits' \
                            for f in froots])

    _df_fnames = pd.DataFrame(fnames_spec.reshape(-1, 1), columns=['file'])
    df_ext, _ = load_tables(use_local_spectra=True, verbose=False)
    roots = np.array(pd.merge(df_ext, _df_fnames, 
                              on='file', how='right')['root'])
    
    if (src_ids is None) & (fnames_dja is None):
        mask = np.ones_like(groupids, dtype=bool)
    elif fnames_dja is not None:
        mask = np.isin(fnames_spec, fnames_dja)
    elif src_ids is not None:
        mask = np.isin(groupids, src_ids)
    
    isna = pd.isna(roots)
    fill_arr = np.full((isna.sum(),), '')
    roots[isna] = fill_arr
    df_mcmc_outputs = pd.DataFrame(data={'GroupID': groupids[mask],
                                         'model_label': model_labels[mask],
                                         'fpath_result': fpaths[mask], 
                                         'file': fnames_spec[mask],
                                         'root': roots[mask],
                                         })
    if fnames_dja is not None:
        df_mcmc_outputs = pd.merge(df_mcmc_outputs, 
                                   pd.DataFrame(fnames_dja, columns=['file']),
                                   how='right', on='file')
    
    groups = {}
    for groupid, group in df_mcmc_outputs.groupby('GroupID'):
        
        labels = np.array(group['model_label'])
        fpaths_result = np.array(group['fpath_result'])
        files = np.array(group['file'])
        roots = np.array(group['root'])
                
        fpaths_pars = [f.replace('mcmc_result', 'mcmc_params') \
                    for f in fpaths_result]
        fpaths_pars = [f.replace('.fits', '.pckl') for f in fpaths_pars]
        fpaths_chain = [f.replace('mcmc_params', 'mcmc_chain') \
                        for f in fpaths_pars]
        
        fpath_model_module = [f"py_model_{label}" for label in labels]
        
        tabs = [Table.read(f) for f in fpaths_result]
        [tab.add_columns([labels[i], files[i], roots[i], fpaths_result[i], 
                          fpaths_pars[i], fpaths_chain[i],
                          fpath_model_module[i]],
                        names=['model_label', 'file', 'root', 
                               'fpath_mcmc_result',
                               'fpath_line_params', 
                               'fpath_chain', 'fpath_model'], 
                        indexes=[0,0,0,0,0,0,0]) for i, tab in enumerate(tabs)]
        
        tab = vstack(tabs)
        tab.sort('model_label')
        groups[groupid] = tab
    return groups

def group_outputs_filtered(outputs, line_label, model_labels,
                           exact_match=True, exclude_labels=[], 
                           exclude_fnames=[], return_best=False,
                           grating_priority=None):
    #=== prepare dict with props of requested sources
    src_outputs_filt = {}
    for srcid, tab in outputs.items():
        
        is_theline = np.array([line_label in l \
                               for l in tab['model_label']])
        if exact_match:
            is_thecomponent = np.array([[lmod == l for lmod in model_labels] \
                                        for l in tab['model_label']])
            not_excluded = np.array([[lexcl not in l \
                                    for lexcl in exclude_labels] \
                                    for l in tab['fpath_mcmc_result']])
            is_thecomponent = is_thecomponent.any(axis=1) &\
                              not_excluded.all(axis=1)
            is_themodel = is_theline & is_thecomponent
        else:
            is_thecomponent = np.array([[lmod in l for lmod in model_labels] \
                                        for l in tab['model_label']])
            not_excluded = np.array([[lexcl not in l \
                                    for lexcl in exclude_labels] \
                                    for l in tab['fpath_mcmc_result']])
            if len(exclude_fnames) > 0:
                not_excluded_fname = np.array([[lexcl not in l \
                                        for lexcl in exclude_fnames] \
                                        for l in tab['fpath_mcmc_result']])
                not_excluded *= not_excluded_fname.all(axis=1)[:, None]
            is_thecomponent = is_thecomponent.any(axis=1) &\
                              not_excluded.all(axis=1)
            is_themodel = is_theline & is_thecomponent    
        
        if is_themodel.sum() == 0:
            continue
        
        # check if models were fitted using multiple gratings
        # if so, keep only the solutions 
        # with the highest res grating or with the lowest chi.sq.
        gratings_hres = ['g140h', 'g235h', 'g395h']
        files = tab['file'][is_themodel].data
        labels = list(tab['model_label'][is_themodel].data)
        duplicates = np.array([labels.count(l) > 1 for l in labels])
        idxs_unique = np.argwhere(~duplicates).ravel()
        idxs_duplicates = np.argwhere(duplicates).ravel()
        if duplicates.any():
            is_hres = [[grat.lower() in files[idx] for grat in GRATINGS_HRES] \
                                                   for idx in idxs_duplicates]
            is_hres = np.array(is_hres).any(axis=1)
            idxs_hres = np.argwhere(is_hres).ravel()
            
            grats = [[grat.lower() for grat in GRATINGS_MRES \
                                        if grat.lower() in files[idx]] \
                                   for idx in idxs_duplicates]
            grats = [[*g] for g in grats]
            if len(idxs_hres) > 0:
                # keep the high res grating
                idxs = np.concatenate([idxs_unique, idxs_duplicates[idxs_hres]])
            elif (len(grats) > 1):
                if grating_priority is not None:
                    # keep a prioritized grating
                    grat = [grat for pr in grating_priority \
                                 for grat in grats \
                                        if pr.lower() == grat][0]
                    idx = list(grats).index(grat)
                    idxs = np.concatenate([idxs_unique, 
                                           idxs_duplicates[idx:idx+1]])
                else:
                    # keep all gratings
                    idxs = np.concatenate([idxs_unique, idxs_duplicates])
            else:
                # keep the lowest reduced chisqr fit
                redchis = tab[is_themodel][duplicates]['redchi'].data
                is_best = [redchis.argmin()]
                idxs = np.concatenate([idxs_unique, idxs_duplicates[is_best]])
                
        else:
            idxs = idxs_unique
        
        idxs_models = np.argwhere(is_themodel).ravel()[idxs]
        src_outputs_filt[srcid] = tab[idxs_models]
        src_outputs_filt[srcid].sort('redchi')
        
        gratings = [f.split('-')[-2].split('_')[-1] \
                    for f in src_outputs_filt[srcid]['file']]
        labels = src_outputs_filt[srcid]['model_label']
        new_labels = [labels[i] + '_' + gratings[i] for i in range(len(labels))]
        src_outputs_filt[srcid]['model_label'] = new_labels
        src_outputs_filt[srcid].add_column(gratings, name='grating')
        
        if return_best:
            for groupid, tab in src_outputs_filt.items():
                src_outputs_filt[groupid] = tab[:1]
                
    return src_outputs_filt

def tabulate_outputs(outputs):
    tab = Table()
    groupids = []
    for groupid, _tab in outputs.items():
        if not isinstance(_tab, Table):
            _tab = _tab.table
        if len(_tab) > 0:
            tab = vstack([tab, _tab])
            groupids.extend([groupid] * len(_tab))
    tab.add_column(col=groupids, index=0, name='GroupID')
    return tab

def get_property_dict(tables_dict, srcid, property):
    if srcid in tables_dict:
        model_labels = tables_dict[srcid]['model_label']
        if property in tables_dict[srcid].columns:
            values = tables_dict[srcid][property]
            
            # check which grating was used to fit a model
            files = tables_dict[srcid]['file']
            is_g140m = ['g140m' in f for f in files]
            is_g140h = ['g140h' in f for f in files]
            is_g235m = ['g235m' in f for f in files]
            is_g235h = ['g235h' in f for f in files]
            is_g395m = ['g395m' in f for f in files]
            is_g395h = ['g395h' in f for f in files]
            gratings = np.zeros_like(files)
            gratings[is_g140m] = 'g140m'
            gratings[is_g140h] = 'g140h'
            gratings[is_g235m] = 'g235m'
            gratings[is_g235h] = 'g235h'
            gratings[is_g395m] = 'g395m'
            gratings[is_g395h] = 'g395h'
            model_labels = np.array([l + f'_{gratings[i]}' \
                                    for i, l in enumerate(model_labels)])
            
            idxs_sorted = np.argsort(model_labels)
            prop_dict = dict(zip(model_labels[idxs_sorted], values[idxs_sorted]))
        else:
            prop_dict = {}
    else:
        prop_dict = {}
    return prop_dict

def tabulate_emcee_results(params, lnprob, chain,
                           sampleFit, mcmc_dict, stats_dict, obj_idx=None):
    # === tabulating best-fit parameters
    params_variable = params.copy()
    [params_variable.pop(p) for p in params \
        if (not params[p].vary)]

    columns, values = np.array([]), np.array([])

    # get median values
    values_params_median = [params_variable[p].value for p in params_variable]
    stderr_params = [params_variable[p].stderr for p in params_variable]
    values = np.append(values, values_params_median)
    if 'z' not in params_variable:
        values = np.append(values, params['z'].value) # add z
    values = np.append(values, stderr_params)
    if 'z' not in params_variable:
        values = np.append(values, 0.0) # add z_err

    columns_params = list(params_variable.keys())
    columns_stderr = [c + '_sig' for c in columns_params]
    columns = np.append(columns, columns_params)
    if 'z' not in params_variable:
        columns = np.append(columns, 'z') # add z
    columns = np.append(columns, columns_stderr)
    if 'z' not in params_variable:
        columns = np.append(columns, 'z_sig') # add z_err

    # get MLE values
    highest_prob = np.argmax(lnprob)
    hp_loc = np.unravel_index(highest_prob, lnprob.shape)
    values_params_mle = chain[hp_loc]
    values = np.append(values, values_params_mle)
    
    columns_params_mle = [c + '_mle' for c in columns_params]
    columns = np.append(columns, columns_params_mle)
    
    # get assymetric uncertainties
    quantiles_num = [15.865, 84.135]
    n_qntl = len(quantiles_num)
    n_var = len(columns_params)
    params_names = list(params_variable.keys())
    flatchain = pd.DataFrame(data=chain.reshape(-1, len(params_names)), 
                             columns=params_names)
    quantiles = np.array([np.percentile(flatchain[p], quantiles_num) \
                        for p in params_variable])
    values_stderr_assym = np.array([values_params_median[i] - q \
                            for i, q in enumerate(quantiles)])
    values_stderr_assym = abs(values_stderr_assym).flatten()
    values = np.append(values, values_stderr_assym)

    columns_stderr_assym = [columns_params[i] + '_siglo' \
                            for i in range(len(params_variable))]
    columns_stderr_assym += [columns_params[i] + '_sigup' \
                            for i in range(len(params_variable))]
    columns_stderr_assym = np.array(columns_stderr_assym).reshape(n_qntl, 
                                                                  n_var).T
    columns_stderr_assym = columns_stderr_assym.flatten()
    columns = np.append(columns, columns_stderr_assym)
    
    # === tabulate line fluxes and EWs
    components = sampleFit.model_components.components
    line_names_gauss = [c.prefix for c in components \
                        if 'gauss' in c._name.lower()]
    line_names_exp = [c.prefix for c in components \
                      if 'exponential' in c._name.lower()]
    component_names = [c._name for c in components]
    is_gauss = np.array(['gauss' in c.lower() for c in component_names])
    is_exp = np.isin(component_names, 'exponential')
    modelsGauss = np.atleast_1d(components)[is_gauss]
    modelsExp = np.atleast_1d(components)[is_exp]
    line_names = line_names_gauss + line_names_exp
    columns_flux = [f'{name}flux' for name in line_names]
    columns_eqw = [f'{name}eqw' for name in line_names]
    columns_flux_eqw = columns_flux + columns_eqw
    columns = np.append(columns, columns_flux_eqw)
    
    # if don't want the convolved model, use the intrinsic gaussian model only
    #idx_gaussXExp = [i for i, c in enumerate(modelsGauss) \
    #                 if c.func.__name__.lower() == 'gaussxexp']
    #func_gauss = [m.func for m in modelsGauss \
    #              if m.func.__name__.lower() == 'gaussianrestframe']
    #if len(func_gauss) == 0:
    #    func_gauss = gaussian
    #else:
    #    func_gauss = func_gauss[0]
    #if len(idx_gaussXExp) > 0:
    #    components[idx_gaussXExp[0]] = Model(prefix='Ha_6565b_', 
    #                                         func=func_gauss)
    
    bkg_mod = [mod for mod in sampleFit.model_components.components \
               if 'cont' in mod._name][0]
    lines_gauss = [SpectralLine(prefix=name, params=params, 
                                line_model=modelsGauss[i],
                                bkg_model=bkg_mod) \
                                    for i, name in enumerate(line_names_gauss)]
    lines_exp = [SpectralLine(prefix=name, params=params, 
                              line_model=modelsExp[i],
                              bkg_model=bkg_mod) \
                                    for i, name in enumerate(line_names_exp)]
    lines = lines_gauss + lines_exp
    values_line_props = np.array([[l.flux(), l.eqw()] \
                                    for l in lines]).T.flatten()
    values = np.append(values, values_line_props)

    # === tabulating fit overview
    #columns_result = ['ndata', 'nvarys', 'chisqr', 'redchi', 'aic', 'bic']
    #values_result = [getattr(result, c) for c in columns_result]
    columns_result = list(stats_dict.keys())
    values_result = list(stats_dict.values())
    values = np.append(values, values_result)
    columns = np.append(columns, columns_result)
    
    if obj_idx is None:
        obj_idx = 0
    edges = (sampleFit.fit_window[obj_idx] / (1 + params['z'].value)).T
    values_edges = [edge for edge in edges.flatten()]
    columns_edges = [f'fit_window_edge_{i}' \
                     for i in range(len(values_edges))]
    values = np.append(values, values_edges)
    columns = np.append(columns, columns_edges)
    
    # === tabulating MCMC set-up
    columns_mcmc = list(mcmc_dict.keys())
    values_mcmc = list(mcmc_dict.values())
    values = np.append(values, values_mcmc)
    columns = np.append(columns, columns_mcmc)
    
    # === tabulating other info
    columns_other = ['flux_norm']
    columns = np.append(columns, columns_other)
    values_other = [sampleFit.flux_norm[obj_idx]]
    values = np.append(values, values_other)
    
    tab = Table(data=values, names=columns)
    tab['flux_norm'].unit = 'erg / s / cm^2 / A'
    return tab

def verify_best_params(args, params, tab_props):
    """
    Verify the best-fit parameters for a given spectrum.
    Compare median and ML parameter estimators (by chisquare) 
    and output the best ones.

    Parameters
    ----------
    args : tuple
        A tuple containing the data and the best-fit model. The elements of the
        tuple are:
        - x : array
            The wavelength array.
        - y : array
            The flux array.
        - ye : array
            The flux uncertainty array.
        - model : lmfit.Model
            Fitted model.
    params : lmfit.Parameters
        The best-fit parameters (median values).
    tab_props : astropy.table.Table
        The table of properties (includes ML values and fit data).

    Returns
    -------
    params : lmfit.Parameters
        The best fitting parameters (that provide the lowest chi-square).
    """
    x, y, ye, model = args
    params = params.copy()
    is_var = np.array([params[p].vary for p in params])
    pars_var_names = np.array([p for p in params])[is_var]
    #best = np.array([params[p] for p in pars_var_names])
    mle = np.array([tab_props[f"{p}_mle"] for p in pars_var_names]).squeeze()
    
    #=== compare chisquare of the best-fit models with MLE and median params
    # medians
    y_model = model.eval(params=params, x=x)
    chisqr = np.sum((y_model - y)**2 / (ye**2))
    
    # MLE values
    params_mle = params.copy()
    [params_mle[p].set(value=v) for p, v in zip(pars_var_names, mle)]
    y_model = model.eval(params=params_mle, x=x)
    chisqr_mle = np.sum((y_model - y)**2 / (ye**2))
    
    params_best = None
    if chisqr_mle < chisqr:
        params_best = params_mle.copy()
    else:
        params_best = params.copy()
    
    print(f"Chisqr Comparison (mle / median): {chisqr_mle:.2f} / {chisqr:.2f}")
    
    return params_best

def plot_mcmc_traces(lmfit_result, fpath_fig=None):
    #from IPython.display import display, Math

    # get the median and +/- one sigma parameter values
    params_best, theta_best = get_mcmc_estimates(lmfit_result.var_names, 
                                                 lmfit_result.flatchain.values)
    #params_spec = params_spec | params
    #print_mcmc_estimates(lmfit_result.var_names, lmfit_result.flatchain.values)

    # get max-probability params
    flat_lnProb = lmfit_result.lnprob.flatten()
    theta_maxProb = lmfit_result.flatchain.values[np.argmax(flat_lnProb)]

    #=== plot the traces ========================================
    ndim = lmfit_result.chain.shape[2]
    idxs = np.array(np.arange(ndim))
    #labels_sel = np.array(labels)[idxs]
    #samples_sel = flat_samples[:, idxs]
    nrows = len(idxs)

    fig, axes = plt.subplots(nrows, figsize=(10, nrows), sharex=True)
    for i, idx in enumerate(idxs):
        ax = axes[i]
        ax.plot(lmfit_result.chain[:, :, idx], "k", alpha=0.3, lw=0.5)
        ax.axhline(theta_best[idx], color="C0", lw=2, ls=':')
        ax.set_xlim(0, len(lmfit_result.chain))
        #ax.set_ylabel(lmfit_result.var_names[idx]))
        ax.annotate(lmfit_result.var_names[idx], (0.01, 0.7), 
                    xycoords='axes fraction', 
                    color='C1', fontweight='bold', 
                    bbox=dict(facecolor='w', alpha=0.7, edgecolor='none'))
        ax.yaxis.set_label_coords(-0.1, 0.5)

    #if ndiscard != None:
    #    for ax in axes:
    #        ax.axvline(ndiscard, lw=1.0, ls='--')
    axes[-1].set_xlabel("step number");
    
    if not fpath_fig is None:
        plt.savefig(fpath_fig, bbox_inches='tight')
    plt.close()

def plot_mcmc_corner(lmfit_result, fpath_fig=None):
    import corner
    
    result_dict = dict([[pname, p.value] \
                        for pname, p in lmfit_result.params.items() \
                        if p.vary == True])
    emcee_plot = corner.corner(lmfit_result.flatchain, 
                               labels=lmfit_result.var_names,
                               truths=list(result_dict.values()))
    if not fpath_fig is None:
        plt.savefig(fpath_fig, bbox_inches='tight')
    plt.close()

def tabulate_output(output):
    
    all_columns = list(output.keys()) # column names
    data_types = [np.int32]*2 + [float]*(len(all_columns)-2) # data types

    values = [] # column values
    for k in output.keys():
        values.append(output[k])
    values = np.hstack(values)

    tab = Table(data=values, names=all_columns, dtype=data_types)
    return tab

def plot_fit(x, y, ye,
             subplot_kw=None, gridspec_kw=None, fig_kw=None):
    
    # set up the keyword arguments
    _fig_kw = {'figsize': (4, 4), 'dpi': 150,}
    if not (fig_kw is None): # update figure kws with any user inputs
        _fig_kw = _fig_kw | fig_kw
    
    _subplot_kw = {}
    if not (subplot_kw is None):
        _subplot_kw = _subplot_kw | subplot_kw
    
    _gridspec_kw = {'height_ratios': [3, 1], 'hspace': 0.0}
    if not (gridspec_kw is None):
        _gridspec_kw = _gridspec_kw | gridspec_kw
    
    # create a figure
    fig, (ax, axr) = plt.subplots(2, 1, sharex=True,
                           subplot_kw=_subplot_kw,
                           gridspec_kw=_gridspec_kw,
                           **_fig_kw)
    
    axr.set_ylim(-10, 10)
    axr.set_yticks([-5, 0, 5])
    axr.set_yticks(np.arange(-5, 6, 1), minor=True)
    axr.grid(True, axis='y', which='both', ls=':')
    axr.set_xlabel(r'Wavelength [${\rm \AA}$]')
    axr.set_ylabel(r'$\chi$')
    ax.set_ylabel(r'$f_{\lambda}$ [erg s$^{-1}$ cm$^{-2}~{\rm \AA^{-1}}$]')
    
    ax.axhline(0.0, ls='--', c='k', lw=0.5)
    
    ax.step(x, y, lw=0.5, where='mid', c='k')
    ax.fill_between(x, y - ye, y + ye, alpha=0.2, 
                    step='mid', color='k')
    
    #plt.show()
    return fig, (ax, axr)

def make_figure(fig_kw=None, subplot_kw=None, gridspec_kw=None):
    
    # set up the keyword arguments
    _fig_kw = {'nrows': 2, 'ncols': 1, 
               'sharex': True, 'dpi': 150}
    if not (fig_kw is None): # update figure kws with any user inputs
        _fig_kw = _fig_kw | fig_kw
    #_fig_kw['figsize'] = (2 * _fig_kw['ncols'], 4)
    
    _subplot_kw = {}
    if not (subplot_kw is None):
        _subplot_kw = _subplot_kw | subplot_kw
    
    _gridspec_kw = {'height_ratios': [3, 1], 
                    'hspace': 0.0, 'wspace': 0.0}
    if not (gridspec_kw is None):
        _gridspec_kw = _gridspec_kw | gridspec_kw
    
    # create a figure
    fig, axes = plt.subplots(**_fig_kw,
                             subplot_kw=_subplot_kw,
                             gridspec_kw=_gridspec_kw)
    axes = np.atleast_2d(axes) 
    if axes.shape[0] == 1:
        axes = axes.T # [row, col] x N
    #for axr in axes[1::2].flatten():
    #    axr.set_ylim(-8, 8)
    #    axr.set_yticks([-5, 0, 5])
    #    axr.set_yticks(np.arange(-5, 6, 1), minor=True)
    #    axr.grid(True, axis='y', which='both', ls=':', zorder=0)
    #    axr.tick_params(which='both', right=True)
    
    ylabel1 = r'$f_{\lambda}$ [erg s$^{-1}$ cm$^{-2}~{\rm \AA^{-1}}$]'
    ylabel2 = r'$\chi$'
    [ax.set_ylabel(ylabel1) for ax in axes[0::2,0].flatten()]
    [ax.set_ylabel(ylabel2) for ax in axes[1::2,0].flatten()]
    
    for ax in axes.flatten():
        #ax.axhline(0.0, ls='--', c='k', lw=0.5)
        ax.tick_params(which='both', right=True)
    [ax.set_yticklabels([]) for ax in axes[:, 1:].flatten()]
    
    return fig, axes

def exclude_fit_windows(input_bins, lines=[], hwhm=1e3):
    # make all individual windows to exclude
    window_bins = []
    for l in lines:
        x1 = l - hwhm / 3e5 * l
        x2 = l + hwhm / 3e5 * l
        window_bins.append([x1, x2])
    window_bins = np.array(window_bins)
    window_bins = window_bins[np.argsort(window_bins[:,0])]
    
    # merge excluded windows if needed
    window_bins_exclude = list(window_bins[:1].copy()) # start with the 1st bin
    for i in range(1, len(window_bins)):
        bin1 = np.asarray(window_bins_exclude[-1])
        bin2 = np.asarray(window_bins[i])
        idxs_bin = np.digitize(bin1, bin2)
        is_subset = idxs_bin == 1
        if is_subset.all():
            window_bins_exclude[-1] = bin2
        elif is_subset.any():
            window_bins_exclude[-1] = [bin1[~is_subset].item(), 
                                       bin2[is_subset].item()]
        else:
            window_bins_exclude.append(bin2)
    window_bins_exclude = np.array(window_bins_exclude)
    
    # exclude the windows
    new_bins = list(input_bins[0:1].copy())
    for bin in input_bins:
        if not np.array_equal(bin, new_bins[-1]):
            new_bins.append(bin)
        for bin_excl in window_bins_exclude:
            idxs_bin = np.digitize(bin_excl, new_bins[-1])
            is_subset = idxs_bin == 1
            
            if is_subset.all():
                # break up the current bin into two
                bin1 = [new_bins[-1][0], bin_excl[0]]
                bin2 = [bin_excl[1], new_bins[-1][1]]
                new_bins[-1] = bin1
                new_bins.append(bin2)
            elif is_subset.any():
                
                if bin_excl[~is_subset].item() < new_bins[-1][0]:
                    new_bins[-1] = [*bin_excl[is_subset], new_bins[-1][1]]
                else:
                    new_bins[-1] = [new_bins[-1][0], *bin_excl[is_subset]]
    new_bins = np.array(new_bins)

    if len(new_bins) == 0:
        new_bins = input_bins
    return new_bins

def check_observability(df, lines_wavel, line_label):
    """
    Checks if a rest-frame wavelength is observable in the pandas.DataFrame
    provided. Returns a copy of the table with a new "observability_X" flag 
    column.
    """
    
    df_out = df.copy()

    # check which grating-filter combinations allow 
    # for Hb + OIII to be observable and 
    # flag the spectra accordingly
    lines_wavel = np.array([*lines_wavel])
    wavel_range_rf = np.array([lines_wavel.min(), lines_wavel.max()])
    redshifts = np.array(df_out['z'])
    gratings = np.array(df_out['grating'])
    filters = np.array(df_out['filter'])
    wavel_range = wavel_range_rf[None, :] * (1 + redshifts[:, None])

    grating_combos = {'g140m_f070lp': [ 7000, 12700],
                      'g140m_f100lp': [ 9700, 18400],
                      'g140h_f070lp': [ 8100, 12700],
                      'g140h_f100lp': [ 9700, 18200],
                      'g235m_f170lp': [16600, 30700],
                      'g235h_f170lp': [16600, 30500],
                      'g395m_f290lp': [28700, 51000],
                      'g395h_f290lp': [28700, 51400],
                      'prism_clear': [5800, 53366],
                     }

    # extend the wavel range to allow at least one wing 
    # of a  broad line to be in the grating
    vel_wing = 3000 # km/s
    for k, bin_edges in grating_combos.items():
        xmin, xmax = bin_edges
        xmin_new = xmin - vel_wing / 3e5 * xmin
        xmax_new = xmax + vel_wing / 3e5 * xmax
        grating_combos[k] = [xmin_new, xmax_new]

    new_column = f'observable_{line_label}'

    #if new_column not in df_out.columns:
        
    grat_filt = np.array([f"{g.lower()}_{f.lower()}"\
                        for g, f in zip(gratings, filters)])
    has_line = np.zeros_like(redshifts, dtype=int)

    # check which grating-filter combos are suitable to see the required ranges
    # and flag the spectra with those combinations
    for combo_name, combo_range in grating_combos.items():
        grat_with_line = np.full_like(redshifts, fill_value='NaN').astype(str)
        wavel_range_bins = np.digitize(wavel_range, bins=combo_range)
        is_grating = (wavel_range_bins == 1).all(axis=1)
        grat_with_line[is_grating] = combo_name
        is_match = grat_with_line == grat_filt # check for "grat_filt" condition
        has_line = has_line + is_match.astype(int)

    if new_column in df_out.columns:
        df_out.drop(columns=[new_column], inplace=True)
    ncols = df_out.shape[1]
    df_out.insert(loc=ncols, column=new_column, value=has_line)
    for groupid, group in df_out.groupby('GroupID'):
        flag = group[new_column] == 1
        #print(groupid, flag.sum())
    
    #print(f"Updated (added {new_column})")
    return df_out



""" Operational Functions """
def gratings_summary(df, sn_thresh=10):

    is_grating = ~df['grating'].isin(['PRISM'])
    is_duplicated = df['GroupID'].duplicated(keep=False)

    _df_sample = df[~is_duplicated & is_grating].copy()
    for groupid, group in df[is_duplicated & is_grating].groupby('GroupID'):
        highest_sn50_Ha = group['sn50_Ha'].argmax(skipna=True)
        _df_sample = pd.concat([_df_sample, group.iloc[[highest_sn50_Ha]]])
        
        has_highres = group['grating'].isin(GRATINGS_HRES)
        if has_highres.any():
            _df_sample = pd.concat([_df_sample, group.loc[has_highres]])

    sel_high_snr = (_df_sample['sn50_Ha'] > sn_thresh) &\
                (_df_sample['observable_Ha_6565'].isin([1]))
    sel_low_snr = (_df_sample['sn50_Ha'] <= sn_thresh) &\
                (_df_sample['observable_Ha_6565'].isin([1]))

    n_high_snr = _df_sample.loc[sel_high_snr, 'GroupID'].unique().shape[0]
    n_low_snr = _df_sample.loc[sel_low_snr, 'GroupID'].unique().shape[0]    
    print(f"highest-SNR sample: {n_high_snr}")
    for grat, group in _df_sample[sel_high_snr].groupby('grating'):
        n_grat = group['GroupID'].unique().shape[0]
        groupids = group['GroupID'].sort_values().unique()
        print(f" * {grat}: {n_grat} \t{groupids}")

    print(f"\nlow-SNR sample: {n_low_snr}")
    for grat, group in _df_sample[sel_low_snr].groupby('grating'):
        n_grat = group['GroupID'].unique().shape[0]
        groupids = group['GroupID'].sort_values().unique()
        print(f" * {grat}: {n_grat} \t{groupids}")
    return

def plot_sample_spectra(fpaths, srcids, z, lines_wavel, plot_sn=False):

    if len(lines_wavel) > 1:
        lines_wavel = np.atleast_1d(lines_wavel)
        xmin = lines_wavel.min() - 1 * (lines_wavel.max() - lines_wavel.min())
        xmax = lines_wavel.max() + 1 * (lines_wavel.max() - lines_wavel.min())
    else:
        wavel = lines_wavel[0]
        xmin = wavel - 20e3 / 3e5 * wavel # some X km/s range around 'wavel'
        xmax = wavel + 20e3 / 3e5 * wavel # some X km/s range around 'wavel'

    n_sel = len(fpaths)
    for idx in range(n_sel):
        
        s = msaexp.spectrum.SpectrumSampler(fpaths[idx])
        to_flam = s.spec['to_flam'][s.valid].data
        x = s.spec['wave'][s.valid].data * 1e4
        y = s.spec['flux'][s.valid].data * to_flam
        ye = s.spec['err'][s.valid].data * to_flam
        
        _xmin = xmin * (z[idx] + 1)
        _xmax = xmax * (z[idx] + 1)
        in_xrange = (x > _xmin) & (x < _xmax)
        if not in_xrange.any():
            continue
        ymin = y[in_xrange].min() - 0.1*(y[in_xrange].max()-abs(y[in_xrange].min()))
        ymax = y[in_xrange].max() + 0.1*(y[in_xrange].max()-abs(y[in_xrange].min()))

        fig, ax = plt.subplots(1, figsize=(4, 2), dpi=200)
        ax.set_xlabel(r'Wavelength ($\mu$m)')
        ax.set_ylabel(r'$F_{\lambda}$')
        ax.set_ylim(ymin, ymax)
        
        if plot_sn:
            ax.step(x, y/ye, lw=0.7, where='mid')
        else:
            ax.step(x, y, lw=0.7, where='mid')
        #ax.fill_between(x, y - ye, y + ye, color='C0', alpha=0.2, step='mid')
        
        if not np.isnan(z[idx]):
            print((z[idx] + 1) * np.array(lines_wavel))
            [ax.axvline(w * (z[idx] + 1), 
                        ax.get_ylim()[0], ax.get_ylim()[1], 
                        color='r', ls='--', lw=0.7) \
                            for w in np.array(lines_wavel)]
            ax.set_xlim(_xmin-1e3, _xmax+1e3)
        
        text = f'ID {srcids[idx]}'
        ax.text(0.05, 0.9, text, transform=ax.transAxes, fontsize=8,
                va='top', ha='left')
        
        plt.show()
    return

def get_sample_dict(df, verbose=True):
    # get info for loading the spec sample into one class object
    sample_info = {}
    sample_info['fnames'] = df['file'].values
    sample_info['roots'] = df['root'].values
    #if download_data:
    #    sample_info['fpaths'] = sample_info['fnames']
    #else:
    #    sample_info['fpaths'] = np.array([PATH_LOCAL.format(file=f) \
    #                                      for f in sample_info['fnames']])
    sample_info['obj_ids'] = df['PROG_ID'].values
    sample_info['idxs'] = df.index.values
    if 'z' in df.columns:
        sample_info['z_input'] = df['z'].values
    else:
        sample_info['z_input'] = np.full_like(sample_info['fnames'], np.nan)

    # if already fitted use the best-fit redshifts
    n_fitted_once = 0
    groupids_notfitted = [srcid for srcid in sample_info['obj_ids']]
    #src_outputs = group_outputs(fnames_dja=sample_info['fnames'],
    #                            src_ids=sample_info['obj_ids'],
    #                            line_label=line_label)
    #for i, srcid in enumerate(sample_info['obj_ids']):
    #    if srcid in src_outputs:
    #        
    #        data = src_outputs[srcid]['z'].data
    #        if hasattr(src_outputs[srcid]['z'], 'mask'):
    #            mask = src_outputs[srcid]['z'].mask
    #        else:
    #            mask = np.zeros_like(data, dtype=bool)
    #        argmin = np.argmin(src_outputs[srcid][~mask]['redchi'])
    #        zbest = data[argmin]
    #        if zbest is not np.ma.masked:
    #            sample_info['z_input'][i] = zbest
    #            n_fitted_once += 1
    #    else:
    #        groupids_notfitted.append(srcid)
    
    n_sel = len(df)
    if verbose:
        print(f'objects selected: {n_sel}')
        print(f'already fitted at least once: {n_fitted_once}/{n_sel}')
        print(f'groupids not fitted: {groupids_notfitted}')
    
    return sample_info

def get_final_sample(df, line_label, sn_thresh, use_prism=False, 
                     write_sample=None, plot_spectra=False, plot_sn=False, 
                     lines_wavel=None):
    
    # keep relevant gratings
    is_observable = df[f'observable_{line_label}'].isin([1])
    df = df[is_observable]
    if len(df) == 0:
        raise ValueError('No spectra found')
    
    # check for duplicates and keep the highest SN spectra
    is_duplicated = df['GroupID'].duplicated(keep=False)
    unique = (~is_duplicated)
    if not use_prism:   
        not_prism = ~df['grating'].isin(['PRISM'])
        unique &= not_prism
        is_duplicated &= not_prism
        #if not unique.any():
        #    highest_sn_idx = df['sn50_Ha'].argmax(skipna=True)
        #    unique.iloc[highest_sn_idx] = True
        #    if not unique.any():
        #        raise ValueError('No unique non-prism spectra.')
    _df = df[unique].copy()
    for _, group in df[is_duplicated].groupby('GroupID'):
        
        use_highres = group['grating'].isin(GRATINGS_HRES) &\
                     (group[f'sn50_{line_label}'] >= sn_thresh)
        if use_highres.any():
            _df = pd.concat([_df, group.loc[use_highres]])
        else:
            highest_sn50_Ha = group['sn50_Ha'].argmax(skipna=True)
            _df = pd.concat([_df, group.iloc[[highest_sn50_Ha]]])
    df = _df.copy()
    df.sort_values(by='sn50_Ha', ascending=False, inplace=True)
    #df = pd.merge(df_ext[['file']], df, on='file', how='right')
    n_sel = len(df)
    del _df
    
    if not (write_sample is None):
        df.to_csv(write_sample, index=False)
    
    # plot to check the redshift estimate
    if plot_spectra:
        sample_info = get_sample_dict(df.copy(), line_label=line_label)
        plot_sample_spectra(sample_info['fpaths'], 
                            sample_info['obj_ids'], 
                            sample_info['z_input'], lines_wavel, 
                            plot_sn=plot_sn)
    df.reset_index(drop=True, inplace=True)
    return df

def load_model_from_file(output_label, sampleFit, 
                         fpaths_spec=None, fit_edges=None,
                         set_dispersion_limit=False,
                         plot_model=0):

    # set_dispersion_limit: set the smallest wavel element in models

    # load a model from file
    fname_model_module = f"py_model_{output_label}"
    module = importlib.import_module(fname_model_module)
    from importlib import reload
    reload(module)
    params_lines = module.params_lines
    model_components = module.model_components
    components = model_components.components

    #=== temporary
    # load the fit window edges
    if not (fpaths_spec is None):
        for i, fn in enumerate(fpaths_spec):
            groupid = sampleFit.obj_ids[i]
            froot = fn.split('/')[-1].split('.')[0]
            fname = f'mcmc_result-{output_label}-{froot}-grp{groupid}.fits'
            fname = fname.replace('_instrum_broad', '')
            fpath = os.path.join(PATH_OUT, fname)
            try:
                _tab = Table.read(fpath)
            except:
                _tab = Table.read(fpath.replace('_offset', ''))
            
            n_edges = np.sum(['fit_window_edge_' in c for c in _tab.columns])
            fit_edges = [_tab[f'fit_window_edge_{i_edge}'].item() \
                        for i_edge in range(n_edges)\
                        if not isinstance(_tab[f'fit_window_edge_{i_edge}'], 
                                            type(np.ma.masked))]
            fit_edges = np.array(fit_edges).reshape(-1, 2)
        sampleFit.update_fit_window(fit_edges)    
    elif not (fit_edges is None):
        sampleFit.update_fit_window(fit_edges)
    else:
        sampleFit.update_fit_window(None)

    if set_dispersion_limit:
        # set the minimum line widths to the resoluiton element 
        # at the centroid wavelength
        keys_sigma = [p for p in params_lines if 'sigma' in p]
        keys_center = [p.replace('_sigma', '_center') for p in keys_sigma]

        x_rf = sampleFit.x / (1 + sampleFit.z_input[:, None])
        Rsig = x_rf / sampleFit.R / 2.355 # resolution element [sigmas]
        Rsig_center = [np.interp(params_lines[k].value, x_rf[i], Rsig[i]) \
                        for k in keys_center \
                        for i in range(sampleFit.n_obj)]

        [params_lines[k].set(min=v) for k, v in zip(keys_sigma, Rsig_center)];
    else:
        keys_sigma = [p for p in params_lines if 'sigma' in p]
        keys_center = [p.replace('_sigma', '_center') for p in keys_sigma]
        Rsig_center = np.full_like(keys_sigma, 0.0, dtype=float)
        [params_lines[k].set(min=v) for k, v in zip(keys_sigma, Rsig_center)];

    n_var = len([p for p in params_lines if params_lines[p].vary])
    print(f"py_model_{output_label}.py")
    print(f"number of parameters: {n_var}")

    # plot a model with the guessed parameters 
    # (plot only one index from a sample)
    idx_obj = np.nan
    if type(plot_model) == int:
        idx_obj = plot_model
    elif plot_model:
        idx_obj = 0
    
    if not np.isnan(idx_obj):
        #=== plot the step=0 model
        z_guess = sampleFit.z_input[idx_obj]
        params_lines['z'].set(value=z_guess)
        
        xmin_fit = np.min(sampleFit.x[idx_obj][sampleFit.in_window[idx_obj]])
        xmin_fit /= (1 + sampleFit.z_input[:, None])
        xmax_fit = np.max(sampleFit.x[idx_obj][sampleFit.in_window[idx_obj]])
        xmax_fit /= (1 + sampleFit.z_input[:, None])
        x_eval = np.arange(xmin_fit, xmax_fit, 1)[None, :] * \
                 (1 + sampleFit.z_input[:, None])
        x = x_eval[idx_obj] / (1 + z_guess)
        y = model_components.eval(params=params_lines, x=x)
        xobs = sampleFit.x[idx_obj] / (1 + z_guess)
        yobs = sampleFit.y[idx_obj]
        in_xrange = (xobs > xmin_fit) & (xobs < xmax_fit)
        _ymin = np.percentile(yobs[in_xrange.ravel()], 0.01)
        _ymax = np.percentile(yobs[in_xrange.ravel()], 99.99)
        ymin = _ymin - 0.2 * (_ymax - _ymin)
        ymax = _ymax + 0.1 * (_ymax - _ymin)
        plt.figure(dpi=200)
        plt.step(xobs, yobs, c='k', lw=0.5)
        plt.plot(x[np.isfinite(y)], y[np.isfinite(y)], lw=1, 
                 c='magenta', label='model')
        plt.xlim(xmin_fit, xmax_fit);
        plt.ylim(ymin, ymax)

        for i, comp in enumerate(components):
            y = comp.eval(params=params_lines, x=x)
            plt.plot(x, y, ls='--', lw=1, label=f'comp {i}')

        plt.fill_between(xobs, ymin, ymax, where=sampleFit.in_window[idx_obj], 
                        alpha=0.1, color='C0', lw=2)
        plt.legend()
        plt.show()
    
    return params_lines, model_components

def plot_best_fits(sampleFit):
    xmin_fit, xmax_fit = np.min(sampleFit.fit_window), np.max(sampleFit.fit_window)
    xmin_fit, xmax_fit = xmin_fit / (1 + sampleFit.z_input), xmax_fit / (1 + sampleFit.z_input)
    x_eval = np.arange(xmin_fit, xmax_fit, 1)[None, :] * \
                    (1 + sampleFit.z_input[:, None])
    components = sampleFit.model_components.components
    for idx in range(sampleFit.n_obj):
        obj_id = sampleFit.obj_ids[idx]
        mask = sampleFit.in_window[idx] & sampleFit.valid[idx]
        #x = np.ma.masked_where(~mask, sampleFit.x[idx])
        #y_flux = np.ma.masked_where(~mask, sampleFit.y[idx])
        #ye = np.ma.masked_where(~mask, sampleFit.ye[idx])
        x = sampleFit.x[idx][mask]
        y_flux = sampleFit.y[idx][mask]
        ye = sampleFit.ye[idx][mask]
        _z = sampleFit.z_input[idx]
        x_rf = x / (1+_z)
        x_eval_plot = x_eval[idx] / (1+_z)
        
        ymod = sampleFit.eval_sample(x_eval=x_eval)[idx]
        
        #y_total_model = y_eval_model[idx] + y_lines_model[idx]
        
        y_comps = [components[i].eval(params=sampleFit.params_sample[idx], 
                                      x=x_rf) \
                    for i in range(len(components))]

        _ymin = y_flux[(x/(1+_z) > xmin_fit) & (x/(1+_z) < xmax_fit)].min()
        _ymax = y_flux[(x/(1+_z) > xmin_fit) & (x/(1+_z) < xmax_fit)].max()
        ymin = _ymin - 0.1 * (_ymax - _ymin)
        ymax = _ymax + 0.1 * (_ymax - _ymin)

        fig, ax = plt.subplots(1, figsize=(5, 3), dpi=150)
        ax.set_xlabel(r'Wavelength ($\AA$)')
        ax.set_ylabel(r'$F_{\lambda}$')
        ax.set_xlim(xmin_fit, xmax_fit)
        ax.set_ylim(ymin, ymax)

        ax.step(x_rf, y_flux, lw=0.7, where='mid', c='C0')
        ax.fill_between(x_rf, y_flux - ye, y_flux + ye, alpha=0.3, 
                        step='mid', color='C0')
        ax.plot(x_eval_plot, ymod, lw=1, c='C1')
        [ax.plot(x_rf, y_comp, lw=0.5, ls='--', label=f'Comp {j+1}') \
            for j, y_comp in enumerate(y_comps)]
        #ax.step(x_rf, y_model_obsx[idx], lw=0.7, where='mid', c='C3')
        ax.text(0.05, 0.9, f"ID {obj_id}", fontsize=8, transform=ax.transAxes)

        ax.axhline(0.0, ls='--', c='k', lw=0.7)
        #ax.legend(fontsize=8)
        
        plt.show()
    return


def logLik(p, x, y, ye, model):
    """ log likelihood function for MCMC """
    #y_pred = sampleFit.model.eval(params=p, x=x)
    y_pred = model.eval(params=p, x=x)
    
    #print("here is main", os.getpid())
    #return -0.5 * np.sum( ((y - y_pred) / ye)**2 )
    #return (y - y_pred) / ye
    return np.abs( ((y - y_pred) / ye) )

#def _reduce_fcn(x):
#    return -0.5 * np.sum(x**2)

def get_model(sampleFit):
    return deepcopy(sampleFit.model)

def fit_emcee(sampleFit, params, mcmc_dict, output_label,
              set_guess_from_leastsq=True, workers=1, fit_redshift=True,
              save_outputs=True, skip_initial_state_check=False):

    from copy import deepcopy
    
    # get a dictionary of all existing fitting outputs
    rng = lmfit.minimizer._make_random_gen(123)
    
    params_current = params.copy()

    for idx in range(sampleFit.n_obj):
        
        print(f"\n=== Fitting object {idx+1} / {sampleFit.n_obj} ===")

        mask = sampleFit.valid[idx] & sampleFit.in_window[idx]
        z_in = sampleFit.z_input[idx]
        x = sampleFit.x[idx][mask]
        y = sampleFit.y[idx][mask]
        ye = sampleFit.ye[idx][mask]
        groupid = sampleFit.obj_ids[idx]
        arg_mod = deepcopy(sampleFit.model)
        #arg_mod = get_model(sampleFit)

        nvarys = np.sum([params_current[p].vary for p in params_current])
        args = (x, y, ye, arg_mod)
        
        eps = rng.randn(mcmc_dict['nwalkers'], nvarys) * 5e-4
        eps[:,-1] = rng.randn(mcmc_dict['nwalkers']) * 1e-4
        p0 = 1 + eps # VR

        # use least_squares solution as the starting guess
        if set_guess_from_leastsq:
            params_current.update(sampleFit.params_sample[idx])
        if fit_redshift:
            params_current['z'].set(value=sampleFit.z_input[idx], 
                                min=0.95*sampleFit.z_input[idx]-1e-6,
                                max=1.05*sampleFit.z_input[idx]+1e-6)
        else:
            params_current['z'].set(vary=False)
        #params_current['scale_disp'].set(value=0.75)
        #params_current['Av'].set(value=1e-2)
        
        # is_weighted=False if the objecting function returns a float; 
        # is_weighted=True if returns an array
        #mp.set_start_method('fork', force=True)
        #with mp.Pool(2) as pool:
        result = lmfit.minimize(logLik, params=params_current.copy(), 
                                args=args,
                                method='emcee', nan_policy='omit', 
                                nwalkers=mcmc_dict['nwalkers'], 
                                burn=mcmc_dict['ndiscard'], 
                                steps=mcmc_dict['nsteps'], 
                                thin=mcmc_dict['nthin'], 
                                #reduce_fcn=_reduce_fcn,
                                is_weighted=True, 
                                progress=True, 
                                workers=workers, 
                                run_mcmc_kwargs={'initial_state': p0,
                                                'skip_initial_state_check':skip_initial_state_check})
        
        ndim = result.nvarys # num of optimized params
        print(f"# of params: {ndim}")
        
        
        #=== save fit results ===========================================
        #params_sample_mcmc.append(result.params.copy())
        #result.params.update(sampleFit.params)
        #result.params.update(result.params.copy())
        
        stats_names = ['ndata', 'nvarys', 'chisqr', 'redchi', 'aic', 'bic']
        stats_dict = dict(zip(stats_names,
                              [getattr(result, c) for c in stats_names]))
        tab = tabulate_emcee_results(result.params, result.lnprob, result.chain,
                                     sampleFit, mcmc_dict, stats_dict,  
                                     obj_idx=0)
        result.params = verify_best_params(args, result.params.copy(), tab[0])
        #sampleFit.params |= result.params.copy()
        
        if save_outputs:
            # first, dump best-fit Parameters from LMFIT
            froot = sampleFit.fpaths[idx].split('/')[-1].split('.')[0]
            fname_save = f'mcmc_params-{output_label}-{froot}-grp{groupid}.pckl'
            fpath_save = os.path.join(PATH_OUT, fname_save)
            if os.path.exists(fpath_save):
                print(f'Path exists. Overwriting... Path: {fpath_save}\n')
            with open(fpath_save, 'wb') as f:
                pickle.dump(result.params.dumps(), f)
            
            # second, MCMC chain
            mcmc_chain = {}
            mcmc_chain['acceptance_fraction'] = result.acceptance_fraction#[nwalk]
            mcmc_chain['lnprob'] = result.lnprob # [nsamp, nwalkers]
            mcmc_chain['chain'] = result.chain # [nsamp, nwalkers, ndim]
            
            fname_save = f'mcmc_chain-{output_label}-{froot}-grp{groupid}.pckl'
            fpath_save = os.path.join(PATH_OUT, fname_save)
            with open(fpath_save, 'wb') as f:
                pickle.dump(mcmc_chain, f)
            
            # finally, save all fit results as a single-entry table
            fname_save = f'mcmc_result-{output_label}-{froot}-grp{groupid}.fits'
            fpath_save = os.path.join(PATH_OUT, fname_save)
            tab.write(fpath_save, overwrite=True)
            
            # save mcmc plots
            fname_save = f'mcmc_trace-{output_label}-{froot}-grp{groupid}.pdf'
            fpath_save = os.path.join(PATH_OUT, fname_save)
            plot_mcmc_traces(result, fpath_fig=fpath_save)
            
            fname_save = f'mcmc_posterior-{output_label}-{froot}-grp{groupid}.pdf'
            fpath_save = os.path.join(PATH_OUT, fname_save)
            plot_mcmc_corner(result, fpath_fig=fpath_save)
        
        return result, tab

def get_line_wavel(w1, w2, lines_dict=lw, exclude=[]):
    names, waves = [], []
    for k, l in lines_dict.items():
        l = np.atleast_1d(l)
        cond = (l > w1) & (l < w2)
        if cond.any() & (k not in exclude):
            names.append(k)
            waves.append([*l[cond]])
    return names, waves