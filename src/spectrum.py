import astropy.units as au
import msaexp.spectrum
import numpy as np

from tqdm import tqdm

import config as cfg

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
        
        # the resolution of g395m was scaled using 
        # one object with both g395h and g395m, where a narrow line 
        # (OIII 4959) is resolved.
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
            self.urls = [cfg.PATH_AWS.format(root=root, file=fname) \
                         for root, fname in zip(roots, fnames)]
        else:
            self.urls = [cfg.PATH_LOCAL.format(file=fname) \
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


class SpectrumSampleFit(SpectrumSample):
    
    def __init__(self, fnames, roots, idxs=None, obj_ids=None, 
                 fit_window_bins=None, z_input=None, flux_norm=1, message=None, 
                 verbose=True, download_data=True, **kwargs):
        
        """
        Set up the fit window of the spectrum
        
        fit_window : list
            Wwavelength window for spectra in the rest frame.  For example, 
            fit_window = [[wav1, wav2], [wav3, wav4], ...].
        """
        
        if z_input is None:
            self.z_input = np.zeros(len(fnames))
        else:
            self.z_input = z_input
        
        # init base class and derived class attributes
        SpectrumSample.__init__(self, fnames, roots, idxs=idxs, obj_ids=obj_ids,
                                z_input=z_input, flux_norm=flux_norm, 
                                verbose=verbose, download_data=download_data)
        self.n_obj = self.x.shape[0]
        
        self.fit_window = np.full((self.n_obj, 1, 2), np.nan)
        self.in_window = np.ones_like(self.x, dtype=bool)
        self.update_fit_window(fit_window_bins)
        
        # TODO: add an attribute: mask with objects with x (in)outside of 
        # fit_window
        
    def update_fit_window(self, fit_window_bins=None):
        
        # check keyword arguments
        if fit_window_bins is None:
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
    