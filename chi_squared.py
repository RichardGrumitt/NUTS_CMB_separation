import numpy as np
import healpy as hp
from numpy_diffuse_models import *


def chi_squared(data_map_list, data_cov_list, freq, map_field, out_filename,
                synch_model='power_law', dust_model='modified_blackbody', synch_nu0=5.0, dust_nu0=402.0,
                synch_amp=None, synch_beta=None, synch_curve=None, dust_amp=None, dust_beta=None,
                dust_temp=None, cmb_amp=None):

    """

    Function for computing chi-squared maps from a given run.
    NOTE: Although this seems to calculate a reduced chi-squared, it isn't actually. The actual
    number of dof in a Bayesian model is complicated somewhat by the use of priors, hyper-parameters
    etc. All that matters is that we have some value that we can threshold when it comes to 
    masking though, so this is ok. The absolute value of chi_squared is not important!

    Inputs
    ------
    data_map_list: List of input maps used in component separation - list.
    data_cov_list: List of associated covariance maps - list.
    freq: Frequencies of maps - numpy.ndarray.
    map_field: Map field to read in for inputs - int.
    out_filename: Output filename for the chi_squared map - str.
    synch_model: Synchrotron model name - str.
    dust_model: Dust model name - str.
    synch_nu0: Synchtron reference frequency - float.
    dust_nu0: Dust reference frequency - float.
    synch_amp: Synchrotron amplitude map - numpy.ndarray.
    synch_beta: Synchtron spectral index map - numpy.ndarray.
    synch_curve: Synchrotron spectral curvature map - numpy.ndarray.
    dust_amp: Dust amplitude map - numpy.ndarray.
    dust_beta: Dust spectral index map - numpy.ndarray.
    dust_temp: Dust temperature map - numpy.ndarray.
    cmb_amp: CMB amplitude map - numpy.ndarray.

    Outputs
    -------
    chi_squared: chi_squared map - numpy.ndarray. 

    """

    chi_squared = []
    
    for i in range(len(data_map_list)):

        data = hp.read_map(data_map_list[i], field=map_field)
        cov = hp.read_map(data_cov_list[i], field=map_field)
        nu = freq[i]

        synch = Synchrotron(Model=synch_model, Nu=nu, Amp=synch_amp, Nu_0=synch_nu0,
                            Spectral_Index=synch_beta, Spectral_Curvature=synch_curve)
        dust = Thermal_Dust(Model=dust_model, Nu=nu, Amp=dust_amp, Nu_0=dust_nu0,
                            Spectral_Index=dust_beta, Dust_Temp=dust_temp)
        cmb = CMB(Model='cmb', Nu=nu, Acmb=cmb_amp)
        model_signal = synch.signal() + dust.signal() + cmb.signal()
        
        chi_squared.append((data - model_signal)**2 / cov)

    chi_squared = np.array(chi_squared)
    chi_squared = np.sum(chi_squared, axis=0)

    hp.write_map(out_filename, chi_squared, overwrite=True)

    return chi_squared
