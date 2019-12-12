import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
from scipy.stats import iqr


h = 6.62607004*10**(-34)
kb = 1.38064852*10**(-23)
Tcmb = 2.7255


def cmb_SED(Acmb, nu):

    """

    CMB SED

    :param Acmb: CMB amplitude - numpy.ndarray.
    :param nu: Frequency to evaluate the CMB SED - float.

    """

    x = h*nu/(kb*Tcmb)
    return Acmb*x**2*np.exp(x)/(np.exp(x) - 1.0)**2


def brightness_mask(cmb_map, synch_template, synch_freq, dust_template, dust_freq, mask_file, regions_file, threshold=10.0):

    """

    Function for creating an analysis mask, following the recipe in the CORE B-mode forecast paper.

    :param cmb_map: CMB P map at 70GHz - numpy.ndarray.
    :param synch_template: Synchrotron P template map - numpy.ndarray.
    :param synch_freq: Synchrotron template frequency (GHz) - float.
    :param dust_template: Dust P template map - numpy.ndarray.
    :param dust_freq: Dust template frequency (GHz) - float.
    :param mask_file: Output filename for the mask - str.
    :param regions_file: Regions file being used in the fitting - str.
    :param threshold: Threshsold for masking - float.
    :return mask: Analysis mask - numpy.ndarray.

    """

    cmb_fluctuation = np.std(cmb_map)
    synch70 = synch_template*(70.0/synch_freq)**(-3.0)
    gamma = h/(kb*19.4)
    dust70 = dust_template*(70.0/dust_freq)**2.6*(np.exp(gamma*dust_freq*1.0e9) - 1.0)/(np.exp(gamma*70.0e9) - 1.0)

    regions = hp.read_map(regions_file)
    region_nums = [i for i in range(1, int(np.amax(regions)) + 1)]

    mask = np.ones(hp.nside2npix(hp.get_nside(cmb_map)))

    synch_idx = np.where(synch70>=threshold*cmb_fluctuation)[0]
    dust_idx = np.where(dust70>=threshold*cmb_fluctuation)[0]
    mask[synch_idx] = 0
    mask[dust_idx] = 0

    hp.write_map(mask_file, mask, overwrite=True)

    return mask

def post_hoc_mask(Q_neff_map, U_neff_map, Q_sigma_map, U_sigma_map, mask_filename,
                  neff_thresh=10000, sigma_thresh=0.7):

    """

    Function for masking on the basis of the effective sample size and sigma values of the CMB Q and U maps
    obtained after component separation.

    Inputs
    ------
    :Q_neff_map: Map of the effective sample size in Q - numpy.ndarray.
    :U_neff_map: Map of the effective sample size in U - numpy.ndarray.
    :Q_sigma_map: Sigma map in Q - numpy.ndarray.
    :U_sigma_map: Sigma map in U - numpy.ndarray.
    :mask_filename: Output filename for mask - str.
    :neff_thresh: Minimum number of effective samples for good pixel - float.
    :sigma_thresh: Fraction of sigma map we keep as good pixels - float.

    Returns
    -------
    :mask: Combined neff and sigma mask - numpy.ndarray.

    """

    mask = np.ones(len(Q_neff_map))
    mask[Q_neff_map < neff_thresh] = 0
    mask[U_neff_map < neff_thresh] = 0
    frac = len(mask[mask == 0]) / len(mask)
    print(f'Fraction masked after neff = {frac}')
    
    assert 0 <= sigma_thresh <= 1

    if sigma_thresh != 1.0:
    
        bw = 2 * iqr(Q_sigma_map) / len(Q_sigma_map)**(1 / 3)
        num_bins = int((np.amax(Q_sigma_map) - np.amin(Q_sigma_map)) / bw)
        hist, bins = np.histogram(Q_sigma_map, bins=num_bins, normed=True)
        dx = bins[1] - bins[0]
        qcum = np.cumsum(hist) * dx

        qspline = interp1d(bins[1:], qcum, kind='cubic', fill_value='extrapolate')
        qsol = root_scalar(lambda x: qspline(x) - sigma_thresh, x0 = np.mean(Q_sigma_map), method='bisect',
                           bracket=[np.amin(bins[1:]), np.amax(bins[1:])])

        bw = 2 * iqr(U_sigma_map) / len(U_sigma_map)**(1 / 3)
        num_bins = int((np.amax(U_sigma_map) - np.amin(U_sigma_map)) / bw)
        hist, bins = np.histogram(U_sigma_map, bins=num_bins, normed=True)
        dx = bins[1] - bins[0]
        ucum = np.cumsum(hist) * dx

        uspline = interp1d(bins[1:], ucum, kind='cubic', fill_value='extrapolate')
        usol = root_scalar(lambda x: uspline(x) - sigma_thresh, x0 = np.mean(U_sigma_map), method='bisect',
                           bracket=[np.amin(bins[1:]), np.amax(bins[1:])])

        mask[Q_sigma_map > qsol.root] = 0
        mask[U_sigma_map > usol.root] = 0

    hp.write_map(mask_filename, mask, overwrite=True)

    return mask
