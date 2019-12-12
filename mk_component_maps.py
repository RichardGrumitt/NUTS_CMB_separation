import healpy as hp
import numpy as np
import pandas as pd
from utils import mk_file_list
import sys


def param_map(trace_summary_dir, param_name, mask_file, out_dir, out_prefix):

    """

    Function for generating parameter maps from pixel traces.
    
    Inputs
    ------
    trace_summary_dir: The directory where trace summaries are stored - str.
    param_name: Name of the parameter in the pymc3 model - str.
    mask_file: Mask file used in component separation - str.
    out_dir: The output directory for the parameter maps - str.
    out_prefix: The output prefix for the parameter maps - str.

    Returns
    -------
    rhat_map: Map of the Gelman-Rubin statistic for the given parameter - numpy.ndarray.
    neff_map: Map of the effective sample size for the given parameter - numpy.ndarray.
    mu_map: Map of the mean parameter values in each pixel - numpy.ndarray.
    sigma_map: Map of the sd for the parameter in each pixel - numpy.ndarray.

    Also saves the maps.

    """

    mask = hp.read_map(mask_file)
    no_pix = len(mask)
    good_idx = np.where(mask == 1.)[0]
    
    rhat_map = np.ones(no_pix)*hp.UNSEEN
    neff_map = np.ones(no_pix)*hp.UNSEEN
    mu_map = np.ones(no_pix)*hp.UNSEEN
    sigma_map = np.ones(no_pix)*hp.UNSEEN

    trace_summary_files = mk_file_list(trace_summary_dir, '.csv', absolute=True)

    for i in range (0, len(good_idx)):

        progress = 100.0 * (i + 1.0) / len(good_idx)
        print(('Percentage through constructing maps = {}%'.format(progress)))

        df = pd.read_csv(trace_summary_files[i], index_col=0)
        rhat_map[good_idx[i]] = df['Rhat'][param_name]
        neff_map[good_idx[i]] = df['n_eff'][param_name]
        mu_map[good_idx[i]] = df['mean'][param_name]
        sigma_map[good_idx[i]] = df['sd'][param_name]
        
    hp.write_map('{}{}_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), rhat_map, overwrite=True)
    hp.write_map('{}{}_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), neff_map, overwrite=True)
    hp.write_map('{}{}_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), mu_map, overwrite=True)
    hp.write_map('{}{}_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), sigma_map, overwrite=True)

    return rhat_map, neff_map, mu_map, sigma_map


def region_param_map(trace_summary_dir, spectral_param_names, hyper_param_names, amp_param_names, region_file,
                     mask_file, out_dir, out_prefix, joint_QU=True, pix_thresh=1):

    """

    Function for generating parameter maps from pixel traces.

    Inputs
    ------
    trace_summary_dir: The directory where trace summaries are stored - str.
    spectral_param_names: List of normal spectral parameter names in the pymc3 model - list.
    hyper_param_names: List of spectral hyper-parameter names in the pymc3 model - list.
    amp_param_names: List of amplitude parameter names in the pymc3 model - list.
    region_file: Region file used in the component separation - str.
    mask_file: Mask file used in component separation - str.
    out_dir: The output directory for the parameter maps - str.
    out_prefix: The output prefix for the parameter maps - str.
    joint_QU: Whether fitting was done jointly on Q and U, assuming common spectral parameters - boolean.
    pix_thresh: Number of pixels at which we allow fitting - str.

    Returns
    -------
    Saves the rhat, neff, mu and sigma maps for each parameter.

    """

    regions = hp.read_map(region_file)
    region_nums = [i for i in range(1, int(np.amax(regions)) + 1)]
    
    mask = hp.read_map(mask_file)
    no_pix = len(mask)

    fitted_regions = []
    for num in region_nums:
        mask_pix = mask[regions == num]
        if len(mask_pix[mask_pix == 1]) >= pix_thresh:
            fitted_regions.append(num)

    trace_summary_files = mk_file_list(trace_summary_dir, 'summary.csv', absolute=True)

    for l, param_name in enumerate(spectral_param_names):

        progress = 100.0 * (l + 1.0) / len(spectral_param_names)
        print('Percentage through spectral parameters = {0}% (param: {1})'.format(progress, param_name))

        rhat_map = np.ones(no_pix) * hp.UNSEEN
        neff_map = np.ones(no_pix) * hp.UNSEEN
        mu_map = np.ones(no_pix) * hp.UNSEEN
        sigma_map = np.ones(no_pix) * hp.UNSEEN

        for i, num in enumerate(fitted_regions):
        
            df = pd.read_csv(trace_summary_files[i], index_col=0)
            region_pix = np.where(regions == num)[0]

            for k, pix in enumerate(region_pix):
                rhat_map[pix] = df['Rhat']['{0}__{1}'.format(param_name, int(k))]
                neff_map[pix] = df['n_eff']['{0}__{1}'.format(param_name, int(k))]
                mu_map[pix] = df['mean']['{0}__{1}'.format(param_name, int(k))]
                sigma_map[pix] = df['sd']['{0}__{1}'.format(param_name, int(k))]

        hp.write_map('{}{}_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), rhat_map, overwrite=True)
        hp.write_map('{}{}_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), neff_map, overwrite=True)
        hp.write_map('{}{}_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), mu_map, overwrite=True)
        hp.write_map('{}{}_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), sigma_map, overwrite=True)

    for l, param_name in enumerate(hyper_param_names):

        progress = 100.0 * (l + 1.0) / len(hyper_param_names)
        print('Percentage through hyper-parameters = {0}% (param: {1})'.format(progress, param_name))

        rhat_map = np.ones(no_pix) * hp.UNSEEN
        neff_map = np.ones(no_pix) * hp.UNSEEN
        mu_map = np.ones(no_pix) * hp.UNSEEN
        sigma_map = np.ones(no_pix) * hp.UNSEEN

        for i, num in enumerate(fitted_regions):

            df = pd.read_csv(trace_summary_files[i], index_col=0)
            region_pix = np.where(regions == num)[0]

            rhat_map[region_pix] = df['Rhat'][param_name]
            neff_map[region_pix] = df['n_eff'][param_name]
            mu_map[region_pix] = df['mean'][param_name]
            sigma_map[region_pix] = df['sd'][param_name]

        hp.write_map('{}{}_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), rhat_map, overwrite=True)
        hp.write_map('{}{}_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), neff_map, overwrite=True)
        hp.write_map('{}{}_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), mu_map, overwrite=True)
        hp.write_map('{}{}_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), sigma_map, overwrite=True)

    for l, param_name in enumerate(amp_param_names):

        progress = 100.0 * (l + 1.0) / len(amp_param_names)
        print('Percentage through amplitude parameters = {0}% (param: {1})'.format(progress, param_name))

        if joint_QU:
            Q_rhat_map = np.ones(no_pix) * hp.UNSEEN
            Q_neff_map = np.ones(no_pix) * hp.UNSEEN
            Q_mu_map = np.ones(no_pix) * hp.UNSEEN
            Q_sigma_map = np.ones(no_pix) * hp.UNSEEN
            U_rhat_map = np.ones(no_pix) * hp.UNSEEN
            U_neff_map = np.ones(no_pix) * hp.UNSEEN
            U_mu_map = np.ones(no_pix) * hp.UNSEEN
            U_sigma_map = np.ones(no_pix) * hp.UNSEEN
        elif not joint_QU:
            rhat_map = np.ones(no_pix) * hp.UNSEEN
            neff_map = np.ones(no_pix) * hp.UNSEEN
            mu_map = np.ones(no_pix) * hp.UNSEEN
            sigma_map = np.ones(no_pix) * hp.UNSEEN

        for i, num in enumerate(fitted_regions):

            df = pd.read_csv(trace_summary_files[i], index_col=0)
            region_pix = np.where(regions == num)[0]

            if joint_QU:
                for k, pix in enumerate(region_pix):
                    Q_rhat_map[pix] = df['Rhat']['{0}__{1}'.format(param_name, int(k))]
                    Q_neff_map[pix] = df['n_eff']['{0}__{1}'.format(param_name, int(k))]
                    Q_mu_map[pix] = df['mean']['{0}__{1}'.format(param_name, int(k))]
                    Q_sigma_map[pix] = df['sd']['{0}__{1}'.format(param_name, int(k))]
                    U_rhat_map[pix] = df['Rhat']['{0}__{1}'.format(param_name, int(k + len(region_pix)))]
                    U_neff_map[pix] = df['n_eff']['{0}__{1}'.format(param_name, int(k + len(region_pix)))]
                    U_mu_map[pix] = df['mean']['{0}__{1}'.format(param_name, int(k + len(region_pix)))]
                    U_sigma_map[pix] = df['sd']['{0}__{1}'.format(param_name, int(k + len(region_pix)))]
            elif not joint_QU:
                for k, pix in enumerate(region_pix):
                    rhat_map[pix] = df['Rhat']['{0}__{1}'.format(param_name, int(k))]
                    neff_map[pix] = df['n_eff']['{0}__{1}'.format(param_name, int(k))]
                    mu_map[pix] = df['mean']['{0}__{1}'.format(param_name, int(k))]
                    sigma_map[pix] = df['sd']['{0}__{1}'.format(param_name, int(k))]

        if joint_QU:

            hp.write_map('{}{}_Q_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), Q_rhat_map, overwrite=True)
            hp.write_map('{}{}_Q_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), Q_neff_map, overwrite=True)
            hp.write_map('{}{}_Q_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), Q_mu_map, overwrite=True)
            hp.write_map('{}{}_Q_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), Q_sigma_map,
                         overwrite=True)
            hp.write_map('{}{}_U_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), U_rhat_map, overwrite=True)
            hp.write_map('{}{}_U_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), U_neff_map, overwrite=True)
            hp.write_map('{}{}_U_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), U_mu_map, overwrite=True)
            hp.write_map('{}{}_U_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), U_sigma_map,
                         overwrite=True)

        elif not joint_QU:

            hp.write_map('{}{}_{}_rhat_map.fits'.format(out_dir, out_prefix, param_name), rhat_map, overwrite=True)
            hp.write_map('{}{}_{}_neff_map.fits'.format(out_dir, out_prefix, param_name), neff_map, overwrite=True)
            hp.write_map('{}{}_{}_mu_map.fits'.format(out_dir, out_prefix, param_name), mu_map, overwrite=True)
            hp.write_map('{}{}_{}_sigma_map.fits'.format(out_dir, out_prefix, param_name), sigma_map, overwrite=True)


def mk_cmb_posterior_samples(trace_summary_dir, region_file, mask_file, out_dir, out_prefix, num_samples,
                             joint_QU=False, pix_thresh=1, thin=1):

    """

    Function for generating parameter maps from pixel traces.

    Inputs
    ------
    trace_summary_dir: The directory where trace summaries are stored - str.
    mask_file: Mask file used in component separation - str.
    out_dir: The output directory for the CMB sample maps - str.
    out_prefix: The output prefix for the parameter maps - str.
    num_samples: Number of CMB amplitude posterior samples we drew - int.
    joint_QU: Whether fitting was done jointly on Q and U, assuming common spectral parameters - boolean.
    pix_thresh: Number of pixels at which we allow fitting - str.

    Returns
    -------
    cmb_maps: The CMB maps corresponding to individual CMB amplitude posterior samples - numpy.ndarray.

    Saves the maps to the output directory.

    """

    regions = hp.read_map(region_file)
    region_nums = [i for i in range(1, int(np.amax(regions)) + 1)]

    mask = hp.read_map(mask_file)
    no_pix = len(mask)

    fitted_regions = []
    for num in region_nums:
        mask_pix = mask[regions == num]
        if len(mask_pix[mask_pix == 1]) >= pix_thresh:
            fitted_regions.append(num)

    if joint_QU:
        Q_cmb_maps = []
        U_cmb_maps = []
    elif not joint_QU:
        cmb_maps = []

    trace_summary_files = mk_file_list(trace_summary_dir, 'cmb_posterior_samples.csv', absolute=True)

    mpi_on = 1
    if mpi_on == 1:
        from mpi4py.MPI import COMM_WORLD as world
        rank = world.Get_rank()
        size = world.Get_size()
    else:
        rank = 0
        size = 1

    samples = np.arange(0, num_samples, thin)
        
    for j, sample in enumerate(samples):

        if j%size!=rank:
            continue
        print('Task {} being done by processor {} of {}'.format(j + 1, rank + 1, size))
        progress = 100.0 * (j + 1.0) / len(samples)
        print('Percentage through constructing maps = {0}% (sample {1})'.format(progress, int(sample)))
        sys.stdout.flush()

        if joint_QU:
            Q_cmb_map = np.ones(no_pix) * hp.UNSEEN
            U_cmb_map = np.ones(no_pix) * hp.UNSEEN
        elif not joint_QU:
            cmb_map = np.ones(no_pix) * hp.UNSEEN

        for i, num in enumerate(fitted_regions):

            df = pd.read_csv(trace_summary_files[i], usecols=['MC__{0}'.format(int(sample))])
            region_pix = np.where(regions == num)[0]
            for k, pix in enumerate(region_pix):
                if joint_QU:
                    Q_cmb_map[pix] = df['MC__{0}'.format(int(sample))][int(k)]
                    U_cmb_map[pix] = df['MC__{0}'.format(int(sample))][int(k + len(region_pix))]
                elif not joint_QU:
                    cmb_map[pix] = df['MC__{0}'.format(int(sample))][int(k)]

        if joint_QU:
            Q_cmb_maps.append(Q_cmb_map)
            U_cmb_maps.append(U_cmb_map)
            hp.write_map('{0}Q_maps/{1}_Q_cmb_posterior_sample_{2}.fits'.format(out_dir, out_prefix, int(sample)), Q_cmb_map,
                         overwrite=True)
            hp.write_map('{0}U_maps/{1}_U_cmb_posterior_sample_{2}.fits'.format(out_dir, out_prefix, int(sample)), U_cmb_map,
                         overwrite=True)

        elif not joint_QU:
            cmb_maps.append(cmb_map)
            hp.write_map('{0}{1}_cmb_posterior_sample_{2}.fits'.format(out_dir, out_prefix, int(sample)), cmb_map,
                         overwrite=True)

    if joint_QU:
        return np.array(Q_cmb_maps), np.array(U_cmb_maps)
    elif not joint_QU:
        return np.array(cmb_maps)
