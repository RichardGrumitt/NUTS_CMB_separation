import pandas as pd
import numpy as np
import healpy as hp
import glob
from natsort import natsorted


def construct_stat_maps(df_directory, mask_file, out_directory, out_prefix, n_eff_thresh=200, rhat_thresh=1.2):

    """

    Inputs
    ------
    :df_directory: Directory containing the csv files for the PyMC3 trace summaries - str.
    :mask_file: Mask file used in component separation - str.
    :out_directory: Output directory - str.
    :out_prefix: Output map file prefix - str.
    :n_eff_thresh: Minimum number of effective samples needed to include pixel - float.
    :rhat_thresh: Maximum Gelman-Rubin value allowed to include pixel - float.

    Returns
    -------

    """

    files = natsorted(glob.glob('{}*.csv'.format(df_directory)))
    mask = hp.read_map(mask_file)
    good_idx = np.where(mask == 1)[0]

    df0 = pd.read_csv(files[0])
    num_rows = df0.shape[0]
    mean_maps = hp.UNSEEN*np.ones((num_rows, len(mask)))
    sd_maps = hp.UNSEEN*np.ones((num_rows, len(mask)))
    neff_maps = hp.UNSEEN*np.ones((num_rows, len(mask)))
    rhat_maps = hp.UNSEEN*np.ones((num_rows, len(mask)))

    for i, f in enumerate(files):

        print('Gridding file {}/{}'.format(i + 1, len(files)))
        df = pd.read_csv(f)

        for j in range(num_rows):

            if df['n_eff'][j] >= n_eff_thresh and df['Rhat'][j] <= rhat_thresh:

                mean_maps[j, good_idx[i]] = df['mean'][j]
                sd_maps[j, good_idx[i]] = df['sd'][j]
                neff_maps[j, good_idx[i]] = df['n_eff'][j]
                rhat_maps[j, good_idx[i]] = df['Rhat'][j]

    print('Saving the summary statistic maps ...')

    for k in range(num_rows):

        hp.write_map('{}{}_mean_{}.fits'.format(out_directory, out_prefix, df['Unnamed: 0'][k]), mean_maps[k, :],
                     overwrite=True)
        hp.write_map('{}{}_sd_{}.fits'.format(out_directory, out_prefix, df['Unnamed: 0'][k]), sd_maps[k, :],
                     overwrite=True)
        hp.write_map('{}{}_neff_{}.fits'.format(out_directory, out_prefix, df['Unnamed: 0'][k]), neff_maps[k, :],
                     overwrite=True)
        hp.write_map('{}{}_rhat_{}.fits'.format(out_directory, out_prefix, df['Unnamed: 0'][k]), rhat_maps[k, :],
                     overwrite=True)

