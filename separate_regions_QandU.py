import pymc3 as pm
import pandas as pd
import healpy as hp
import configparser
import argparse
import json
from sample import *
from custom_priors import *
from utils import class_import
import os


"""

This is a generic wrapper for running the component separation code. Variants on the example .ini file 
will produce a component separation run. To call the script you want to do:

python separate_regions_QandU.py {ini_name}.ini {region_number}

The choice of passing the region_number separately has been chosen for easy parallelisation on the Glamdring cluster!
You can generate a multirun file for this using the mk_params_file function in utils.py.

"""

parser = argparse.ArgumentParser(description='Bayesian component separation using NUTS')
parser.add_argument('param_file', help='Main parameters file.')
parser.add_argument('region_num', type=int, help='Region number')
region_num = parser.parse_args().region_num

Config = configparser.ConfigParser()
Config.read(parser.parse_args().param_file)

model_class_name = Config.get('Input Params', 'model_class_name')
model_name = Config.get('Input Params', 'model_name')
nu = np.asarray(json.loads(Config.get('Input Params', 'frequencies')), dtype='float')
map_files = json.loads(Config.get('Input Params', 'map_files'))
cov_files = json.loads(Config.get('Input Params', 'cov_files'))
mask = hp.read_map(Config.get('Input Params', 'mask_file')).astype(int)
fwhm = Config.getfloat('Input Params', 'fwhm')
smooth_init = Config.getboolean('Input Params', 'smooth_init')
regions = hp.read_map(Config.get('Input Params', 'regions_file')).astype(int)
synch_nu0 = Config.getfloat('Input Params', 'synch_nu0')
dust_nu0 = Config.getfloat('Input Params', 'dust_nu0')
ame_nu0 = Config.getfloat('Input Params', 'ame_nu0')

num_burn = Config.getint('Output Params', 'num_burn')
num_thin = Config.getint('Output Params', 'num_thin')
save_trace = Config.getboolean('Output Params', 'save_trace')
trace_out_dir = Config.get('Output Params', 'trace_out_dir')
trace_summary_dir = Config.get('Output Params', 'trace_summary_dir')
trace_out_prefix = Config.get('Output Params', 'trace_out_prefix')

nsamples = Config.getint('Sampling Params', 'nsamples')
ntune = Config.getint('Sampling Params', 'ntune')
accept_prob = Config.getfloat('Sampling Params', 'accept_prob')
advi = Config.getboolean('Sampling Params', 'advi')
progressbar = Config.getboolean('Sampling Params', 'progressbar')
discard_tuned_samples = Config.getboolean('Sampling Params', 'discard_tuned_samples')

# Note, all input maps should be in uK_RJ (where appropriate)!
# Initial spectral parameters are taken to be the same for Q and U.
if smooth_init:
    init_synch_amp_Q = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_synch_amp'), field=1),
                                    np.radians(fwhm))
    init_synch_amp_U = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_synch_amp'), field=2),
                                    np.radians(fwhm))
    init_synch_beta = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_synch_beta')), np.radians(fwhm))
    init_synch_curve = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_synch_curve')), np.radians(fwhm))
    init_dust_amp_Q = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_dust_amp'), field=1),
                                   np.radians(fwhm))
    init_dust_amp_U = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_dust_amp'), field=2),
                                   np.radians(fwhm))
    init_dust_beta = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_dust_beta')), np.radians(fwhm))
    init_dust_T = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_dust_T')), np.radians(fwhm))
    init_ame_amp_Q = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_ame_amp'), field=1),
                                  np.radians(fwhm))
    init_ame_amp_U = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_ame_amp'), field=2),
                                  np.radians(fwhm))
    init_ame_nup = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_ame_nup')), np.radians(fwhm))
    init_cmb_amp_Q = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_cmb_amp'), field=1),
                                  np.radians(fwhm))
    init_cmb_amp_U = hp.smoothing(hp.read_map(Config.get('Starting Values', 'init_cmb_amp'), field=2),
                                  np.radians(fwhm))

elif not smooth_init:
    init_synch_amp_Q = hp.read_map(Config.get('Starting Values', 'init_synch_amp'), field=1)
    init_synch_amp_U = hp.read_map(Config.get('Starting Values', 'init_synch_amp'), field=2)
    init_synch_beta = hp.read_map(Config.get('Starting Values', 'init_synch_beta'))
    init_synch_curve = hp.read_map(Config.get('Starting Values', 'init_synch_curve'))
    init_dust_amp_Q = hp.read_map(Config.get('Starting Values', 'init_dust_amp'), field=1)
    init_dust_amp_U = hp.read_map(Config.get('Starting Values', 'init_dust_amp'), field=2)
    init_dust_beta = hp.read_map(Config.get('Starting Values', 'init_dust_beta'))
    init_dust_T = hp.read_map(Config.get('Starting Values', 'init_dust_T'))
    init_ame_amp_Q = hp.read_map(Config.get('Starting Values', 'init_ame_amp'), field=1)
    init_ame_amp_U = hp.read_map(Config.get('Starting Values', 'init_ame_amp'), field=2)
    init_ame_nup = hp.read_map(Config.get('Starting Values', 'init_ame_nup'))
    init_cmb_amp_Q = hp.read_map(Config.get('Starting Values', 'init_cmb_amp'), field=1)
    init_cmb_amp_U = hp.read_map(Config.get('Starting Values', 'init_cmb_amp'), field=2)

jfg_synch_beta_ref = Config.getfloat('Priors', 'jfg_synch_beta_ref')
jfg_synch_beta_sd = Config.getfloat('Priors', 'jfg_synch_beta_sd')
jfg_synch_curve_ref = Config.getfloat('Priors', 'jfg_synch_curve_ref')
jfg_synch_curve_sd = Config.getfloat('Priors', 'jfg_synch_curve_sd')
jfg_dust_beta_ref = Config.getfloat('Priors', 'jfg_dust_beta_ref')
jfg_dust_beta_sd = Config.getfloat('Priors', 'jfg_dust_beta_sd')
jfg_dust_T_ref = Config.getfloat('Priors', 'jfg_dust_T_ref')
jfg_dust_T_sd = Config.getfloat('Priors', 'jfg_dust_T_sd')
jfg_ame_peak_nu_ref = Config.getfloat('Priors', 'jfg_ame_peak_nu_ref')
jfg_ame_peak_nu_sd = Config.getfloat('Priors', 'jfg_ame_peak_nu_sd')

hrl_synch_beta_ref = Config.getfloat('Priors', 'hrl_synch_beta_ref')
hrl_synch_beta_sd = Config.getfloat('Priors', 'hrl_synch_beta_sd')
hrl_synch_beta_sigma_ref = Config.getfloat('Priors', 'hrl_synch_beta_sigma_ref')
hrl_synch_curve_ref = Config.getfloat('Priors', 'hrl_synch_curve_ref')
hrl_synch_curve_sd = Config.getfloat('Priors', 'hrl_synch_curve_sd')
hrl_synch_curve_sigma_ref = Config.getfloat('Priors', 'hrl_synch_curve_sigma_ref')
hrl_dust_beta_ref = Config.getfloat('Priors', 'hrl_dust_beta_ref')
hrl_dust_beta_sd = Config.getfloat('Priors', 'hrl_dust_beta_sd')
hrl_dust_beta_sigma_ref = Config.getfloat('Priors', 'hrl_dust_beta_sigma_ref')
hrl_dust_T_ref = Config.getfloat('Priors', 'hrl_dust_T_ref')
hrl_dust_T_sd =	Config.getfloat('Priors', 'hrl_dust_T_sd')
hrl_dust_T_sigma_ref = Config.getfloat('Priors', 'hrl_dust_T_sigma_ref')
hrl_ame_peak_nu_ref = Config.getfloat('Priors', 'hrl_ame_peak_nu_ref')
hrl_ame_peak_nu_sd = Config.getfloat('Priors', 'hrl_ame_peak_nu_sd')
hrl_ame_peak_nu_sigma_ref = Config.getfloat('Priors', 'hrl_ame_peak_nu_sigma_ref')
    
Q_maps = np.array([hp.read_map(m, field=1) for m in map_files])
for f in map_files:
    print(hp.get_nside(hp.read_map(f, field=1)))
Q_covs = np.array([hp.read_map(c, field=1) for c in cov_files])
U_maps = np.array([hp.read_map(m, field=2) for m in map_files])
U_covs = np.array([hp.read_map(c, field=2) for c in cov_files])

region_idx = np.where(regions == region_num)[0]
Q_region_map_vals = Q_maps[:, region_idx]
U_region_map_vals = U_maps[:, region_idx]
Q_region_sig_vals = np.sqrt(Q_covs[:, region_idx])
U_region_sig_vals = np.sqrt(U_covs[:, region_idx])
region_map_vals = np.concatenate((Q_region_map_vals, U_region_map_vals), axis=1)
region_sig_vals = np.concatenate((Q_region_sig_vals, U_region_sig_vals), axis=1)
model_class = class_import(model_class_name)

if model_class_name == 'pol_sky_model.Pol_JeffGauss_Models':

    init_synch_amp = np.concatenate((init_synch_amp_Q[region_idx], init_synch_amp_U[region_idx]))
    init_dust_amp = np.concatenate((init_dust_amp_Q[region_idx], init_dust_amp_U[region_idx]))
    init_ame_amp = np.concatenate((init_ame_amp_Q[region_idx], init_ame_amp_U[region_idx]))
    init_cmb_amp = np.concatenate((init_cmb_amp_Q[region_idx], init_cmb_amp_U[region_idx]))
    
    synch_amp_shift = np.concatenate((np.mean(init_synch_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                      np.mean(init_synch_amp_U[region_idx]) * np.ones(len(region_idx))))
    synch_amp_spread = np.concatenate((np.std(init_synch_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                       np.std(init_synch_amp_U[region_idx]) * np.ones(len(region_idx))))
    synch_beta_shift = np.mean(init_synch_beta[region_idx])
    synch_beta_spread = np.std(init_synch_beta[region_idx])
    synch_curve_shift = np.mean(init_synch_curve[region_idx])
    synch_curve_spread = np.std(init_synch_curve[region_idx])

    dust_amp_shift = np.concatenate((np.mean(init_dust_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                      np.mean(init_dust_amp_U[region_idx]) * np.ones(len(region_idx))))
    dust_amp_spread = np.concatenate((np.std(init_dust_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                       np.std(init_dust_amp_U[region_idx]) * np.ones(len(region_idx))))
    dust_beta_shift = np.mean(init_dust_beta[region_idx])
    dust_beta_spread = np.std(init_dust_beta[region_idx])
    dust_T_shift = np.mean(init_dust_T[region_idx])
    dust_T_spread = np.std(init_dust_T[region_idx])

    cmb_amp_shift = np.concatenate((np.mean(init_cmb_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                    np.mean(init_cmb_amp_U[region_idx]) * np.ones(len(region_idx))))
    cmb_amp_spread = np.concatenate((np.std(init_cmb_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                     np.std(init_cmb_amp_U[region_idx]) * np.ones(len(region_idx))))

    init_synch_amp_norm = (init_synch_amp - synch_amp_shift) / synch_amp_spread
    init_synch_beta_norm = (init_synch_beta[region_idx] - synch_beta_shift) / synch_beta_spread
    init_synch_curve_norm = (init_synch_curve[region_idx] - synch_curve_shift) / synch_curve_spread

    init_dust_amp_norm = (init_dust_amp - dust_amp_shift) / dust_amp_spread
    init_dust_beta_norm = (init_dust_beta[region_idx] - dust_beta_shift) / dust_beta_spread
    init_dust_T_norm = (init_dust_T[region_idx] - dust_T_shift) / dust_T_spread

    init_cmb_amp_norm = (init_cmb_amp - cmb_amp_shift) / cmb_amp_spread
    
    start_dict = {'synch_amp_norm': init_synch_amp_norm, 'synch_beta_norm': init_synch_beta_norm,
                  'synch_curve_norm': init_synch_curve_norm, 'dust_amp_norm': init_dust_amp_norm,
                  'dust_beta_norm': init_dust_beta_norm, 'dust_T_norm': init_dust_T_norm,
                  'cmb_amp_norm': init_cmb_amp_norm,
                  'synch_amp': init_synch_amp, 'synch_beta': np.mean(init_synch_beta[region_idx]),
                  'synch_curve': np.mean(init_synch_curve[region_idx]), 'dust_amp': init_dust_amp,
                  'dust_beta': np.mean(init_dust_beta[region_idx]), 'dust_T': np.mean(init_dust_T[region_idx]),
                  'ame_amp': init_ame_amp, 'ame_peak_nu': np.mean(init_ame_nup[region_idx]),
                  'cmb_amp': init_cmb_amp}
    
    region_model = model_class(model_name, nu, region_map_vals, region_sig_vals,
                               synch_nu0=synch_nu0, dust_nu0=dust_nu0, ame_nu0=ame_nu0,
                               synch_amp_shift=synch_amp_shift, synch_amp_spread=synch_amp_spread,
                               synch_beta_ref=jfg_synch_beta_ref, synch_beta_sd=jfg_synch_beta_sd,
                               synch_beta_shift=synch_beta_shift, synch_beta_spread=synch_beta_spread,
                               synch_curve_ref=jfg_synch_curve_ref, synch_curve_sd=jfg_synch_curve_sd,
                               synch_curve_shift=synch_curve_shift, synch_curve_spread=synch_curve_spread,
                               dust_amp_shift=dust_amp_shift, dust_amp_spread=dust_amp_spread,
                               dust_beta_ref=jfg_dust_beta_ref, dust_beta_sd=jfg_dust_beta_sd,
                               dust_beta_shift=dust_beta_shift, dust_beta_spread=dust_beta_spread,
                               dust_T_ref=jfg_dust_T_ref, dust_T_sd=jfg_dust_T_sd,
                               dust_T_shift=dust_T_shift, dust_T_spread=dust_T_spread,
                               ame_peak_nu_ref=jfg_ame_peak_nu_ref, ame_peak_nu_sd=jfg_ame_peak_nu_sd,
                               cmb_amp_shift=cmb_amp_shift, cmb_amp_spread=cmb_amp_spread).get_model()

if model_class_name == 'pol_sky_model.Hierarchical_Pol_Models':

    init_synch_amp = np.concatenate((init_synch_amp_Q[region_idx], init_synch_amp_U[region_idx]))
    init_dust_amp = np.concatenate((init_dust_amp_Q[region_idx], init_dust_amp_U[region_idx]))
    init_ame_amp = np.concatenate((init_ame_amp_Q[region_idx], init_ame_amp_U[region_idx]))
    init_cmb_amp = np.concatenate((init_cmb_amp_Q[region_idx], init_cmb_amp_U[region_idx]))
    
    synch_amp_shift = np.concatenate((np.mean(init_synch_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                      np.mean(init_synch_amp_U[region_idx]) * np.ones(len(region_idx))))
    synch_amp_spread = np.concatenate((np.std(init_synch_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                       np.std(init_synch_amp_U[region_idx]) * np.ones(len(region_idx))))
    synch_beta_shift =	np.mean(init_synch_beta[region_idx])
    synch_beta_spread =	np.std(init_synch_beta[region_idx])
    synch_curve_shift =	np.mean(init_synch_curve[region_idx])
    synch_curve_spread = np.std(init_synch_curve[region_idx])

    dust_amp_shift = np.concatenate((np.mean(init_dust_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                      np.mean(init_dust_amp_U[region_idx]) * np.ones(len(region_idx))))
    dust_amp_spread = np.concatenate((np.std(init_dust_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                       np.std(init_dust_amp_U[region_idx]) * np.ones(len(region_idx))))
    dust_beta_shift = np.mean(init_dust_beta[region_idx])
    dust_beta_spread = np.std(init_dust_beta[region_idx])
    dust_T_shift = np.mean(init_dust_T[region_idx])
    dust_T_spread = np.std(init_dust_T[region_idx])

    cmb_amp_shift = np.concatenate((np.mean(init_cmb_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                    np.mean(init_cmb_amp_U[region_idx]) * np.ones(len(region_idx))))
    cmb_amp_spread = np.concatenate((np.std(init_cmb_amp_Q[region_idx]) * np.ones(len(region_idx)),
                                     np.std(init_cmb_amp_U[region_idx]) * np.ones(len(region_idx))))

    init_synch_amp_norm = (init_synch_amp - synch_amp_shift) / synch_amp_spread
    init_synch_beta_mu_norm = (hrl_synch_beta_ref - synch_beta_shift) / synch_beta_spread
    init_synch_beta_sigma_norm = hrl_synch_beta_sigma_ref / synch_beta_spread
    init_synch_curve_mu_norm = (hrl_synch_curve_ref - synch_curve_shift) / synch_curve_spread
    init_synch_curve_sigma_norm = hrl_synch_curve_sigma_ref / synch_curve_spread
    
    init_dust_amp_norm = (init_dust_amp - dust_amp_shift) / dust_amp_spread
    init_dust_beta_mu_norm = (hrl_dust_beta_ref - dust_beta_shift) / dust_beta_spread
    init_dust_beta_sigma_norm = hrl_dust_beta_sigma_ref / dust_beta_spread
    init_dust_T_mu_norm = (hrl_dust_T_ref - dust_T_shift) / dust_T_spread
    init_dust_T_sigma_norm = hrl_dust_T_sigma_ref / dust_T_spread
    
    init_cmb_amp_norm =	(init_cmb_amp - cmb_amp_shift) / cmb_amp_spread

    start_dict = {'synch_amp_norm': init_synch_amp_norm, 'synch_beta_mu_norm': init_synch_beta_mu_norm,
                  'synch_beta_sigma_norm': init_synch_beta_sigma_norm, 'synch_curve_mu_norm': init_synch_curve_mu_norm,
                  'synch_curve_sigma_norm': init_synch_curve_sigma_norm, 'dust_amp_norm': init_dust_amp_norm,
                  'dust_beta_mu_norm': init_dust_beta_mu_norm, 'dust_beta_sigma_norm': init_dust_beta_sigma_norm,
                  'dust_T_mu_norm': init_dust_T_mu_norm, 'dust_T_sigma_norm': init_dust_T_sigma_norm,
                  'cmb_amp_norm': init_cmb_amp_norm,
                  'synch_beta_mu': hrl_synch_beta_ref, 'synch_beta_sigma': hrl_synch_beta_sigma_ref,
                  'dust_beta_mu': hrl_dust_beta_ref, 'dust_beta_sigma': hrl_dust_beta_sigma_ref,
                  'dust_T_mu': hrl_dust_T_ref, 'dust_T_sigma': hrl_dust_T_sigma_ref,
                  'synch_amp': init_synch_amp, 'synch_beta': init_synch_beta[region_idx],
                  'synch_curve': init_synch_curve[region_idx], 'dust_amp': init_dust_amp,
                  'dust_beta': init_dust_beta[region_idx], 'dust_T': init_dust_T[region_idx],
                  'ame_amp': init_ame_amp, 'ame_peak_nu': init_ame_nup[region_idx],
                  'cmb_amp': init_cmb_amp}
    
    region_model = model_class(model_name, nu, region_map_vals, region_sig_vals,
                               synch_nu0=synch_nu0, dust_nu0=dust_nu0, ame_nu0=ame_nu0,
                               synch_amp_shift=synch_amp_shift, synch_amp_spread=synch_amp_spread,
                               synch_beta_ref=hrl_synch_beta_ref, synch_beta_sd=hrl_synch_beta_sd,
                               synch_beta_sigma_ref=hrl_synch_beta_sigma_ref, synch_beta_shift=synch_beta_shift,
                               synch_beta_spread=synch_beta_spread, synch_curve_ref=hrl_synch_curve_ref,
                               synch_curve_sd=hrl_synch_curve_sd, synch_curve_sigma_ref=hrl_synch_curve_sigma_ref,
                               synch_curve_shift=synch_curve_shift, synch_curve_spread=synch_curve_spread,
                               dust_amp_shift=dust_amp_shift, dust_amp_spread=dust_amp_spread,
                               dust_beta_ref=hrl_dust_beta_ref, dust_beta_sd=hrl_dust_beta_sd,
                               dust_beta_sigma_ref=hrl_dust_beta_sigma_ref, dust_beta_shift=dust_beta_shift,
                               dust_beta_spread=dust_beta_spread, dust_T_ref=hrl_dust_T_ref, dust_T_sd=hrl_dust_T_sd,
                               dust_T_sigma_ref=hrl_dust_T_sigma_ref, dust_T_shift=dust_T_shift,
                               dust_T_spread=dust_T_spread, ame_peak_nu_ref=hrl_ame_peak_nu_ref,
                               ame_peak_nu_sd=hrl_ame_peak_nu_sd, ame_peak_nu_sigma_ref=hrl_ame_peak_nu_sigma_ref,
                               cmb_amp_shift=cmb_amp_shift, cmb_amp_spread=cmb_amp_spread).get_model()

print('Sampling region number {}'.format(region_num))

trace_out_prefix_region = trace_out_prefix + '_regionNo{}'.format(region_num)
trace_out_dir = trace_out_dir.format(region_num)
if not os.path.isdir(trace_out_dir):
    os.mkdir(trace_out_dir)

try:
    
    if advi:
        trace = sample_model(region_model, nsamples, ntune, accept_prob, trace_out_dir=trace_out_dir,
                             progressbar=progressbar, discard_tuned_samples=discard_tuned_samples,
                             save_trace=save_trace, metropolis=False)
    elif not advi:
        trace = sample_model(region_model, nsamples, ntune, accept_prob, trace_out_dir=trace_out_dir, start=start_dict,
                             progressbar=progressbar, discard_tuned_samples=discard_tuned_samples,
                             save_trace=save_trace, metropolis=False)

    cmb_samples = np.transpose(trace.get_values(varname='cmb_amp', burn=num_burn, thin=num_thin))
    df_index = ['cmb_amp__{}'.format(int(i)) for i in range(np.shape(cmb_samples)[0])]
    df_columns = ['MC__{}'.format(int(i)) for i in range(np.shape(cmb_samples)[1])]
    df = pd.DataFrame(data=cmb_samples, index=df_index, columns=df_columns)
    df.to_csv('{}{}_cmb_posterior_samples.csv'.format(trace_summary_dir, trace_out_prefix_region))

    df = pm.summary(trace)
    df.to_csv('{}{}_summary.csv'.format(trace_summary_dir, trace_out_prefix_region))

except:

    print('Sampler failed for region {}. No output being saved for this region.'.format(region_num))

    
    

