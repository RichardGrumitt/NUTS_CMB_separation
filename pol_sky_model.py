import pymc3 as pm
from diffuse_component_models import *
from custom_priors import *
import numpy as np


class Pol_JeffGauss_Models(object):

    """
    
    Model class for polarised emission models employing Jeffreys priors, with spectral parameters multiplied by Gaussian
    profiles (no hard limits).

    These models all fit for constant spectral parameters over the given fitting region.

    Attributes
    ----------
    pol_model: The name of the polarised emission model - str.
    nu: Frequencies of the sky maps we are 'component separating' - numpy.ndarray.
    sky_vals: Values of the sky pixel being studied, at each frequency - numpy.ndarray.
    sky_sigma: Associated one sigma uncertainties on sky pixel values - numpy.ndarray.
    synch_nu0: Synchrotron SED reference frequency - float.
    dust_nu0: Dust SED reference frequency - float.
    ame_nu0: AME SED reference frequency - float.
    synch_amp_shift: Synchrotron amplitude shift term - float.
    synch_amp_spread: Synchrotron amplitude scaling term - float.
    synch_beta_ref: Reference synchrotron spectral index for the Gaussian profile - float.
    synch_beta_sd: Standard deviation of the synchrotron spectral index Gaussian profile - float.
    synch_beta_shift: Synchrotron beta shift term - float.
    synch_beta_spread: Synchrotron beta scaling term - float.
    synch_curve_ref: Reference synchrotron curvature for the Gaussian profile - float.
    synch_curve_sd: Standard deviation of the synchrotron curvature Gaussian profile - float.
    synch_curve_shift: Synchrotron curvature shift term - float.
    synch_curve_spread: Synchrotron curvature scaling term - float.
    dust_beta_ref: Reference dust spectral index for the Gaussian profile - float.
    dust_beta_sd: Standard deviation of the dust spectral index Gaussian profile - float.
    dust_beta_shift: Dust beta shift term - float.
    dust_beta_spread: Dust beta scaling term - float.
    dust_T_ref: Reference dust temperature for the Gaussian profile - float.
    dust_T_sd: Standard deviation for the dust temeprature Gaussian profile - float.
    dust_T_shift: Dust temperature shift term - float.
    dust_T_spread: Dust temperature scaling term - float.
    ame_peak_nu_ref: Reference AME peak frequency for the Gaussian profile - float.
    ame_peak_nu_sd: Standard deviation of the AME peak frequency Gaussian profile - float.
    ame_peak_nu_shift: AME peak nu shift term - float.
    ame_peak_nu_spread: AME peak nu scaling term - float.

    """

    def __init__(self, pol_model, nu, sky_vals, sky_sigma, synch_nu0=5.0, dust_nu0=353.0, ame_nu0=22.8,
                 synch_amp_shift=100.0, synch_amp_spread=100.0, synch_beta_ref=-3.0, synch_beta_sd=0.1,
                 synch_beta_shift=-3.0, synch_beta_spread=0.1, synch_curve_ref=0.0, synch_curve_sd=0.1,
                 synch_curve_shift=0.0, synch_curve_spread=0.1, dust_amp_shift=100.0, dust_amp_spread=100.0,
                 dust_beta_ref=1.6, dust_beta_sd=0.3, dust_beta_shift=1.6, dust_beta_spread=0.3,
                 dust_T_ref=19.4, dust_T_sd=1.5, dust_T_shift=19.4, dust_T_spread=1.5,
                 ame_amp_shift=100.0, ame_amp_spread=100.0, ame_peak_nu_ref=22.8, ame_peak_nu_sd=1.0,
                 ame_peak_nu_shift=22.8, ame_peak_nu_spread=1.0, cmb_amp_shift=0.0, cmb_amp_spread=1.0):

        self.pol_model = pol_model
        self.nu = nu
        self.nu_matrix = np.transpose(np.broadcast_to(nu, (np.shape(sky_vals)[1], len(nu))))
        self.sky_vals = sky_vals
        self.sky_sigma = sky_sigma
        self.jeff_sigma = np.sqrt(np.sum(sky_sigma ** 2, axis=1)) / len(sky_sigma)
        self.synch_nu0 = synch_nu0
        self.dust_nu0 = dust_nu0
        self.ame_nu0 = ame_nu0
        self.synch_amp_shift = synch_amp_shift
        self.synch_amp_spread = synch_amp_spread
        self.synch_beta_ref = synch_beta_ref
        self.synch_beta_sd = synch_beta_sd
        self.synch_beta_shift = synch_beta_shift
        self.synch_beta_spread = synch_beta_spread
        self.synch_curve_ref = synch_curve_ref
        self.synch_curve_sd = synch_curve_sd
        self.synch_curve_shift = synch_curve_shift
        self.synch_curve_spread = synch_curve_spread
        self.dust_amp_shift = dust_amp_shift
        self.dust_amp_spread = dust_amp_spread
        self.dust_beta_ref = dust_beta_ref
        self.dust_beta_sd = dust_beta_sd
        self.dust_beta_shift = dust_beta_shift
        self.dust_beta_spread = dust_beta_spread
        self.dust_T_ref = dust_T_ref
        self.dust_T_sd = dust_T_sd
        self.dust_T_shift = dust_T_shift
        self.dust_T_spread = dust_T_spread
        self.ame_amp_shift = ame_amp_shift
        self.ame_amp_spread = ame_amp_spread
        self.ame_peak_nu_ref = ame_peak_nu_ref
        self.ame_peak_nu_sd = ame_peak_nu_sd
        self.ame_peak_nu_shift = ame_peak_nu_shift
        self.ame_peak_nu_spread = ame_peak_nu_spread
        self.cmb_amp_shift = cmb_amp_shift
        self.cmb_amp_spread = cmb_amp_spread

    def get_model(self):
        """                                                                                                                               
        Function for returning the pymc3 model instance from the specified model.           
        """
        return getattr(self, self.pol_model)()

    def synch_mbb(self):

        """
        Generates the pymc3 model instance for the following components:
        1. Power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. CMB.
        """

        pol_JeffGauss_model_0 = pm.Model()

        with pol_JeffGauss_model_0:
            
            synch_amp_norm = pm.Flat('synch_amp_norm', shape=np.shape(self.sky_vals)[1])
            synch_amp = pm.Deterministic('synch_amp', self.synch_amp_shift + synch_amp_norm * self.synch_amp_spread)
            synch_beta = pm.Normal('synch_beta', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            
            dust_amp_norm = pm.Flat('dust_amp_norm', shape=np.shape(self.sky_vals)[1])
            dust_amp = pm.Deterministic('dust_amp', self.dust_amp_shift + dust_amp_norm * self.dust_amp_spread)
            dust_beta = pm.Normal('dust_beta', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_T = pm.Normal('dust_T', mu=self.dust_T_ref, sigma=self.dust_T_sd)
            
            cmb_amp_norm = pm.Flat('cmb_amp_norm', shape=np.shape(self.sky_vals)[1])
            cmb_amp = pm.Deterministic('cmb_amp', self.cmb_amp_shift + cmb_amp_norm * self.cmb_amp_spread)
            
            synch_beta_logp = synch_beta_Jeff_0(synch_beta, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_beta_potential = pm.Potential('synch_beta_potential', synch_beta_logp)

            dust_beta_logp = dust_beta_Jeff(dust_beta, dust_T, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_T_logp = dust_T_Jeff(dust_T, dust_beta, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_beta_potential = pm.Potential('dust_beta_potential', dust_beta_logp)
            dust_T_potential = pm.Potential('dust_T_potential', dust_T_logp)

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_JeffGauss_model_0

    def curve_synch_mbb(self):

        """
        Generates the pymc3 model instance for the following components:
        1. Curved power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. CMB.
        """

        pol_JeffGauss_model_1 = pm.Model()

        with pol_JeffGauss_model_1:

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Normal('synch_beta', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_curve = pm.Flat('synch_curve', mu=self.synch_curve_ref, sigma=self.synch_curve_sd)

            dust_amp = pm.Flat('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Normal('dust_beta', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_T = pm.Normal('dust_T', mu=self.dust_T_ref, sigma=self.dust_T_sd)

            cmb_amp = pm.Normal('cmb_amp', shape=np.shape(self.sky_vals)[1])

            synch_beta_logp = synch_beta_Jeff_1(synch_beta, synch_curve, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_curve_logp = synch_curve_Jeff(synch_curve, synch_beta, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_beta_potential = pm.Potential('synch_beta_potential', synch_beta_logp)
            synch_curve_potential = pm.Potential('synch_curve_potential', synch_curve_logp)

            dust_beta_logp = dust_beta_Jeff(dust_beta, dust_T, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_T_logp = dust_T_Jeff(dust_T, dust_beta, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_beta_potential = pm.Potential('dust_beta_potential', dust_beta_logp)
            dust_T_potential = pm.Potential('dust_T_potential', dust_T_logp)

            sky = Synchrotron(Model='curved_power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta, Spectral_Curvature=synch_curve).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_JeffGauss_model_1

    def synch_mbb_spdust(self):

        """
        Generates the pymc3 model instance for the following components:
        1. Power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. Polarised AME (spdust2).
        4. CMB.
        """

        pol_JeffGauss_model_2 = pm.Model()

        with pol_JeffGauss_model_2:

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Normal('synch_beta', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)

            dust_amp = pm.Flat('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Normal('dust_beta', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_T = pm.Normal('dust_T', mu=self.dust_T_ref, simga=self.dust_T_sd)

            ame_amp = pm.Flat('ame_amp', shape=np.shape(self.sky_vals)[1])
            ame_peak_nu = pm.Normal('ame_peak_nu', mu=self.ame_peak_nu_ref, sigm=self.ame_peak_nu_sd)

            cmb_amp = pm.Flat('cmb_amp', shape=np.shape(self.sky_vals)[1])

            synch_beta_logp = synch_beta_Jeff_0(synch_beta, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_beta_potential = pm.Potential('synch_beta_potential', synch_beta_logp)

            dust_beta_logp = dust_beta_Jeff(dust_beta, dust_T, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_T_logp = dust_T_Jeff(dust_T, dust_beta, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_beta_potential = pm.Potential('dust_beta_potential', dust_beta_logp)
            dust_T_potential = pm.Potential('dust_T_potential', dust_T_logp)

            ame_peak_nu_logp = AME_Jeff(ame_peak_nu, self.nu, self.ame_nu0, self.jeff_sigma)
            ame_peak_nu_potential = pm.Potential('ame_peak_nu_potential', ame_peak_nu_logp)

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                AME(Model='spdust2', Nu=self.nu_matrix, Nu_0=self.ame_nu0, Nu_peak=ame_peak_nu, Amp=ame_amp).signal() \
                + CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_JeffGauss_model_2

    def curve_synch_mbb_spdust(self):

        """
        Generates the pymc3 model instance for the following components:
        1. Curved power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. Polarised AME (spdust2).
        4. CMB.
        """

        pol_JeffGauss_model_3 = pm.Model()

        with pol_JeffGauss_model_3:

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Normal('synch_beta', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_curve = pm.Normal('synch_curve', mu=self.synch_curve_ref, sigma=self.synch_curve_sd)

            dust_amp = pm.Normal('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Normal('dust_beta', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_T = pm.Normal('dust_T', mu=self.dust_T_ref, sigma=self.dust_T_sd)

            ame_amp = pm.Flat('ame_amp', shape=np.shape(self.sky_vals)[1])
            ame_peak_nu = pm.Normal('ame_peak_nu', mu=self.ame_peak_nu_ref, sd=self.ame_peak_nu_sd)

            cmb_amp = pm.Flat('cmb_amp', shape=np.shape(self.sky_vals)[1])

            synch_beta_logp = synch_beta_Jeff_1(synch_beta, synch_curve, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_curve_logp = synch_curve_Jeff(synch_curve, synch_beta, self.nu, self.synch_nu0, self.jeff_sigma)
            synch_beta_potential = pm.Potential('synch_beta_potential', synch_beta_logp)
            synch_curve_potential = pm.Potential('synch_curve_potential', synch_curve_logp)

            dust_beta_logp = dust_beta_Jeff(dust_beta, dust_T, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_T_logp = dust_T_Jeff(dust_T, dust_beta, self.nu, self.dust_nu0, self.jeff_sigma)
            dust_beta_potential = pm.Potential('dust_beta_potential', dust_beta_logp)
            dust_T_potential = pm.Potential('dust_T_potential', dust_T_logp)

            ame_peak_nu_logp = AME_Jeff(ame_peak_nu, self.nu, self.ame_nu0, self.jeff_sigma)
            ame_peak_nu_potential = pm.Potential('ame_peak_nu_potential', ame_peak_nu_logp)

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta, Spectral_Curvature=synch_curve).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                AME(Model='spdust2', Nu=self.nu_matrix, Nu_0=self.ame_nu0, Nu_peak=ame_peak_nu, Amp=ame_amp).signal() \
                + CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_JeffGauss_model_3


class Hierarchical_Pol_Models():

    """

    Model class for hierarchical emission models.

    Attributes
    ----------
    pol_model: The name of the polarised emission model - str.
    nu: Frequencies of the sky maps we are 'component separating' - numpy.ndarray.
    sky_vals: Values of the sky pixel being studied, at each frequency - numpy.ndarray.
    sky_sigma: Associated one sigma uncertainties on sky pixel values - numpy.ndarray.
    synch_nu0: Synchrotron SED reference frequency - float.
    dust_nu0: Dust SED reference frequency - float.
    ame_nu0: AME SED reference frequency - float.
    synch_amp_shift: Synchrotron amplitude shift term - float.
    synch_amp_spread: Synchrotron amplitude scaling term - float.
    synch_beta_ref: Reference synchrotron spectral index for the Gaussian profile - float.
    synch_beta_sd: Standard deviation of the synchrotron spectral index Gaussian profile - float.
    synch_beta_shift: Synchrotron beta shift term - float.
    synch_beta_spread: Synchrotron beta scaling term - float.
    synch_curve_ref: Reference synchrotron curvature for the Gaussian profile - float.
    synch_curve_sd: Standard deviation of the synchrotron curvature Gaussian profile - float.
    synch_curve_shift: Synchrotron curvature shift term - float.
    synch_curve_spread: Synchrotron curvature scaling term - float.
    dust_beta_ref: Reference dust spectral index for the Gaussian profile - float.
    dust_beta_sd: Standard deviation of the dust spectral index Gaussian profile - float.
    dust_beta_shift: Dust beta shift term - float. 
    dust_beta_spread: Dust beta scaling term - float.
    dust_T_ref: Reference dust temperature for the Gaussian profile - float.
    dust_T_sd: Standard deviation for the dust temeprature Gaussian profile - float.
    dust_T_shift: Dust temperature shift term - float.
    dust_T_spread: Dust temperature scaling term - float.
    ame_peak_nu_ref: Reference AME peak frequency for the Gaussian profile - float.
    ame_peak_nu_sd: Standard deviation of the AME peak frequency Gaussian profile - float.
    ame_peak_nu_shift: AME peak nu shift term - float.
    ame_peak_nu_spread: AME peak nu scaling term - float. 

    """

    def __init__(self, pol_model, nu, sky_vals, sky_sigma, synch_nu0=5.0, dust_nu0=353.0, ame_nu0=22.8,
                 synch_amp_shift=100.0, synch_amp_spread=100.0, synch_beta_ref=-3.0, synch_beta_sd=0.1,
                 synch_beta_sigma_ref=0.5, synch_beta_shift=-3.0, synch_beta_spread=0.1, synch_curve_ref=0.0,
                 synch_curve_sd=0.1, synch_curve_sigma_ref=0.5, synch_curve_shift=0.0, synch_curve_spread=0.1,
                 dust_amp_shift=100.0, dust_amp_spread=100.0, dust_beta_ref=1.6, dust_beta_sd=0.3, dust_beta_sigma_ref=0.5,
                 dust_beta_shift=1.6, dust_beta_spread=0.3, dust_T_ref=19.4, dust_T_sd=1.5, dust_T_sigma_ref=1.5,
                 dust_T_shift=19.4, dust_T_spread=1.5, ame_amp_shift=100.0, ame_amp_spread=100.0,
                 ame_peak_nu_ref=22.8, ame_peak_nu_sd=1.0, ame_peak_nu_sigma_ref=1.0, ame_peak_nu_shift=22.8,
                 ame_peak_nu_spread=1.0, cmb_amp_shift=0.0, cmb_amp_spread=1.0):

        self.pol_model = pol_model
        self.nu = nu
        self.nu_matrix = np.transpose(np.broadcast_to(nu, (np.shape(sky_vals)[1], len(nu))))
        self.sky_vals = sky_vals
        self.sky_sigma = sky_sigma
        self.synch_nu0 = synch_nu0
        self.dust_nu0 = dust_nu0
        self.ame_nu0 = ame_nu0
        self.synch_amp_shift = synch_amp_shift
        self.synch_amp_spread = synch_amp_spread
        self.synch_beta_ref = synch_beta_ref
        self.synch_beta_sd = synch_beta_sd
        self.synch_beta_sigma_ref = synch_beta_sigma_ref
        self.synch_beta_shift = synch_beta_shift
        self.synch_beta_spread = synch_beta_spread
        self.synch_curve_ref = synch_curve_ref
        self.synch_curve_sd = synch_curve_sd
        self.synch_curve_sigma_ref = synch_curve_sigma_ref
        self.synch_curve_shift = synch_curve_shift
        self.synch_curve_spread = synch_curve_spread
        self.dust_amp_shift = dust_amp_shift
        self.dust_amp_spread = dust_amp_spread
        self.dust_beta_ref = dust_beta_ref
        self.dust_beta_sd = dust_beta_sd
        self.dust_beta_sigma_ref = dust_beta_sigma_ref
        self.dust_beta_shift = dust_beta_shift
        self.dust_beta_spread = dust_beta_spread
        self.dust_T_ref = dust_T_ref
        self.dust_T_sd = dust_T_sd
        self.dust_T_sigma_ref = dust_T_sigma_ref
        self.dust_T_shift = dust_T_shift
        self.dust_T_spread = dust_T_spread
        self.ame_amp_shift = ame_amp_shift
        self.ame_amp_spread = ame_amp_spread
        self.ame_peak_nu_ref = ame_peak_nu_ref
        self.ame_peak_nu_sd = ame_peak_nu_sd
        self.ame_peak_nu_sigma_ref = ame_peak_nu_sigma_ref
        self.ame_peak_nu_shift = ame_peak_nu_shift
        self.ame_peak_nu_spread = ame_peak_nu_spread
        self.cmb_amp_shift = cmb_amp_shift
        self.cmb_amp_spread = cmb_amp_spread
        
    def get_model(self):
        """
        Function for returning the pymc3 model instance from the specified model.
        """
        return getattr(self, self.pol_model)()

    def synch_mbb(self):
        """
        Generates the pymc3 model instance for the following components:
        1. Power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. CMB.
        """

        pol_hierarchical_model_0 = pm.Model()

        with pol_hierarchical_model_0:

            hyper_synch_beta_ref = (self.synch_beta_ref - self.synch_beta_shift) / self.synch_beta_spread
            hyper_synch_beta_sd = self.synch_beta_sd / self.synch_beta_spread
            synch_beta_mu_norm  = pm.Normal('synch_beta_mu_norm', mu=hyper_synch_beta_ref, sigma=hyper_synch_beta_sd)
            synch_beta_mu = pm.Deterministic('synch_beta_mu', self.synch_beta_shift + synch_beta_mu_norm * self.synch_beta_spread)
            hyper_synch_beta_sigma_ref = self.synch_beta_sigma_ref / self.synch_beta_spread
            synch_beta_sigma_norm = pm.HalfNormal('synch_beta_sigma_norm', sigma=hyper_synch_beta_sigma_ref)
            synch_beta_sigma = pm.Deterministic('synch_beta_sigma', synch_beta_sigma_norm * self.synch_beta_spread)
            
            hyper_dust_beta_ref = (self.dust_beta_ref - self.dust_beta_shift) / self.dust_beta_spread
            hyper_dust_beta_sd = self.dust_beta_sd / self.dust_beta_spread
            dust_beta_mu_norm = pm.Normal('dust_beta_mu_norm', mu=hyper_dust_beta_ref, sigma=hyper_dust_beta_sd)
            dust_beta_mu = pm.Deterministic('dust_beta_mu', self.dust_beta_shift + dust_beta_mu_norm * self.dust_beta_spread)
            hyper_dust_beta_sigma_ref = self.dust_beta_sigma_ref / self.dust_beta_spread
            dust_beta_sigma_norm = pm.HalfNormal('dust_beta_sigma_norm', sigma=hyper_dust_beta_sigma_ref)
            dust_beta_sigma = pm.Deterministic('dust_beta_sigma', dust_beta_sigma_norm * self.dust_beta_spread)
            
            hyper_dust_T_ref = (self.dust_T_ref - self.dust_T_shift) / self.dust_T_spread
            hyper_dust_T_sd = self.dust_T_sd / self.dust_T_spread
            dust_T_mu_norm = pm.Normal('dust_T_mu_norm', mu=hyper_dust_T_ref, sigma=hyper_dust_T_sd)
            dust_T_mu = pm.Deterministic('dust_T_mu', self.dust_T_shift + dust_T_mu_norm * self.dust_T_spread)
            hyper_dust_T_sigma_ref = self.dust_T_sigma_ref / self.dust_T_spread
            dust_T_sigma_norm = pm.HalfNormal('dust_T_sigma_norm', sigma=hyper_dust_T_sigma_ref)
            dust_T_sigma = pm.Deterministic('dust_T_sigma', dust_T_sigma_norm * self.dust_T_spread)

            synch_amp_norm = pm.Flat('synch_amp_norm', shape=np.shape(self.sky_vals)[1])
            synch_amp = pm.Deterministic('synch_amp', self.synch_amp_shift + synch_amp_norm * self.synch_amp_spread)
            synch_beta_offset = pm.Normal('synch_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Deterministic('synch_beta', synch_beta_mu + synch_beta_offset * synch_beta_sigma)

            dust_amp_norm = pm.Flat('dust_amp_norm', shape=np.shape(self.sky_vals)[1])
            dust_amp = pm.Deterministic('dust_amp', self.dust_amp_shift + dust_amp_norm * self.dust_amp_spread)
            dust_beta_offset = pm.Normal('dust_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Deterministic('dust_beta', dust_beta_mu + dust_beta_offset * dust_beta_sigma)
            dust_T_offset = pm.Normal('dust_temp_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_T = pm.Deterministic('dust_temp', dust_T_mu + dust_T_offset * dust_T_sigma)

            cmb_amp_norm = pm.Flat('cmb_amp_norm', shape=np.shape(self.sky_vals)[1])                                                                        
            cmb_amp = pm.Deterministic('cmb_amp', self.cmb_amp_shift + cmb_amp_norm * self.cmb_amp_spread)

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta).signal() + \
                  Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                               Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                  CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_hierarchical_model_0

    def synch_mbb_QU(self):
        """
        Generates the pymc3 model instance for the following components, assuming common QU spectral parameters:
        1. Power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. CMB.
        """

        pol_hierarchical_QU_model_0 = pm.Model()

        with pol_hierarchical_QU_model_0:

            synch_beta_mu = pm.Normal('synch_beta_mu', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_beta_sigma = pm.HalfNormal('synch_beta_sigma', sigma=self.synch_beta_sigma_ref)

            dust_beta_mu = pm.Normal('dust_beta_mu', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_beta_sigma = pm.HalfNormal('dust_beta_sigma', sigma=self.dust_beta_sigma_ref)

            dust_T_mu = pm.Normal('dust_T_mu', mu=self.dust_T_ref, sigma=self.dust_T_sd)
            dust_T_sigma = pm.HalfNormal('dust_T_sigma', sigma=self.dust_T_sigma_ref)

            synch_amp_norm = pm.Flat('synch_amp_norm', shape=np.shape(self.sky_vals)[1])
            synch_amp = pm.Deterministic('synch_amp', self.synch_amp_shift + synch_amp_norm * self.synch_amp_spread)
            synch_beta_offset = pm.Normal('synch_beta_offset', mu=0, sigma=1, shape=int(np.shape(self.sky_vals)[1] / 2))
            stack_synch_beta_offset = pm.math.concatenate((synch_beta_offset, synch_beta_offset))
            synch_beta = pm.Deterministic('synch_beta', synch_beta_mu + stack_synch_beta_offset * synch_beta_sigma)

            dust_amp_norm = pm.Flat('dust_amp_norm', shape=np.shape(self.sky_vals)[1])
            dust_amp = pm.Deterministic('dust_amp', self.dust_amp_shift + dust_amp_norm * self.dust_amp_spread)
            dust_beta_offset = pm.Normal('dust_beta_offset', mu=0, sd=1, shape=int(np.shape(self.sky_vals)[1] / 2))
            stack_dust_beta_offset = pm.math.concatenate((dust_beta_offset, dust_beta_offset))
            dust_beta = pm.Deterministic('dust_beta', dust_beta_mu + stack_dust_beta_offset * dust_beta_sigma)
            dust_T_offset = pm.Normal('dust_temp_offset', mu=0, sd=1, shape=int(np.shape(self.sky_vals)[1] / 2))
            stack_dust_T_offset = pm.math.concatenate((dust_T_offset, dust_T_offset))
            dust_T = pm.Deterministic('dust_temp', dust_T_mu + stack_dust_T_offset * dust_T_sigma)

            cmb_amp_norm = pm.Flat('cmb_amp_norm', shape=np.shape(self.sky_vals)[1])
            cmb_amp = pm.Deterministic('cmb_amp', self.cmb_amp_shift + cmb_amp_norm * self.cmb_amp_spread)

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta).signal() + \
                  Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                               Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                  CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_hierarchical_QU_model_0

    def curve_synch_mbb(self):
        """
        Generates the pymc3 model instance for the following components:
        1. Curved power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. CMB.
        """

        pol_hierarchical_model_1 = pm.Model()

        with pol_hierarchical_model_1:

            synch_beta_mu = pm.Normal('synch_beta_mu', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_beta_sigma = pm.HalfNormal('synch_beta_sigma', sigma=self.synch_beta_sigma_ref)
            synch_curve_mu = pm.Normal('synch_curve_mu', mu=self.synch_curve_ref, sigma=self.synch_curve_sd)
            synch_curve_sigma = pm.HalfNormal('synch_curve_sigma', sigma=self.synch_curve_sigma_ref)
            dust_beta_mu = pm.Normal('dust_beta_mu', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_beta_sigma = pm.HalfNormal('dust_beta_sigma', sigma=self.dust_beta_sigma_ref)
            dust_T_mu = pm.Normal('dust_T_mu', mu=self.dust_T_ref, sigma=self.dust_T_sd)
            dust_T_sigma = pm.HalfNormal('dust_T_sigma', sigma=self.dust_T_sigma_ref)

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta_offset = pm.Normal('synch_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Deterministic('synch_beta', synch_beta_mu + synch_beta_offset * synch_beta_sigma)
            synch_curve_offset = pm.Normal('synch_curve_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_curve = pm.Deterministic('synch_curve', synch_curve_mu + synch_curve_offset * synch_curve_sigma)

            dust_amp = pm.Flat('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta_offset = pm.Normal('dust_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Deterministic('dust_beta', dust_beta_mu + dust_beta_offset * dust_beta_sigma)
            dust_T_offset = pm.Normal('dust_temp_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_T = pm.Deterministic('dust_temp', dust_T_mu + dust_T_offset * dust_T_sigma)

            cmb_amp = pm.Flat('cmb_amp', shape=np.shape(self.sky_vals)[1])

            sky = Synchrotron(Model='curved_power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta, Spectral_Curvature=synch_curve).signal() + \
                  Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                               Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                  CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_hierarchical_model_1

    def synch_mbb_spdust(self):
        """
        Generates the pymc3 model instance for the following components:
        1. Power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. Polarised AME (spdust2).
        4. CMB.
        """

        pol_hierarchical_model_2 = pm.Model()

        with pol_hierarchical_model_2:

            synch_beta_mu = pm.Normal('synch_beta_mu', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_beta_sigma = pm.HalfNormal('synch_beta_sigma', sigma=self.synch_beta_sigma_ref)
            dust_beta_mu = pm.Normal('dust_beta_mu', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_beta_sigma = pm.HalfNormal('dust_beta_sigma', sigma=self.dust_beta_sigma_ref)
            dust_T_mu = pm.Normal('dust_T_mu', mu=self.dust_T_ref, sigma=self.dust_T_sd)
            dust_T_sigma = pm.HalfNormal('dust_T_sigma', sigma=self.dust_T_sigma_ref)
            ame_peak_nu_mu = pm.Normal('ame_peak_nu_mu', mu=self.ame_peak_nu_ref, sigma=self.ame_peak_nu_sd)
            ame_peak_nu_sigma = pm.HalfNormal('ame_peak_nu_sigma', sigma=self.ame_peak_nu_sigma_ref)

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta_offset = pm.Normal('synch_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Deterministic('synch_beta', synch_beta_mu + synch_beta_offset * synch_beta_sigma)

            dust_amp = pm.Flat('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta_offset = pm.Normal('dust_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Deterministic('dust_beta', dust_beta_mu + dust_beta_offset * dust_beta_sigma)
            dust_T_offset = pm.Normal('dust_temp_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_T = pm.Deterministic('dust_temp', dust_T_mu + dust_T_offset * dust_T_sigma)

            ame_amp = pm.Flat('ame_amp', shape=np.shape(self.sky_vals)[1])
            ame_peak_nu_offset = pm.Normal('ame_peak_nu_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            ame_peak_nu = pm.Deterministic('ame_peak_nu', ame_peak_nu_mu + ame_peak_nu_offset * ame_peak_nu_sigma)

            cmb_amp = pm.Flat('cmb_amp', shape=np.shape(self.sky_vals)[1])

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                AME(Model='spdust2', Nu=self.nu_matrix, Nu_0=self.ame_nu0, Nu_peak=ame_peak_nu, Amp=ame_amp).signal() \
                + CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_hierarchical_model_2

    def curve_synch_mbb_spdust(self):
        """
        Generates the pymc3 model instance for the following components:
        1. Curved power-law synchrotron.
        2. Single modified blackbody thermal dust.
        3. Polarised AME (spdust2).
        4. CMB.
        """

        pol_hierarchical_model_3 = pm.Model()

        with pol_hierarchical_model_3:

            synch_beta_mu = pm.Normal('synch_beta_mu', mu=self.synch_beta_ref, sigma=self.synch_beta_sd)
            synch_beta_sigma = pm.HalfNormal('synch_beta_sigma', sigma=self.synch_beta_sigma_ref)
            synch_curve_mu = pm.Normal('synch_curve_mu', mu=self.synch_curve_ref, sigma=self.synch_curve_sd)
            synch_curve_sigma = pm.HalfNormal('synch_curve_sigma', sigma=self.synch_curve_sigma_ref)
            dust_beta_mu = pm.Normal('dust_beta_mu', mu=self.dust_beta_ref, sigma=self.dust_beta_sd)
            dust_beta_sigma = pm.HalfNormal('dust_beta_sigma', sigma=self.dust_beta_sigma_ref)
            dust_T_mu = pm.Normal('dust_T_mu', mu=self.dust_T_ref, sigma=self.dust_T_sd)
            dust_T_sigma = pm.HalfNormal('dust_T_sigma', sigma=self.dust_T_sigma_ref)
            ame_peak_nu_mu = pm.Normal('ame_peak_nu_mu', mu=self.ame_peak_nu_ref, sigma=self.ame_peak_nu_sd)
            ame_peak_nu_sigma = pm.HalfNormal('ame_peak_nu_sigma', sigma=self.ame_peak_nu_sigma_ref)

            synch_amp = pm.Flat('synch_amp', shape=np.shape(self.sky_vals)[1])
            synch_beta_offset = pm.Normal('synch_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_beta = pm.Deterministic('synch_beta', synch_beta_mu + synch_beta_offset * synch_beta_sigma)
            synch_curve_offset = pm.Normal('synch_curve_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            synch_curve = pm.Deterministic('synch_curve', synch_curve_mu + synch_curve_offset * synch_curve_sigma)

            dust_amp = pm.Flat('dust_amp', shape=np.shape(self.sky_vals)[1])
            dust_beta_offset = pm.Normal('dust_beta_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_beta = pm.Deterministic('dust_beta', dust_beta_mu + dust_beta_offset * dust_beta_sigma)
            dust_T_offset = pm.Normal('dust_temp_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            dust_T = pm.Deterministic('dust_temp', dust_T_mu + dust_T_offset * dust_T_sigma)

            ame_amp = pm.Flat('ame_amp', shape=np.shape(self.sky_vals)[1])
            ame_peak_nu_offset = pm.Normal('ame_peak_nu_offset', mu=0, sd=1, shape=np.shape(self.sky_vals)[1])
            ame_peak_nu = pm.Deterministic('ame_peak_nu', ame_peak_nu_mu + ame_peak_nu_offset * ame_peak_nu_sigma)

            cmb_amp = pm.Flat('cmb_amp', shape=np.shape(self.sky_vals)[1])

            sky = Synchrotron(Model='power_law', Nu=self.nu_matrix, Amp=synch_amp, Nu_0=self.synch_nu0,
                              Spectral_Index=synch_beta, Spectral_Curvature=synch_curve).signal() + \
                Thermal_Dust(Model='modified_blackbody', Nu=self.nu_matrix, Amp=dust_amp, Nu_0=self.dust_nu0,
                             Spectral_Index=dust_beta, Dust_Temp=dust_T).signal() + \
                AME(Model='spdust2', Nu=self.nu_matrix, Nu_0=self.ame_nu0, Nu_peak=ame_peak_nu, Amp=ame_amp).signal() \
                + CMB(Model='cmb', Nu=self.nu_matrix, Acmb=cmb_amp).signal()

            sky_like = pm.Normal('sky_like', mu=pm.math.flatten(sky), sigma=np.ndarray.flatten(self.sky_sigma),
                                 observed=np.ndarray.flatten(self.sky_vals))

        return pol_hierarchical_model_3
