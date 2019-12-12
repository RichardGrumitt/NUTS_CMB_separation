import numpy as np
import theano.tensor as tt
from spdust_interp import spdust_theano, spdust_deriv1, spdust_Nup0


#########################################################################################################################

# Priors in this file consist of two classes. The Jeffreys prior (assuming a Gaussian likelihood) multiplied by a Gaussian
# profile, and the raw Jeffreys prior. Be warned, if you choose to use the raw Jeffreys prior you will need a lot of patience
# if you want things to converge. A relatively diffuse Gaussian profile helps with the sampling speed quite dramatically 
# (~ 100 times speed-up). This is what is used in current pixel based component separation e.g. COMMANDER, BayesFore etc.
# I can see why, without it the problems become un-scalable to the whole sky. From some initial tests with the available
# frequency channels, it seems wuite rare that the prior ends up driving the posterior.

# The terms here are added as potentials to the pymc3 likelihood. This is the easiest way to implement Jeffreys priors,
# given that parameters appear in the Jeffreys priors on other parameters. Such parameters should be initialised with 
# a flat distribution e.g. synch_beta = pm.Flat('synch_beta'), or with pm.Uniform('synch_beta', min_val, max_val) if you want 
# to include hard bounds on your parameters. This can then be passed to the logp function in the potential. 

# Also worth noting that I personally am not a fan of many of the arguments used for Jeffreys priors and uninformative priors
# more generally. They are included here though because everyone in the field seems to like them.

#########################################################################################################################

# Synchrotron priors - Jeffreys priors multiplied by a Gaussian profile and plain Jeffreys priors.


def synch_beta_JeffGauss_0(beta, Nu, Nu0, sigma, beta_ref, sd_beta):

    """

    Logp function for the synchrotron beta Jeffreys prior multiplied by a Gaussian profile (no curvature).

    Inputs
    ------
    beta: Synchrotron beta parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    beta_ref: Central beta value for the Gaussian profile - float.
    sd_beta: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta)*tt.log(Nu/Nu0)/sigma)**2)) \
        - (beta - beta_ref)**2/(2.0*sd_beta**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_beta**2))


def synch_beta_JeffGauss_1(beta, C, Nu, Nu0, sigma, beta_ref, sd_beta):

    """

    Logp function for the synchrotron beta Jeffreys prior multiplied by a Gaussian profile (curvature).

    Inputs
    ------
    beta: Synchrotron beta parameter - pymc3 object.
    C: Curvature parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    beta_ref: Central beta value for the Gaussian profile - float.
    sd_beta: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 0.5*C*tt.log(Nu/Nu0))*tt.log(Nu/Nu0)/sigma)**2)) \
        - (beta - beta_ref)**2/(2.0*sd_beta**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_beta**2))


def synch_curve_JeffGauss(C, beta, Nu, Nu0, sigma, C_ref, sd_C):

    """

    Logp function for the synchrotron curvature Jeffreys prior multiplied by a Gaussian profile.

    Inputs
    ------
    C: Curvature parameter - pymc3 object.
    beta: Synchrotron beta parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    C_ref: Central curvature value for the Gaussian profile - float.
    sd_C: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 0.5*C*tt.log(Nu/Nu0))*tt.log(Nu/Nu0)**2/sigma)**2)) \
        - (C - C_ref)**2/(2.0*np.pi*sd_C**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_C**2))


def synch_beta_Jeff_0(beta, Nu, Nu0, sigma):

    """

    Logp for the synchrotron beta Jeffreys prior (no curvature).

    Inputs
    ------
    beta: Synchrotron beta parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.

    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta)*tt.log(Nu/Nu0)/sigma)**2))


def synch_beta_Jeff_1(beta, C, Nu, Nu0, sigma):

    """

    Logp function for the synchrotron beta Jeffreys prior.

    Inputs
    ------
    beta: Synchrotron beta parameter - pymc3 object.
    C: Curvature parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.

    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 0.5*C*tt.log(Nu/Nu0))*tt.log(Nu/Nu0)/sigma)**2))


def synch_curve_Jeff(C, beta, Nu, Nu0, sigma):

    """
    
    Logp function for the synchrotron curvature Jeffreys prior.

    Inputs
    ------
    C: Curvature parameter - pymc3 object.
    beta: Synchrotron beta parameter - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    
    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 0.5*C*tt.log(Nu/Nu0))*tt.log(Nu/Nu0)**2/sigma)**2))


######################################################################################################################### 

# Thermal dust priors - Jeffreys priors multiplied by a Gaussian profile and plain Jeffreys priors.

h = 6.62607004e-34 * 1.0e9 # 1.0e9 factor because frequencies are in GHz.
k = 1.38064852e-23


def dust_beta_JeffGauss(beta, T, Nu, Nu0, sigma, beta_ref, sd_beta):

    """

    Dust spectral index Jeffreys prior, multiplied by a Gaussian profile.

    Inputs
    ------
    beta: Dust spectral index - pymc3 object.
    T: Dust temperature - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    beta_ref: Central beta value for the Gaussian profile - float.
    sd_beta: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 1.0)*((tt.exp(h*Nu0/(k*T)) - 1.0)/(tt.exp(h*Nu/(k*T)) - 1.0))*tt.log(Nu/Nu0)/sigma)**2)) \
        - (beta - beta_ref)**2/(2.0*sd_beta**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_beta**2))


def dust_T_JeffGauss(T, beta, Nu, Nu0, sigma, T_ref, sd_T):

    """

    Dust temperature Jeffreys prior, multiplied by a Gaussian profile.

    Inputs
    ------
    T: Dust temperature - pymc3 object.
    beta: Dust spectral index - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    T_ref: Central T value for the Gaussian profile - float.
    sd_T: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 1.0)*((tt.exp(h*Nu0/(k*T)) - 1.0)/(tt.exp(h*Nu/(k*T)) - 1.0))
                              * (Nu0/(1 - tt.exp(-h*Nu0/(k*T))) - Nu/(1 - tt.exp(-h*Nu/(k*T))))/(sigma*T**2))**2)) \
        - (T - T_ref)**2/(2.0*sd_T**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_T**2))


def dust_beta_Jeff(beta, T, Nu, Nu0, sigma):

    """

    Dust spectral index Jeffreys prior.

    Inputs
    ------
    beta: Dust spectral index - pymc3 object.
    T: Dust temperature - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.

    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum(( (Nu/Nu0)**(beta + 1.0) * ((tt.exp(h*Nu0/(k*T)) - 1.0)/(tt.exp(h*Nu/(k*T)) - 1.0)) * tt.log(Nu/Nu0)/sigma)**2))


def dust_T_Jeff(T, beta, Nu, Nu0, sigma):

    """

    Dust temperature Jeffreys prior.

    Inputs
    ------
    T: Dust temperature - pymc3 object.
    beta: Dust spectral index - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    
    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum(((Nu/Nu0)**(beta + 1.0)*((tt.exp(h*Nu0/(k*T)) - 1.0)/(tt.exp(h*Nu/(k*T)) - 1.0)) \
                              *(Nu0/(1 - tt.exp(-h*Nu0/(k*T))) - Nu/(1 - tt.exp(-h*Nu/(k*T))))/(sigma*T**2))**2))

#########################################################################################################################

# Free-free priors - Jeffreys priors multiplied by a Gaussian profile and plain Jeffreys priors.

def freefree_EM_JeffGauss(EM, Te, Nu, sigma, EM_ref, sd_EM):

    """

    Free-free emission measure Jeffreys prior, multiplied by a Gaussian profile.

    Inputs
    ------
    EM: The free-free emission measure - pymc3 object.
    Te: The electron temeprature - float.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    EM_ref: The central emission measure value for the Gaussian profile - float.
    sd_EM: The standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    gff = tt.log(tt.exp(5.96 - tt.sqrt(3)*tt.log(Nu*(Te/10000.0)**(-1.5))/np.pi) + tt.exp(1.0))
    FT = 0.05468*Te**(-1.5)*Nu**(-2.0)*gff
    
    return 0.5*tt.log(tt.sum((Te*FT*tt.exp(-FT*EM)/sigma)**2)) \
        + (EM - EM_ref)**2/(2.0*sd_EM**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_EM**2))

def freefree_EM_Jeff(EM, Te, Nu, sigma):

    """

    Free-free emission measure Jeffreys prior.

    Inputs
    ------
    EM: The free-free emission measure - pymc3 object.
    Te: The electron temeprature - float.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.

    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    gff= tt.log(tt.exp(5.96 - tt.sqrt(3)*tt.log(Nu*(Te/10000.0)**(-1.5))/np.pi) + tt.exp(1.0))
    FT= 0.05468*Te**(-1.5)*Nu**(-2.0)*gff

    return 0.5*tt.log(tt.sum((Te*FT*tt.exp(-FT*EM)/sigma)**2))

#########################################################################################################################

# AME priors - Jeffreys priors multiplied by a Gaussian profile and plain Jeffreys priors.

# Import AME template and derivative functions - will need to wrap these as theano ops.
# Call these spdust() and Dspdust()

def AME_JeffGauss(Nup, Nu, Nu0, sigma, Nup_ref, sd_Nup):

    """

    AME peak frequency Jeffreys prior multiplied by a Gaussian profile.

    Inputs
    ------
    Nup: The AME peak frequency - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    Nup_ref: Central peak frequency value for the Gaussian profile - float.
    sd_Nup: Standard deviation of the Gaussian profile - float.

    Returns
    -------
    logp: Logp for the Jeffreys prior multiplied by a Gaussian profile - theano tensor.

    """

    return 0.5*tt.log(tt.sum((Nu0/Nu)**2*(spdust_theano(Nu*spdust_Nup0/Nup)/spdust_theano(Nu0*spdust_Nup0/Nup))*(spdust_Nup0/Nup**2)* \
                             (spdust_deriv1(Nu0*spdust_Nup0/Nup)*Nu0/spdust_theano(Nu0*spdust_Nup0/Nup) -
                              spdust_deriv1(Nu*spdust_Nup0/Nup)*Nu/spdust_theano(Nu*spdust_Nup0/Nup))**2)) \
                              + (Nup - Nup_ref)**2/(2.0*sd_Nup**2) + tt.log(1.0/tt.sqrt(2.0*np.pi*sd_Nup**2))

def AME_Jeff(Nup, Nu, Nu0, sigma):

    """

    AME peak frequency Jeffreys prior.

    Inputs
    ------
    Nup: The AME peak frequency - pymc3 object.
    Nu: Frequencies of the maps you are 'component separating' - numpy.ndarray.
    Nu0: Reference frequency - float.
    sigma: Sigma uncertainties for each map in the pixel being evaluated - numpy.ndarray.
    
    Returns
    -------
    logp: Logp for the Jeffreys prior - theano tensor.

    """

    return 0.5*tt.log(tt.sum((Nu0/Nu)**2*(spdust_theano(Nu*spdust_Nup0/Nup)/spdust_theano(Nu0*spdust_Nup0/Nup))*(spdust_Nup0/Nup**2)* \
                             (spdust_deriv1(Nu0*spdust_Nup0/Nup)*Nu0/spdust_theano(Nu0*spdust_Nup0/Nup) - 
                              spdust_deriv1(Nu*spdust_Nup0/Nup)*Nu/spdust_theano(Nu*spdust_Nup0/Nup))**2))
