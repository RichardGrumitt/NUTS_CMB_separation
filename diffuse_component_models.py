import numpy as np
import theano.tensor as tt
from spdust_interp import spdust_theano, spdust_Nup0


# Some physical constants
h = 6.62607004*10**(-34)
kb = 1.38064852*10**(-23)
Tcmb = 2.7255

'''

Note that frequencies should be given in GHz for all these models.

'''

class Synchrotron(object):

    '''

    Synchrotron emission model class, defining various theano compatible SED expressions.
    Attributes can apply to intensity or polarisation, but you will need to call separate
    instances for I, Q and U etc. Current methods are simple power law, curved power law
    and moment expansion SEDs. 

    Attributes
    ----------

    Model: SED model being used, power_law, curved_power_law or moment_expansion.
    Nu: Frequency at which to calculate the synchrotron signal -- numpy.ndarray or float.
    Amp: Amplitude of synchrotron emission at reference frequency -- float, numpy.ndarray or pymc3 object.
    Nu_0: Reference frequency for the synchrotron SED -- float.
    Spectral_Index: Spectral index for synchrotron emission -- float, numpy.ndarray or pymc3 object.
    Spectral_Curvature: Spectral curvature for synchrotron emission -- float, numpy.ndarray or pymc3 object.
    Moments: Sum of power law moments for synchrotron (no amplitude - spectral index correlation) 
    -- numpy.ndarray or theano/pymc3 object.

    '''

    def __init__(self, Model, Nu, Amp=1.0, Nu_0=23.0, Spectral_Index=-3.0, Spectral_Curvature=0.2, Moments=[0.0, 0.0]):

        self.Model = Model
        self.Nu = Nu
        self.Amp = Amp
        self.Nu_0 = Nu_0
        self.Spectral_Index = Spectral_Index
        self.Spectral_Curvature = Spectral_Curvature
        self.Moments = Moments

    def signal(self):
        '''
        Function for returning the signal from the specified model.
        '''
        return getattr(self, self.Model)()

    def power_law(self):
        '''
        Function for evaluating a simple power law SED for synchrotron.
        '''
        return self.Amp*(self.Nu/self.Nu_0)**(self.Spectral_Index)

    def curved_power_law(self):
        '''
        Function for evaluating a curved power law SED for synchrotron.
        '''
        return self.Amp*(self.Nu/self.Nu_0)**(self.Spectral_Index + 
                                              0.5*self.Spectral_Curvature*tt.log(self.Nu/self.Nu_0))
        
    def moment_expansion(self):
        '''
        Function for evaluating a moment expansion SED for synchrotron.
        '''
        moment_factor = 1.
        for i in range(0, len(self.Moments)):
            moment_factor = moment_factor + (self.Moments[i]*(tt.log(self.Nu/self.Nu_0))**float(i + 2))/tt.gamma(float(i + 3))
        return (self.Amp*(self.Nu/self.Nu_0)**(self.Spectral_Index))*moment_factor
        

class Thermal_Dust(object):

    '''

    Thermal dust emission model class, defining various theano compatible SED expressions.
    Attributes can apply to intensity or polarisation, but you will need to call separate 
    instances for I, Q and U etc. Current methods are modified blackbody (add two population
    dust model, Hensley/Draine and moment expansion SEDs).

    Attributes
    ----------

    Model: SED model being used, modified blackbody, two population dust model,
    Hensley/Draine or moment expansion.
    Nu: Frequency at which to calculate the thermal dust signal -- numpy.ndarray or float.
    Amp: Amplitude of dust emission at reference frequency -- float or pymc3/theano object.
    Nu_0: Reference frequency for thermal dust emission -- float.
    Spectral_Index: Emissitivty spectral index for thermal dust -- float or pymc3/theano object.
    Dust_Temp: Temperature of thermal dust grains -- float or pymc3/theano object.

    '''

    def __init__(self, Model, Nu, Amp=1.0, Nu_0=353.0, Spectral_Index=1.0, Dust_Temp=20.0):

        self.Model = Model
        self.Nu = Nu
        self.Amp = Amp
        self.Nu_0 = Nu_0
        self.Spectral_Index = Spectral_Index
        self.Dust_Temp = Dust_Temp

    def signal(self):
        '''
        Function for returning the signal from the specified model.
        '''
        return getattr(self, self.Model)()

    def modified_blackbody(self):
        '''
        Function for evaluating a modified blackbody SED for thermal dust.
        '''
        gamma = h*1.0e9/(kb*self.Dust_Temp) # 1.0e9 factor there so we can work in GHz.
        scaling = ((self.Nu/self.Nu_0)**(self.Spectral_Index + 1.0))*((tt.exp(gamma*self.Nu_0) - 1.0)/(tt.exp(gamma*self.Nu) - 1.0))
        return self.Amp*scaling


class FreeFree(object):

    '''

    Free-free emission model class, defining various theano compatible SED expressions.
    Attributes can apply to intensity or polarisation, but you will need to call separate
    instances for I, Q and U etc. Current methods are Planck free-free emission model.

    Attributes 
    ----------

    Model: SED model being used, Planck free-free model.
    Nu: Frequency at which to calculate the free-free signal -- numpy.ndarray or float.
    EM: Emission measure -- pymc3/theano object.
    Te: Electron temperature -- pymc3/theano object.

    '''

    def __init__(self, Model, Nu, EM=1.0, Te=1.0):

        self.Model = Model
        self.Nu = Nu
        self.EM = EM
        self.Te = Te

    def signal(self):
        '''
        Function for returning the signal from the specified model.
        '''
        return getattr(self, self.Model)()

    def freefree(self):
        '''
        Function for evaluating the Planck free-free SED model.
        '''
        gff = tt.log(tt.exp(5.96 - tt.sqrt(3)*tt.log((self.Nu)*((self.Te/10.0**4)**(-1.5)))/np.pi) + np.exp(1.0))
        tau = 0.05468*self.Te**(-1.5)*(self.Nu)**(-2.0)*self.EM*gff
        return 10.0**6*self.Te*(1.0 - tt.exp(-tau))
        


class AME(object):

    '''
   
    AME emission model class, defining various theano compatible SED expressions.
    Attributes can apply to intensity or polarisation, but you will need to call separate
    instances for I, Q and U etc. Current model is spdust2.

    Attributes
    ----------

    Model: Model being used, spdust2.
    Nu: Frequency at which to calculate the AME signal -- numpy.ndarray or float.
    Nu_0: Reference frequency for AME emission -- float.
    Nu_peak: Peak frequency of AME emission -- float or pymc3/theano object.
    Amp: Amplitude of AME emission -- float or pymc3/theano object.

    '''

    def __init__(self, Model, Nu, Nu_0=22.8, Nu_peak=30.0, Amp=1.0):

        self.Model = Model
        self.Nu = Nu
        self.Nu_0 = Nu_0
        self.Nu_peak = Nu_peak
        self.Amp = Amp

    def signal(self):
        '''
        Function for returning the signal from the specified model.
        '''
        return getattr(self, self.Model)()
    
    def spdust2(self):
        '''
        Function for evaluating the spdust2 SED.
        '''
        return self.Amp*(self.Nu_0/self.Nu)**2*spdust_theano(self.Nu*spdust_Nup0/self.Nu_peak)/spdust_theano(self.Nu_0*spdust_Nup0/self.Nu_peak)


class CMB(object):

    '''

    CMB emission model class, defining various theano compatible SED expressions.
    Attributes can apply to intensity or polarisation, but you will need to call separate
    instances for I, Q and U etc. Current model is CMB blackbody SED.


    Attributes
    ----------

    Model: Model being used, CMB blackbody.
    Nu: Frequency at which to calculate the CMB signal -- numpy.ndarray or float.
    Acmb: CMB amplitude -- float or pymc3/theano object.

    '''

    def __init__(self, Model, Nu, Acmb=0.0001):

        self.Model = Model
        self.Nu = Nu
        self.Acmb = Acmb

    def signal(self):
        '''
        Function for returning the signal from the specified model.
        '''
        return getattr(self, self.Model)()

    def cmb(self):
        '''
        Function for evaluating the CMB blackbody SED.
        '''
        x = h*10**9*self.Nu/(kb*Tcmb)
        g = (tt.exp(x) - 1.0)**2/(x**2*tt.exp(x))
        return self.Acmb/g
