import numpy as np


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
                                              0.5*self.Spectral_Curvature*np.log(self.Nu/self.Nu_0))
        
    def moment_expansion(self):
        '''
        Function for evaluating a moment expansion SED for synchrotron.
        '''
        moment_factor = 1.
        for i in range(0, len(self.Moments)):
            moment_factor = moment_factor + (self.Moments[i]*(np.log(self.Nu/self.Nu_0))**float(i + 2))/np.gamma(float(i + 3))
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
        scaling = ((self.Nu/self.Nu_0)**(self.Spectral_Index + 1.0))*((np.exp(gamma*self.Nu_0) - 1.0)/(np.exp(gamma*self.Nu) - 1.0))
        return self.Amp*scaling


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
        g = (np.exp(x) - 1.0)**2/(x**2*np.exp(x))
        return self.Acmb/g
