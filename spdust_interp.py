import theano.tensor as tt
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np


# This is included for completeness because people like to use the SpDust models for AME.
# You need these custom theano ops to use the model with PyMC3.
# I do not recommend this - the model does not play well numerically when evaluating gradients,
# which can lead to unreliable results when using HMC sampling algorithms!

# SpDust theano interpolation ops - spdust template is function of frequency in GHz!

spdust_freq = np.loadtxt('spdust_template.csv', usecols=0)
spdust_amp = np.loadtxt('spdust_template.csv', usecols=1)

# The AME template peak frequency, in GHz.
spdust_Nup0 = spdust_freq[np.where(spdust_amp==np.amax(spdust_amp))[0]][0]

scipy_spdust_interp = InterpolatedUnivariateSpline(spdust_freq, spdust_amp, k=4)
scipy_spdust_deriv1 = scipy_spdust_interp.derivative(n=1)
scipy_spdust_deriv2 = scipy_spdust_interp.derivative(n=2)

class SpDust_Deriv2_Op(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def perform(self, nodes, inputs, outputs):

        x, = inputs
        z = outputs[0]
        z[0] = scipy_spdust_deriv2(x)

spdust_deriv2 = SpDust_Deriv2_Op()

class SpDust_Deriv1_Op(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def perform(self, nodes, inputs, outputs):

        x, = inputs
        z = outputs[0]
        z[0] = scipy_spdust_deriv1(x)

    def grad(self, inputs, gradients):

        x, = inputs
        gz, = gradients
        return [gz*spdust_deriv2(x)]

spdust_deriv1 = SpDust_Deriv1_Op()

class SpDust_Op(tt.Op):

    itypes = [tt.dvector]
    otypes = [tt.dvector]

    def perform(self, nodes, inputs, outputs):

        x, = inputs
        z = outputs[0]
        z[0] = scipy_spdust_interp(x)

    def grad(self, inputs, gradients):

        x, = inputs
        gz, = gradients
        return [gz*spdust_deriv1(x)]

spdust_theano = SpDust_Op()
