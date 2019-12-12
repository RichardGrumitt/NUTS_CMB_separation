from pymc3.step_methods.hmc.quadpotential import QuadPotentialFull
import pymc3 as pm
import numpy as np


"""

This code has been taken from Dan Foreman-Mackey, and can be found at:

https://dfm.io/posts/pymc3-mass-matrix/

This is currently implemented in the exoplanet code:

https://github.com/dfm/exoplanet

The code implements an additional tuning schedule to estimate off-diagonal elements of 
the mass matrix. Matrix elements are estimated from the covariances of tuning traces.

For a moderate number of parameters, estimating the off-diagonal elements of the 
mass matrix can help to de-correlate the target distribution, and significantly
improve the sampler performance in the case of correlated parameters.

The advantages are reduced as the number of parameters becomes very large, due
to the need to evaluate a matrix inverse at the end of tuning, and the need to
evaluate full matrix products during leapfrog steps.

"""

def get_step_for_trace(trace=None, model=None,
                       regular_window=5, regular_variance=1e-3,
                       **kwargs):
    model = pm.modelcontext(model)
    
    # If not given, use the trivial metric
    if trace is None:
        potential = QuadPotentialFull(np.eye(model.ndim))
        return pm.NUTS(potential=potential, **kwargs)
        
    # Loop over samples and convert to the relevant parameter space;
    # I'm sure that there's an easier way to do this, but I don't know
    # how to make something work in general...
    samples = np.empty((len(trace) * trace.nchains, model.ndim))
    i = 0
    for chain in list(trace._straces.values()):
        for p in chain:
            samples[i] = model.bijection.map(p)
            i += 1
    
    # Compute the sample covariance
    cov = np.cov(samples, rowvar=0)
    
    # Stan uses a regularized estimator for the covariance matrix to
    # be less sensitive to numerical issues for large parameter spaces.
    # In the test case for this blog post, this isn't necessary and it
    # actually makes the performance worse so I'll disable it, but I
    # wanted to include the implementation here for completeness
    N = len(samples)
    cov = cov * N / (N + regular_window)
    cov[np.diag_indices_from(cov)] += \
        regular_variance * regular_window / (N + regular_window)
    
    # Use the sample covariance as the inverse metric
    potential = QuadPotentialFull(cov)
    return pm.NUTS(potential=potential, **kwargs)
