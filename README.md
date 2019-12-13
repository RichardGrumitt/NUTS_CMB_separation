# NUTS_CMB_separation
Hierarchical Bayesian CMB component separation with the No-U-Turn Sampler. The method/algorithm is described in https://arxiv.org/abs/1910.14170.

Primary file that will perform a component separation run is separate_regions_QandU.py. This needs to be called with an ini file, specifying input parameters and a region number i.e.

python separate_regions_QandU.py {ini_name}.ini {region_number}

This is just one way of running things. Many choices were made for ease of parallelisation on the Galmdring cluster. Instead of this, you can use the classes in pol_sky_model.py to construct your probabilistic models, and sample.py to run the PyMC3 magic inference button over these models.

Note - This is still under heavy development, and has mainly been tested on the Oxford Glamdring cluster. These are just the core scripts. A large amount of associated data (used in running/testing) is stored on the cluster.
