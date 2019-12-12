import numpy as np
import healpy as hp
from astropy.stats import mad_std


def amp_residuals(q_map_file, u_map_file, input_map_file, param_name,
                  out_dir, out_prefix):

    """

    Function for calculating residuals of output sky maps.

    Inputs
    ------
    :q_map_file: Output Stokes Q map file - str.
    :u_map_file: Output Stokes U map file - str.
    :input_map_file: Input map file - str.
    :param_name: Name of the parameter you are interested in (used for output formatting) - str.
    :out_dir: Output directory for residual maps - str.
    :out_prefix: Output filename prefix for residual maps - str.

    Returns
    -------
    :q_resid: Parameter Q map residuals - numpy.ndarray.
    :u_resid: Parameter U map residuals - numpy.ndarray.

    """
    
    q_map = hp.read_map(q_map_file)
    u_map = hp.read_map(u_map_file)
    in_map = hp.read_map(input_map_file, field=(1, 2))
    q_resid = q_map - in_map[0]
    u_resid = u_map - in_map[1]
    q_mean = np.mean(q_resid)
    u_mean = np.mean(u_resid)
    q_std = mad_std(q_resid)
    u_std = mad_std(u_resid)
    
    hp.write_map(f'{out_dir}{out_prefix}_Q_{param_name}_residuals.fits', q_resid, overwrite=True)
    hp.write_map(f'{out_dir}{out_prefix}_U_{param_name}_residuals.fits', u_resid, overwrite=True)
    print(f'Residual mean for Q {param_name} = {q_mean}')
    print(f'Residual mean for U {param_name} = {u_mean}')
    print(f'Residual std for Q {param_name} = {q_std}')
    print(f'Residual std for U {param_name} = {u_std}')
    
    return q_resid, u_resid
    

def spectral_residuals(component_map_file, input_map_file, param_name, out_dir, out_prefix):

    """

    Function for calculating residuals of output spectral parameter maps i.e. no Q and U.

    Inputs
    ------
    :component_map_file: Spectral parameter output map file - str.
    :input_map_file: Spectral parameter input map file - str.
    :param_name: Name of the parameter you are interested in (used for output formatting) - str.
    :out_dir: Output directory for residual map - str.
    :out_prefix: Output filename prefix for residual map - str.

    Returns
    -------
    :resid: Residual map of the spectral parameter - numpy.ndarray.

    """
    
    comp_map = hp.read_map(component_map_file)
    in_map = hp.read_map(input_map_file)
    resid = comp_map - in_map
    resid_mean = np.mean(resid)
    resid_std = mad_std(resid)
    
    hp.write_map(f'{out_dir}{out_prefix}_{param_name}_residuals.fits', resid, overwrite=True)
    print(f'Residual mean for {param_name} = {resid_mean}')
    print(f'Residual std for {param_name} = {resid_std}')
    
    return resid
