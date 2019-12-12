import numpy as np
import healpy as hp
import os
import glob
from natsort import natsorted
import re

# Miscellaneous helper functions.

def class_import(name):

    """

    Function for importing a python class, given a name e.g. pol_sky_model.Pol_Jeff_Models

    Inputs
    ------
    :name: Class name - str.

    Outputs
    -------
    Imports the class.
    
    """

    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
   

def mk_params_file(regions_file, mask_file, ini_file, out_params_file, joint_QU=False, pix_thresh=1):

    """
    
    Function for generating params file for multirun, listing the jobs
    to be run.

    Inputs
    ------
    :regions_file: The filename for the regions map - str.
    :mask_file: The filename for the mask map - str.
    :ini_file: The filename for the ini file containing the general 
               component separation params - str.
    :out_params_file: The output filename for the params file - str.
    :pix_thresh: Number of pixels at which we allow fitting - str.

    """

    regions = hp.read_map(regions_file)
    region_nums = [i for i in range(1, int(np.amax(regions)) + 1)]
    mask = hp.read_map(mask_file)

    f = open(out_params_file, 'w+')
    for num in region_nums:
        region_pix = regions[regions == num]
        mask_pix = mask[regions == num]
        region_size = len(region_pix)
        if len(mask_pix[mask_pix == 1]) >= pix_thresh:
            if not joint_QU:
                print('Writing command: PYTHONPATH=\"\" python separate_regions_QorU.py {} {}'.format(ini_file, int(num)))
                f.write("PYTHONPATH=\"\" python separate_regions_QorU.py {} {}\r\n".format(ini_file, int(num)))
            elif joint_QU:
                print('Writing command: PYTHONPATH=\"\" python separate_regions_QandU.py {} {}'.format(ini_file, int(num)))
                f.write("PYTHONPATH=\"\" python separate_regions_QandU.py {} {}\r\n".format(ini_file, int(num)))
    f.close()


def theano_cleanup(trace_summary_dir, num_regions, ini_file, joint_QU=False, col_num=3,
                   out_filename='./multirun_theano_cleanup.txt'):

    files = glob.glob(f'{trace_summary_dir}*_summary.csv')
    regex = re.compile(r'\d+')
    completed_numbers = []

    for f in files:
        completed_numbers.append(regex.findall(f))
    completed_numbers = np.array(completed_numbers)[:, col_num].astype(np.int64)
    
    regions = np.arange(1, num_regions + 1).astype(np.int64)
    done = np.isin(regions, completed_numbers)
    need_cleaning = regions[~done]

    f = open(out_filename, 'w+')
    for num in need_cleaning:
        if not joint_QU:
            print(f'Writing command: PYTHONPATH=\"\" python separate_regions_QorU.py {ini_file} {int(num)}')
            f.write(f"PYTHONPATH=\"\" python separate_regions_QorU.py {ini_file} {int(num)}\r\n")
        elif joint_QU:
            print(f'Writing command: PYTHONPATH=\"\" python separate_regions_QandU.py {ini_file} {int(num)}')
            f.write(f"PYTHONPATH=\"\" python separate_regions_QandU.py {ini_file} {int(num)}\r\n")
    f.close()


def mk_file_list(directory, suffix, absolute=False):

    '''
    Function for creating a text file listing all files in a given directory, with some suffix.
    This sorts the files so they are in ascending numerical order.

    Inputs
    ------
    directory - The directory where the map files are located.
    suffix - The suffix of the files you are interested in.
    absolute (optional) - Set top True if you want the absolute file paths to be listed.

    Outputs
    -------
    file_list - A list of the map files in the specified directory.

    '''

    file_list = natsorted(glob.glob('{}*{}'.format(directory, suffix)))
    
    if absolute == True:
        
        temp_list = []
        
        for path in file_list:
            
            temp_list.append(os.path.abspath(path))
            
        file_list = temp_list

    return file_list
