import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from turbo_colormap import *
import healpy as hp
import numpy as np
from scipy.stats import iqr


fontsize = 36
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({'font.size': fontsize})
plt.rcParams.update({'figure.figsize': [20.0, 18.4]})

def component_map_fig(map_filename, output_filename, map_field=0, min_val=None, max_val=None, norm=None, title='', unit=''):

    """

    Generate figures of component maps.

    Inputs
    ------
    map_filename: Filename of the map you want to make a figure of - str.
    output_filename: Output filename for the map figure - str.
    map_field: Map field you are interested in - int.
    min_val: Lower limit for the map colour scale - float.
    max_val: Upper limit for the map colour scale - float.
    norm: Colour scale normalisation to be applied to the map - str.
    title: Title to add to the map if desired - str.
    unit: Unit label to add to the map if desired - str.

    """

    comp_map = hp.read_map(map_filename, field=map_field)

    plt.figure()
    hp.mollview(comp_map, norm=norm, min=min_val, max=max_val, title=title, unit=unit, cmap=ListedColormap(turbo_colormap_data))
    f = plt.gcf().get_children()
    HpxAx = f[1]
    CbAx = f[2]
    coord_text_obj = HpxAx.get_children()[0]
    coord_text_obj.set_fontsize(fontsize)
    unit_text_obj = CbAx.get_children()[1]
    unit_text_obj.set_fontsize(24)
    plt.savefig(output_filename, dpi=900)
    plt.close()


def norm_diff_hist(Q_map_filename, U_map_filename, Q_sigma_filename, U_sigma_filename, in_cmb_filename, out_dir,
                   out_prefix, mask_filename, labels=[r'$\eta_Q$', r'$\eta_U$'], save_dir=None):

    """
    
    Generates normalised deviation histograms for the CMB Q and U output maps.

    Inputs
    ------
    Q_map_filename: Filename for the output CMB Q map - list.
    U_map_filename: Filename for the output CMB U map - list.
    Q_sigma_filename: Filename for the output CMB Q sigma map - list.
    U_sigma_filename: Filename for the output CMB U sigma map - list.
    in_cmb_filename: Filename for the input CMB map - str.
    out_dir: Output directory for the output histogram - str.
    out_prefix: Filename prefix for the output histogram -str.
    labels: Labels for the various histograms - list.
    save_dir: Directory for saving normalised deviations - str.

    """

    assert len(Q_map_filename) == 3 or len(Q_map_filename) == 1

    in_cmbQ = hp.read_map(in_cmb_filename, field=1)
    in_cmbU = hp.read_map(in_cmb_filename, field=2)
    mask = hp.read_map(mask_filename)
    in_cmbQ = in_cmbQ[mask != 0]
    in_cmbU = in_cmbU[mask != 0]
    
    plt.figure()
    max_vals = []
    std_vals = []
    colours = ['#000000', '#004488', '#BB5566']
    for i in range(len(Q_map_filename)):
        cmbQ = hp.read_map(Q_map_filename[i])
        cmbU = hp.read_map(U_map_filename[i])
        sigmaQ = hp.read_map(Q_sigma_filename[i])
        sigmaU = hp.read_map(U_sigma_filename[i])
        cmbQ = cmbQ[mask != 0]
        cmbU = cmbU[mask != 0]
        sigmaQ = sigmaQ[mask != 0]
        sigmaU = sigmaU[mask != 0]
        normQ = (in_cmbQ - cmbQ)/sigmaQ
        normU = (in_cmbU - cmbU)/sigmaU
        bw = 2 * iqr(normQ) / len(normQ)**(1 / 3)
        num_bins = int((np.amax(normQ) - np.amin(normQ)) / bw)
        plt.hist(normQ, num_bins, histtype='step', color=colours[i], linestyle='-', label=labels[int(2 * i)], density=True, linewidth=4)
        plt.hist(normU, num_bins, histtype='step', color=colours[i], linestyle='-.', label=labels[int(2 * i + 1)], density=True, linewidth=4)
        max_vals.append(np.amax(normQ))
        std_vals.append(np.std(normQ))
    x = np.linspace(-5, 5, 1000)
    plt.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color='#DDAA33', linestyle='--', label=r'$\mathcal{N}(0, 1)$', linewidth=4)
    plt.xlim(-5, 5)
    plt.xlabel(r'$\eta_{\lambda=\{Q,U\}}$')
    plt.ylabel('Density')
    plt.legend(loc='upper right')
    plt.savefig(f'{out_dir}{out_prefix}_norm_dev_hist.pdf')
    plt.close()

    if save_dir is not None:

        print('Warning! This only saves a single set of normalised deviations!')
        np.save(f'{save_dir}{out_prefix}_Q_norm_dev.npy', normQ)
        np.save(f'{save_dir}{out_prefix}_U_norm_dev.npy', normU)

    return normQ, normU


class Power_Spectrum_Plots(object):

    def __init__(self, clBB_filename, clBB_err_filename, clEE_filename, clEE_err_filename, leff_filename, out_dir,
                 out_prefix, fid_r0_lensed_filename='./fiducial_spectra/cmb_fiducial_r0_lensed.dat',
                 fid_r0p01_unlensed_filename='./fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat',
                 theory_r=5e-3, theory_AL=0.4, lmin=5, lmax=150):

        """

        Attributes
        ----------
        
        clBB_filename: Filename of .npy file containing Cl_BB bandpowers - str.
        clBB_err_filename: Filename of .npy file containing Cl_BB bandpower errors - str.
        clEE_filename: Filename of .npy file containing Cl_EE bandpowers - str.
        clEE_err_filename: Filename of .npy file containing Cl_EE bandpower errors - str.
        leff_filename: Filename of .npy file containing the effective multipole values - str.
        out_dir: Output directory for figures - str.
        out_prefix: Prefix for output filenames - str.
        fid_r0_lensed_filename: Filename corresponding to the fiducial spectra with r=0 and lensing - str.
        fid_r0p01_unlensed_filename: Filename corresponding to the primordial only fiducial spectra with r=0.01 - str.
        theory_r: Tensor-to-scalar value used in simulations - float.
        theory_AL: Lensing amplitude used in simulations - float.
        lmin: Minimum multipole to plot - int.
        lmax: Maximum multipole to plot - int.

        """

        self.clBB_filename = clBB_filename
        self.clBB_err_filename = clBB_err_filename
        self.clEE_filename = clEE_filename
        self.clEE_err_filename = clEE_err_filename
        self.leff_filename = leff_filename
        self.out_dir = out_dir
        self.out_prefix = out_prefix
        self.fid_r0_lensed_filename = fid_r0_lensed_filename
        self.fid_r0p01_unlensed_filename = fid_r0p01_unlensed_filename
        self.theory_r = theory_r
        self.theory_AL = theory_AL
        self.lmin = lmin
        self.lmax = lmax
        
    def extract_theory_spectra(self):
        
        fid_dl_lens = np.loadtxt(self.fid_r0_lensed_filename, usecols=3)
        fid_dl_BB = np.loadtxt(self.fid_r0p01_unlensed_filename, usecols=3)
        fid_dl_EE = np.loadtxt(self.fid_r0p01_unlensed_filename, usecols=2)
        fid_ell = np.loadtxt(self.fid_r0p01_unlensed_filename, usecols=0)
        theory_BB = self.theory_r * fid_dl_BB / 0.01 + self.theory_AL * fid_dl_lens
        theory_EE = np.copy(fid_dl_EE)

        return fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens

    def extract_leff(self):

        leff = np.ndarray.flatten(np.load(self.leff_filename))

        return leff
    
    def extract_clBB(self):

        clBB = np.ndarray.flatten(np.load(self.clBB_filename))
        clBB_err = np.ndarray.flatten(np.load(self.clBB_err_filename))

        return clBB, clBB_err

    def extract_clEE(self):

        clEE = np.ndarray.flatten(np.load(self.clEE_filename))
        clEE_err = np.ndarray.flatten(np.load(self.clEE_err_filename))

        return clEE, clEE_err

    def clBB_plot(self):

        fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens = self.extract_theory_spectra()
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()

        plt.figure()
        plt.errorbar(leff, clBB, yerr=clBB_err, color='k', fmt='x')
        plt.plot(fid_ell, 2 * np.pi * theory_BB / (fid_ell * (fid_ell + 1)), color='k', label='Total')
        plt.plot(fid_ell, self.theory_r * 2 * np.pi * fid_dl_BB / (0.01 * fid_ell * (fid_ell + 1)), linestyle='-.',
                 color='m', label='Primordial')
        plt.plot(fid_ell, self.theory_AL * 2 * np.pi * fid_dl_lens / (fid_ell * (fid_ell + 1)), linestyle='--', color='r', label='Lensing')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{BB}\quad [\mu\mathrm{K}^2]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.lmin, self.lmax)
        plt.legend(loc='upper left')
        plt.savefig(f'{self.out_dir}{self.out_prefix}_clBB.pdf', dpi=900)
        plt.close()
        
        return leff, clBB, clBB_err
        
    def clEE_plot(self):

        fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens = self.extract_theory_spectra()
        leff = self.extract_leff()
        clEE, clEE_err = self.extract_clEE()
        
        plt.figure()
        plt.errorbar(leff, clEE, yerr=clEE_err, color='k', fmt='x')
        plt.plot(fid_ell, 2 * np.pi * theory_EE / (fid_ell * (fid_ell + 1)), linestyle='-.', color='m', label='Theory EE')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{EE}\quad [\mu\mathrm{K}^2]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.lmin, self.lmax)
        plt.legend(loc='upper left')
        plt.savefig(f'{self.out_dir}{self.out_prefix}_clEE.pdf', dpi=900)
        plt.close()
        
        return leff, clEE, clEE_err

    def dlBB_plot(self):

        fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens = self.extract_theory_spectra()
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()
        
        dlBB = leff * (leff + 1) * clBB / (2 * np.pi)
        dlBB_err = leff * (leff + 1) * clBB_err / (2 * np.pi)
        
        plt.figure()
        plt.errorbar(leff, dlBB, yerr=dlBB_err, color='k', fmt='x')
        plt.plot(fid_ell, theory_BB, color='k', label='Total')
        plt.plot(fid_ell, self.theory_r * fid_dl_BB / 0.01, linestyle='-.', color='m', label='Primordial')
        plt.plot(fid_ell, self.theory_AL * fid_dl_lens, linestyle='--', color='r', label='Lensing')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell^{BB}/2\pi\quad [\mu\mathrm{K}^2]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.lmin, self.lmax)
        plt.legend(loc='upper left')
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlBB.pdf', dpi=900)
        plt.close()
        
        return leff, dlBB, dlBB_err

    def dlEE_plot(self):
        
        fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens = self.extract_theory_spectra()
        leff = self.extract_leff()
        clEE, clEE_err = self.extract_clEE()

        dlEE = leff * (leff + 1) * clEE / (2 * np.pi)
        dlEE_err = leff * (leff + 1) * clEE_err / (2 * np.pi)
        
        plt.figure()
        plt.errorbar(leff, dlEE, yerr=dlEE_err, color='k', fmt='x')
        plt.plot(fid_ell, theory_EE, linestyle='-.', color='m', label='Theory EE')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_\ell^{EE}\quad [\mu\mathrm{K}^2]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(self.lmin, self.lmax)
        plt.legend(loc='upper left')
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlEE.pdf', dpi=900)
        plt.close()
        
        return leff, dlEE, dlEE_err
    
    def dlBB_dlEE_plot(self):
        
        fid_ell, theory_BB, theory_EE, fid_dl_BB, fid_dl_lens = self.extract_theory_spectra()
        leff = self.extract_leff()
        clBB, clBB_err = self.extract_clBB()
        clEE, clEE_err = self.extract_clEE()
        
        dlBB = leff * (leff + 1) * clBB / (2 * np.pi)
        dlBB_err = leff * (leff + 1) * clBB_err / (2 * np.pi)
        dlEE = leff * (leff + 1) * clEE / (2 * np.pi)
        dlEE_err = leff * (leff + 1) * clEE_err / (2 * np.pi)
        
        plt.figure()
        plt.errorbar(leff, dlBB, yerr=dlBB_err, color='k', fmt='x', label='Measured B-mode')
        plt.errorbar(leff, dlEE, yerr=dlEE_err, color='k', fmt='o', label='Measured E-mode')
        plt.plot(fid_ell, theory_BB, color='k', linestyle='-', label='Input B-mode')
        plt.plot(fid_ell, theory_EE, color='k', linestyle='-.', label='Input E-mode')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi\quad [\mu\mathrm{K}^2]$')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='upper left')
        plt.xlim(self.lmin, self.lmax)
        plt.ylim(1e-5)
        plt.savefig(f'{self.out_dir}{self.out_prefix}_dlBB_dlEE.pdf', dpi=900)
        plt.close()
        
        return leff, dlBB, dlBB_err, dlEE, dlEE_err


def joint_dlBB_dlEE_plot(clBB_filenames, clBB_err_filenames, clEE_filenames, clEE_err_filenames, leff_filenames,
                         out_dir, out_prefix, fid_r0_lensed_filename='./fiducial_spectra/cmb_fiducial_r0_lensed.dat',
                         fid_r0p01_unlensed_filename='./fiducial_spectra/cmb_fiducial_r0p01_unlensed.dat',
                         theory_r=5e-3, theory_AL=0.4, lmin=5, lmax=150, bb_fmts=['x', 'D', 's'],
                         ee_fmts=['o', 'v', '^'], bb_labels=[r'$B$-mode: CP(L)', r'$B$-mode: CP(LC)', r'$B$-mode: H(LC)'],
                         ee_labels=[r'$E$-mode: CP(L)', r'$E$-mode: CP(LC)', r'$E$-mode: H(LC)'],
                         bb_colours=['#EE3377', '#EE7733', '#33BBEE'], ee_colours=['#CC3311', '#009988', '#0077BB']):

    """

    Inputs
    ------

    clBB_filename: Filenames of .npy files containing Cl_BB bandpowers - list.
    clBB_err_filename: Filenames of .npy files containing Cl_BB bandpower errors - list.
    clEE_filename: Filenames of .npy files containing Cl_EE bandpowers - list.
    clEE_err_filename: Filenames of .npy files containing Cl_EE bandpower errors - list.
    leff_filename: Filenames of .npy files containing the effective multipole values - list.
    out_dir: Output directory for figures - str.
    out_prefix: Prefix for output filenames - str.
    fid_r0_lensed_filename: Filename corresponding to the fiducial spectra with r=0 and lensing - str.
    fid_r0p01_unlensed_filename: Filename corresponding to the primordial only fiducial spectra with r=0.01 - str.
    theory_r: Tensor-to-scalar value used in simulations - float.
    theory_AL: Lensing amplitude used in simulations - float.
    lmin: Minimum multipole to plot - int.
    lmax: Maximum multipole to plot - int.

    """

    fid_dl_lens = np.loadtxt(fid_r0_lensed_filename, usecols=3)
    fid_dl_BB = np.loadtxt(fid_r0p01_unlensed_filename, usecols=3)
    fid_dl_EE = np.loadtxt(fid_r0p01_unlensed_filename, usecols=2)
    fid_ell = np.loadtxt(fid_r0p01_unlensed_filename, usecols=0)
    theory_BB = theory_r * fid_dl_BB / 0.01 + theory_AL * fid_dl_lens
    theory_EE = np.copy(fid_dl_EE)

    plt.figure()
    for i in range(len(clBB_filenames)):
        clBB = np.ndarray.flatten(np.load(clBB_filenames[i]))
        clBB_err = np.ndarray.flatten(np.load(clBB_err_filenames[i]))
        clEE = np.ndarray.flatten(np.load(clEE_filenames[i]))
        clEE_err = np.ndarray.flatten(np.load(clEE_err_filenames[i]))
        leff = np.ndarray.flatten(np.load(leff_filenames[i]))
        dlBB = leff * (leff + 1) * clBB / (2 * np.pi)
        dlBB_err = leff * (leff + 1) * clBB_err / (2 * np.pi)
        dlEE = leff * (leff + 1) * clEE / (2 * np.pi)
        dlEE_err = leff * (leff + 1) * clEE_err / (2 * np.pi)
        if i%3 == 0:
            leff = leff - 2
        elif i%3 == 2:
            leff = leff + 2
        plt.errorbar(leff, dlBB, yerr=dlBB_err, color=bb_colours[i], fmt=bb_fmts[i], label=bb_labels[i])
        plt.errorbar(leff, dlEE, yerr=dlEE_err, color=ee_colours[i], fmt=ee_fmts[i], label=ee_labels[i])
    plt.plot(fid_ell, theory_BB, color='k', linestyle='-', label=r'Input $B$-mode')
    plt.plot(fid_ell, theory_EE, color='k', linestyle='-.', label=r'Input $E$-mode')
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$\ell(\ell+1)C_\ell/2\pi\quad [\mu\mathrm{K}^2]$')
    plt.yscale('log')
    plt.legend(loc='upper left')
    plt.xlim(lmin, lmax)
    plt.ylim(1e-5)
    plt.gca().xaxis.set_minor_formatter(mticker.ScalarFormatter())
    plt.savefig(f'{out_dir}{out_prefix}_dlBB_dlEE.pdf', dpi=900)
    plt.close()
