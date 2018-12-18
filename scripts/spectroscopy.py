"""
luvoir_spectroscopy.py
======================
Methods for calculating exposure times and yields for the habitable exoplanet
characterization design reference mission for the LUVOIR STDT Final Report.
"""
from __future__ import (division as _, print_function as _,
                absolute_import as _, unicode_literals as _)

import os
import platform
import sys
import numpy as np
import scipy as sp
import scipy.optimize
from scipy import stats
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
import pdb
import h5py
import subprocess
import datetime
import multiprocessing
from functools import partial
import pandas as pd

try:
    import coronagraph as cg
except ImportError:
    print("Failed to import `coronagraph`. Install from https://github.com/jlustigy/coronagraph")


import platform
if platform.system() == 'Darwin':
    # On a Mac: usetex ok
    mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=True)
elif platform.node().startswith("D"):
    # On hyak: usetex not ok, must change backend to 'agg'
    mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=False)
    plt.switch_backend('agg')
else:
    # On astro machine or other linux: usetex not ok
    plt.switch_backend('agg')
    mpl.rc('font', family='Times New Roman')
    mpl.rcParams['font.size'] = 25.0
    mpl.rc('text', usetex=False)

NCPU = multiprocessing.cpu_count()
HERE = os.path.abspath(os.path.split(__file__)[0])

def default_luvoir(architecture = "A", channel = "vis"):
    """
    Returns the :class:`coronagraph.Telescope` for the `architecture` and
    `channel` specified by the user.
    """

    telescope = cg.Telescope()

    # Set paramaters for Architecture A
    if (architecture.lower() == "A".lower()):
        telescope.diameter = 13.5
        telescope.A_collect = 155.
        telescope.Tsys = 270.
        telescope.OWA = 64.
        if channel.lower() == "vis".lower():
            telescope.IWA = 3.5
            telescope.resolution = 150.
            telescope.throughput = 0.15
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 1e-2
            telescope.lammin = 0.5
            telescope.lammax = 1.03
        elif channel.lower() == "UV".lower():
            telescope.IWA = 3.5
            telescope.resolution = 10.
            telescope.throughput = 0.15
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 1e-2
            telescope.lammin = 0.2
            telescope.lammax = 0.5
        elif channel.lower() == "NIR".lower():
            telescope.IWA = 2.4
            telescope.resolution = 150.
            telescope.throughput = 0.15
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.03
            telescope.lammax = 1.8
        else:
            print("Unknown `channel`")
            return None

    # Set paramaters for Architecture B
    elif (architecture.lower() == "B".lower()):
        telescope.diameter = 6.7
        telescope.A_collect = 43.8
        telescope.Tsys = 270.
        telescope.OWA = 64.
        if channel.lower() == "vis".lower():
            telescope.IWA = 2.4
            telescope.resolution = 150.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 1e-2
            telescope.lammin = 0.5
            telescope.lammax = 1.03
        elif channel.lower() == "UV".lower():
            telescope.IWA = 3.5
            telescope.resolution = 10.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 1e-2
            telescope.lammin = 0.2
            telescope.lammax = 0.5
        elif channel.lower() == "NIR".lower():
            telescope.IWA = 2.4
            telescope.resolution = 150.
            telescope.throughput = 0.18
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.03
            telescope.lammax = 1.8
        else:
            print("Unknown `channel`")
            return None
    else:
        print("Unknown `architecture`")
        return None

    # Set wavelength-dependent throughput
    tpath = os.path.join(HERE, "../inputs/optical_throughput.txt")
    data = np.genfromtxt(tpath, skip_header=1)
    midlamt = 1e-3 * data[:,2]
    Tput_optics = data[:,3]
    telescope.Tput_lam = (midlamt, Tput_optics)

    return telescope

def read_luvoir_stars(path = os.path.join(HERE, '../inputs/luvoir-A_stars.txt')):
    """
    Read-in file of stars in the biased list from Aki and Chris
    """

    # Read in table of 50 biased draws
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype = str)

    # Parse
    hip = np.array(data[:,0], dtype=int)
    dist = np.array(data[:,1], dtype=float)
    stype = np.array(data[:,2], dtype=str)

    # Construct dictionary
    dic = {
        "hip" : hip,
        "dist" : dist,
        "stype" : stype
    }

    return dic

# Make LUVOIR Stars a global dictionary
STARS = read_luvoir_stars()

def read_stellar_properties(path = os.path.join(HERE, '../inputs/stellar_properties.txt')):
    """
    Read-in file of stellar properties
    """

    # Read in table of stellar types
    data = np.loadtxt(path, skiprows=19, dtype = str)

    # Parse
    stypes = data[:,0]
    masses = np.array(data[:,1], dtype=float)
    lums = np.array(data[:,2], dtype=float)
    rads = np.array(data[:,3], dtype=float)
    temps = np.array(data[:,4], dtype=float)
    mvs = np.array(data[:,6], dtype=float)

    # Construct dictionary
    dic = {
        "stypes" : stypes,
        "masses" : masses,
        "lums" : lums,
        "rads" : rads,
        "temps" : temps,
        "mvs" : mvs
    }

    return dic

# Make stellar properties a global dictionary
STARPROPS = read_stellar_properties()

def match_stellar_type(s, verbose = True):
    """
    Match stellar type ``s`` with the table of stellar types and return the index in table
    that matches
    """

    # Localize array of stellar types (look how formally I am describing nothing)
    stypes = STARPROPS["stypes"]

    # Case: exact match
    for i, st in enumerate(stypes):
        if s == st:
            return i

    if verbose: print("No exact match found")

    # Case: beginning matches
    for i, st in enumerate(stypes):
        if s.startswith(st):
            if verbose: print("Using %s for %s" %(st, s))
            return i

    # Case: ending matches
    for i, st in enumerate(stypes):
        if s.endswith(st):
            if verbose: print("Using %s for %s" %(st, s))
            return i

    # Case: the addition of a "V" provides an exact match
    for i, st in enumerate(stypes):
        if s+"V" == st:
            if verbose: print("Using %s for %s" %(st, s))
            return i

def calc_seff(Teff, S0, coeffs):
    """
    Calculate the Habitable Zone stellar flux from Equation 2 of
    Kopparapu et al. (2013).

    Parameters
    ----------
    Teff : float
        Stellar effective temperature [K]
    S0 : float
        Flux scaling value for particular HZ limit
    coeffs : array-like
        ``[a, b, c, d]`` coefficient values for particular HZ limit

    Returns
    -------
    Seff : float
        Stellar flux for HZ limit
    """
    a, b, c, d = coeffs
    T = Teff - 5780.
    return S0 + a*T + b*T**2 + c*T**3 + d*T**4

def calc_dist(L, Seff):
    """
    Calculate the Habitable Zone distance [AU] from Equation 3 of
    Kopparapu et al. (2013).

    Parameters
    ----------
    L : float
        Luminosity of the star compared to the Sun
    Seff : float
        Stellar flux for HZ limit

    Returns
    -------
    a : float
        Semi-major axis of HZ limit [AU]
    """
    return (L / Seff)**0.5

# Using the moist greenhouse inner edge
S0_inner = 1.0140
inner_edge = [8.1774e-5, 1.7063e-9, -4.3241e-12, -6.6462e-16]

# Using the maximum greenhouse outer edge
S0_outer = 0.3438
outer_edge = [5.8942e-5, 1.6558e-9, -3.0045e-12, -5.2983e-16]

def find_time_from_snr(times, snrs, wantsnr):
    """
    Find the exposure time to a given S/N (``wantsnr``) by fitting a curve of
    S/N (``snrs``) vs. time (``times``) and solving for the time at ``wantsnr``.

    Parameters
    ----------
    times : array-like
    snrs : array-like
    wantsnr : float

    Returns
    -------
    t_snr : float
        Time to ``wantsnr`` [same units as ``times``]
    """

    x = times
    y = snrs**2

    # Fit S/N**2 curve with a line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    y_fit = slope * x + intercept
    snr_fit = np.sqrt(y_fit)

    # Solve for number of occultations to get desired SNR
    t_snr = (wantsnr**2 - intercept) / slope

    return t_snr

def determine_exposure_time(cn, bandlims, wantSNR = 10.0, wantetime = 5.0,
                            plot_snr_curves = False, plot_spectrum = False,
                            title = ""):
    """
    Determine the exposure time needed to get a desired S/N

    Parameters
    ----------
    cn : :class:`coronagraph.CoronagraphNoise`
        Instantiated ``CoronagraphNoise`` object containing ``Telescope``, ``Planet``, and ``Star``
        objects with their associated parameters
    bandlims : list or tuple
        Lower and upper wavelength limits to absorption band [microns]

    Returns
    -------
    etime_band : float
        Exposure time to get S/N on band below the continuum
    etime_bot : float
        Exposure time to get S/N on spectral element at the bottom of the band
    etime_cont : float
        Exposure time to ger S/N on the spectral elements in the continuum nearest to the band
    """

    # Specify band via wavelength
    icont = np.array([np.argmin(np.fabs(cn.lam - bandlims[0])), np.argmin(np.fabs(cn.lam - bandlims[1]))])
    iband = np.arange(icont[0]+1, icont[1])
    ibottom = np.argmin(np.fabs(cn.Cratio - np.min(cn.Cratio[iband])))

    # Specify Kat's fiducial S/N
    i550nm = np.argmin(np.fabs(cn.lam - 0.550))

    # Calculate the continuum planet photon counts and contrast ratio
    ccont = cg.observe.interp_cont_over_band(cn.lam, cn.cp, icont, iband)
    ccrat = cg.observe.interp_cont_over_band(cn.lam, cn.Cratio, icont, iband)

    # Calculate varies SNRs as a function of exposure time
    Nt = 1000
    times = np.linspace(1.0, 100.0, Nt)
    band_snrs = np.zeros(len(times))
    bot_snrs = np.zeros(len(times))
    cont_snrs = np.zeros(len(times))
    fid_snrs = np.zeros(len(times))
    for i, time in enumerate(times):
        cn.make_fake_data(texp = times[i])
        band_snrs[i] = cg.observe.SNR_band(cn.cp, ccont, cn.cb, iband, itime=times[i])
        bot_snrs[i] = cn.SNRt[ibottom]
        cont_snrs[i] = np.mean(cn.SNRt[icont])
        fid_snrs[i] = cn.SNRt[i550nm]

    # Fit for time to desired snr value
    etime_band = find_time_from_snr(times, band_snrs, wantSNR) #times[np.argmin(np.fabs(band_snrs - wantSNR))]
    etime_bot = find_time_from_snr(times, bot_snrs, wantSNR) #times[np.argmin(np.fabs(bot_snrs - wantSNR))]
    etime_cont = find_time_from_snr(times, cont_snrs, wantSNR) #times[np.argmin(np.fabs(cont_snrs - wantSNR))]
    etime_fid = find_time_from_snr(times, fid_snrs, wantSNR) #times[np.argmin(np.fabs(fid_snrs - wantSNR))]

    # Check for incomplete bands which can cause anomalously low exposure times
    if False in np.isfinite(cn.Cobs[iband]):
        etime_band = np.nan

    if plot_snr_curves:
        # Make plot of SNR vs exposure time
        fig, ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel("Exposure Time [hrs]")
        ax.set_ylabel("S/N")
        ax.plot(times, band_snrs, label = "detect band rel. to cont.")
        ax.plot(times, bot_snrs, label = "bottom of band")
        ax.plot(times, cont_snrs, label = "avg. continuum")
        ax.plot(times, fid_snrs, label = "at 550 nm")
        ax.scatter(etime_band, wantSNR, c="C0")
        ax.scatter(etime_bot, wantSNR, c="C1")
        ax.scatter(etime_cont, wantSNR, c="C2")
        ax.scatter(etime_fid, wantSNR, c="C3")
        ax.axhline(wantSNR, ls = "--", c = "grey")
        ax.axvline(etime_band, ls = "--", c = "C0")
        ax.axvline(etime_bot, ls = "--", c = "C1")
        ax.axvline(etime_cont, ls = "--", c = "C2")
        ax.axvline(etime_fid, ls = "--", c = "C3")
        ylims = ax.get_ylim()
        ax.text(etime_band, ylims[1]-.5*ylims[1], "%.2f" %etime_band, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C0")
        ax.text(etime_bot, ylims[1]-.1*ylims[1], "%.2f" %etime_bot, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C1")
        ax.text(etime_cont, ylims[1]-.15*ylims[1], "%.2f" %etime_cont, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C2")
        ax.text(etime_fid, ylims[1]-.20*ylims[1], "%.2f" %etime_fid, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C3")
        ax.legend(framealpha = 0.75, fontsize = 14)

    if plot_spectrum:
        # Construct noised spectrum plot
        cn.make_fake_data(texp = etime_band)
        fig, ax = plt.subplots(figsize = (8,6))
        ax.plot(cn.lam, cn.Cratio, ls = "steps-mid", color = "grey")
        ax.errorbar(cn.lam, cn.Cobs, yerr=cn.Csig, fmt = "o", ms = 2.0, alpha = 0.7, color = "k")
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel("Fp/Fs")
        ax.set_title(title)
        # Identify specific points in band
        for i in icont:
            ax.scatter(cn.lam[i], cn.Cratio[i], s = 20.0, c = "C8", marker = "o", zorder = 100)
        for i in iband:
            ax.scatter(cn.lam[i], cn.Cratio[i], s = 20.0, c = "C1", marker = "o", zorder = 100)
        ax.scatter(cn.lam[ibottom], cn.Cratio[ibottom], s = 20.0, c = "C8", marker = "o", zorder = 100)
        # Identify specific continuum points in band
        for i, ic in enumerate(iband):
            ax.scatter(cn.lam[ic], ccrat[i], s = 20.0, c = "C9", marker = "o", zorder = 100)

    # Return exposure times
    return etime_band, etime_bot, etime_cont, etime_fid

def nsig_intervals(x, intvls=[0.16, 0.5, 0.84]):
    # Compute median and n-sigma intervals
    q_l, q_50, q_h = np.percentile(x, list(100.0 * np.array(intvls)))
    q_m, q_p = q_50-q_l, q_h-q_50
    return q_l, q_50, q_h, q_m, q_p

def prep_ith_star(cn, i):
    """
    Takes a :class:`coronagraph.CoronagraphNoise` object and an index and returns the object
    with parameters set for that system.

    Note
    ----
    Currently requires a bunch of arrays to exist outside of this functions scope,
    e.g. `stype`, `dist`, `temps`, `rads`, `lums`
    """

    # Get index with matching stellar type in stellar properties table
    imatch = match_stellar_type(STARS['stype'][i], verbose = False)

    # Set system distance
    cn.planet.distance = STARS['dist'][i]

    # Set stellar temperature
    cn.star.Teff = STARPROPS['temps'][imatch]

    # Set stellar radius
    cn.star.Rs = STARPROPS['rads'][imatch]

    # Calculate the Earth-equivalent insolation distance
    a_eeq = np.sqrt(STARPROPS['lums'][imatch])

    # Set semi-major axis
    cn.planet.a = a_eeq

    # Set stellar spectrum based on type
    # Calculate stellar flux at TOA assuming a blackbody
    Fs = cg.noise_routines.Fstar(lamhr, cn.star.Teff, cn.star.Rs, cn.planet.a, AU=True)
    #Fs = fstar

    # Run count rates
    cn.run_count_rates(Ahr, lamhr, Fs)

    return cn

if __name__ == "__main__":

    pass
