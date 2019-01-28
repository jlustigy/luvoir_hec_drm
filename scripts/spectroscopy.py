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
import copy

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

CHANNELS = ["UV", "vis", "NIR"]
LAMHR, AHR, FSTAR = cg.get_earth_reflect_spectrum()

def default_luvoir(architecture = "A", channel = "vis"):
    """
    Returns the :class:`coronagraph.Telescope` for the `architecture` and
    `channel` specified by the user.
    """

    telescope = cg.Telescope()

    # Set paramaters for Architecture A
    if (architecture.lower() == "A".lower()):
        telescope.diameter = 15.0
        telescope.contrast = 1e-10
        telescope.A_collect = 155.2
        telescope.diam_inscribed = 13.5
        telescope.Tsys = 270.
        telescope.OWA = 64.
        telescope.qe = 0.9 * 0.75   # Detector QE * charge transfer term
        if channel.lower() == "vis".lower():
            telescope.IWA = 3.5
            telescope.resolution = 140.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.515
            telescope.lammax = 1.030
            telescope.Rc = 1.3e-3
        elif channel.lower() == "UV".lower():
            telescope.IWA = 4.0
            telescope.resolution = 7.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.2
            telescope.lammax = 0.525
            telescope.Rc = 1.3e-3
        elif channel.lower() == "NIR".lower():
            telescope.IWA = 3.5
            telescope.resolution = 70.
            telescope.throughput = 0.18
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.0
            telescope.lammax = 2.0
            telescope.Rc = 0.0
        else:
            print("Unknown `channel`")
            return None

    # Set paramaters for Architecture B
    elif (architecture.lower() == "B".lower()):
        telescope.diameter = 8.0
        telescope.contrast = 1e-10
        telescope.A_collect = 43.4
        telescope.diam_inscribed = 6.7
        telescope.Tsys = 270.
        telescope.OWA = 64.
        telescope.qe = 0.9 * 0.75   # Detector QE * charge transfer term
        if channel.lower() == "vis".lower():
            telescope.IWA = 3.5
            telescope.resolution = 140.
            telescope.throughput = 0.20
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.515
            telescope.lammax = 1.030
            telescope.Rc = 1.3e-3
        elif channel.lower() == "UV".lower():
            telescope.IWA = 4.0
            telescope.resolution = 7.
            telescope.throughput = 0.20
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.200
            telescope.lammax = 0.525
            telescope.Rc = 1.3e-3
        elif channel.lower() == "NIR".lower():
            telescope.IWA = 3.5
            telescope.resolution = 70.
            telescope.throughput = 0.20
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.00
            telescope.lammax = 2.00
            telescope.Rc = 0.0
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

def determine_exposure_time(cn, bandlims, wantSNR = 10.0, wantetime = 5.0, ref_lam = 0.550,
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

    # Specify Kat's fiducial S/N
    iref = np.argmin(np.fabs(cn.lam - ref_lam))

    if bandlims is not None:

        # Specify band via wavelength
        icont = np.array([np.argmin(np.fabs(cn.lam - bandlims[0])), np.argmin(np.fabs(cn.lam - bandlims[1]))])
        iband = np.arange(icont[0]+1, icont[1])
        ibottom = np.argmin(np.fabs(cn.Cratio - np.min(cn.Cratio[iband])))

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
        fid_snrs[i] = cn.SNRt[iref]
        if bandlims is not None:
            band_snrs[i] = cg.observe.SNR_band(cn.cp, ccont, cn.cb, iband, itime=times[i])
            bot_snrs[i] = cn.SNRt[ibottom]
            cont_snrs[i] = np.mean(cn.SNRt[icont])

    # Fit for time to desired snr value
    etime_fid = find_time_from_snr(times, fid_snrs, wantSNR) #times[np.argmin(np.fabs(fid_snrs - wantSNR))]
    if bandlims is not None:
        etime_band = find_time_from_snr(times, band_snrs, wantSNR) #times[np.argmin(np.fabs(band_snrs - wantSNR))]
        etime_bot = find_time_from_snr(times, bot_snrs, wantSNR) #times[np.argmin(np.fabs(bot_snrs - wantSNR))]
        etime_cont = find_time_from_snr(times, cont_snrs, wantSNR) #times[np.argmin(np.fabs(cont_snrs - wantSNR))]

    # Check for incomplete bands which can cause anomalously low exposure times
    if  bandlims is None:
        etime_band = np.nan
        etime_bot = np.nan
        etime_cont = np.nan
    else:
        if (False in np.isfinite(cn.Cobs[iband])):
            etime_band = np.nan

    # Make plot of SNR vs exposure time
    if plot_snr_curves:

        fig, ax = plt.subplots(figsize = (8,6))
        ax.set_xlabel("Exposure Time [hrs]")
        ax.set_ylabel("S/N")
        if bandlims is not None:
            ax.plot(times, band_snrs, label = "detect band rel. to cont.")
            ax.plot(times, bot_snrs, label = "bottom of band")
            ax.plot(times, cont_snrs, label = "avg. continuum")
        ax.plot(times, fid_snrs, label = "at %.2f $\mu$m" %cn.lam[iref])
        if bandlims is not None:
            ax.scatter(etime_band, wantSNR, c="C0")
            ax.scatter(etime_bot, wantSNR, c="C1")
            ax.scatter(etime_cont, wantSNR, c="C2")
        ax.scatter(etime_fid, wantSNR, c="C3")
        ax.axhline(wantSNR, ls = "--", c = "grey")
        if bandlims is not None:
            ax.axvline(etime_band, ls = "--", c = "C0")
            ax.axvline(etime_bot, ls = "--", c = "C1")
            ax.axvline(etime_cont, ls = "--", c = "C2")
        ax.axvline(etime_fid, ls = "--", c = "C3")
        ylims = ax.get_ylim()
        if bandlims is not None:
            ax.text(etime_band, ylims[1]-.5*ylims[1], "%.2f" %etime_band, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C0")
            ax.text(etime_bot, ylims[1]-.1*ylims[1], "%.2f" %etime_bot, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C1")
            ax.text(etime_cont, ylims[1]-.15*ylims[1], "%.2f" %etime_cont, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C2")
        ax.text(etime_fid, ylims[1]-.20*ylims[1], "%.2f" %etime_fid, ha = "center", va = "top", fontsize = 12, bbox=dict(facecolor='w', alpha=1.0, ec = "w"), color = "C3")
        ax.legend(framealpha = 0.75, fontsize = 14)

    if plot_spectrum:

        # Construct noised spectrum plot
        if bandlims is not None:
            cn.make_fake_data(texp = etime_band)
        else:
            cn.make_fake_data(texp = etime_fid)

        fig, ax = plt.subplots(figsize = (8,6))
        ax.plot(cn.lam, cn.Cratio, ls = "steps-mid", color = "grey")
        ax.errorbar(cn.lam, cn.Cobs, yerr=cn.Csig, fmt = "o", ms = 2.0, alpha = 0.7, color = "k")
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel("Fp/Fs")
        ax.set_title(title)

        if bandlims is not None:
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
    Fs = cg.noise_routines.Fstar(LAMHR, cn.star.Teff, cn.star.Rs, cn.planet.a, AU=True)
    #Fs = fstar

    # Run count rates
    cn.run_count_rates(AHR, LAMHR, Fs)

    return cn

def calculate_bandpass_edges(lammin, lammax, bandwidth = 0.2):
    """
    Calculate the wavelengths of the edges of bandpasses given a minimum and
    maximum wavelength for the channel and the bandwidth.

    Returns
    -------
    ``edges`` : numpy.array
    """

    edge = lammin
    edges = []

    edges.append(edge)
    while edge < lammax:
        edge += bandwidth*edge
        edges.append(edge)

    edges[-1] = lammax

    edges = np.array(edges)

    return edges

def calc_observable_spectrum(cn):
    """
    Calculate the percentage of the spectrum observable, considering IWA and OWA
    """

    # Calculate percent of observable spectrum
    pct_obs_iwa = float(sum(np.isfinite(cn.Cobs))) / float(len(cn.Cobs))

    if pct_obs_iwa > 0.0:
        lammax_obs_iwa = np.max(cn.lam[np.isfinite(cn.Cobs)])
    else:
        lammax_obs_iwa = np.nan

    return pct_obs_iwa, lammax_obs_iwa

def apply_two_channels(t_chan):
    """
    Deal with the "two channels at a time" thing
    """

    # Sort the exposure times
    isort = np.argsort(t_chan)

    # Find channel that requires the longest exposure time
    imax = isort[-1]

    # Record longest time
    t_long = t_chan[imax]

    # Record sum of the other two channels
    t_sum = (np.sum(t_chan) - t_chan[imax])

    # Logic for deciding total exposure time
    if t_long > t_sum:
        # if the longest exposure channel is longer than the other two combined
        # then they can both be observed during t_long
        t_tot = t_long
    else:
        # Otherwise the t_long can be observed while the during both the second and third
        # longest exposure time channels
        t_tot = t_sum

    return t_tot

def complete_spectrum_time(cn, Ahr_flat = 0.1618, wantSNR = 5.0, plot = False, verbose = False):
    """
    Time for a complete spectrum

    Parameters
    ----------
    Ahr_flat : float
        Flat albedo spectrum (default is the median Earth
        spectrum between 0.2 - 1.8 um)
    wanrSNR : float
        Desired SNR on spectrum
    plot : bool
        Produce a plot?
    verbose : bool
        Print things?

    Returns
    -------
    t_tot : array-like
        total exposure time
    t_per_band_per_chan : list
        time per band per channel
    spectrum : tuple
        (lam, Cratio, Cobs, Csig)
    iwa : tuple
        (pct_obs_iwa, lammax_obs_iwa)

    """

    if plot: fig, ax = plt.subplots()

    cc = ["C0", "C2", "C3"]
    t_chan = np.zeros(len(CHANNELS))
    Nbands_per_chan = np.zeros(len(CHANNELS))
    t_per_band_per_chan = []
    full_lam = []
    full_Cobs = []
    full_Cratio = []
    full_Csig = []
    pct_obs_iwa = []
    lammax_obs_iwa = []

    # Loop over telescope channels
    for j, channel in enumerate(CHANNELS):

        t_tmp = []

        # Get the channel specific telescope parameters
        luvoir = default_luvoir(channel=channel)
        cn.telescope = luvoir

        if verbose: print(channel, luvoir.lammin, luvoir.lammax)

        # Calculate the bandpass edges
        edges = calculate_bandpass_edges(luvoir.lammin, luvoir.lammax, bandwidth = 0.2)

        # Calculate the number of bandpasses
        Nbands = len(edges) - 1
        Nbands_per_chan[j] = Nbands

        # Run count rates (necessary to generate new wavelength grid)
        cn.run_count_rates(AHR, LAMHR, FSTAR)

        # Calculate how much of the spectrum is observable
        pct, lammax_obs = calc_observable_spectrum(cn)
        pct_obs_iwa.append(pct)
        lammax_obs_iwa.append(lammax_obs)

        # Loop over bandpasses
        for i in range(Nbands):

            # Get the max, min, and middle wavelenths for this bandpass
            lammin = edges[i]
            lammax = edges[i+1]
            lammid = 0.5*(lammax + lammin)

            # Set telescope wavelength range
            cn.telescope.lammin = lammin
            cn.telescope.lammax = lammax

            # Set spectrum to use for exposure time calcs
            # Using flat spectrum so not biased by bottom of bands
            Ahr_flat  = Ahr_flat * np.ones(len(LAMHR))

            # Run count rates (necessary to generate new wavelength grid)
            cn.run_count_rates(Ahr_flat, LAMHR, FSTAR)

            # Calculate exposure times to wantSNR
            etimes = determine_exposure_time(cn, None, plot_snr_curves=False,
                        plot_spectrum=False, wantSNR=wantSNR, ref_lam = lammid)
            t_ref_lam = etimes[-1]

            # Re-do count rate calcs for true Earth spectrum
            cn.run_count_rates(AHR, LAMHR, FSTAR)

            # Draw random samples of data for a plot
            cn.make_fake_data(texp=t_ref_lam)

            if verbose: print(lammid, t_ref_lam)

            # Plot
            if plot:
                ax.axvspan(lammin, lammax, alpha = 0.2, color = cc[j])
                ax.plot(cn.lam, cn.Cratio, ls = "steps-mid", color = "grey", zorder = 100)
                ax.errorbar(cn.lam, cn.Cobs, yerr=cn.Csig, fmt = "o", ms = 2.0, alpha = 0.7, color = "k")
                ax.set_xlabel("Wavelength [$\mu$m]")
                ax.set_ylabel("$F_p / F_s$")

            # Save values
            t_tmp.append(t_ref_lam)
            full_lam.append(cn.lam)
            full_Cratio.append(cn.Cratio)
            full_Cobs.append(cn.Cobs)
            full_Csig.append(cn.Csig)

            # Add time
            if np.isfinite(t_ref_lam):
                t_chan[j] += t_ref_lam

        # Save tmp times per band
        t_per_band_per_chan.append(t_tmp)

    # Deal with the "two channels at a time" thing
    t_tot = apply_two_channels(t_chan)

    spectrum = (np.array(full_lam), np.array(full_Cratio), np.array(full_Cobs), np.array(full_Csig))
    iwa = (pct_obs_iwa, lammax_obs_iwa)

    return t_tot, t_per_band_per_chan, spectrum, iwa

def calc_dtdc(tpbpcs, specs):
    """
    Calculate partial derivatives of exposure time wrt completeness for each bandpass

    Parameters
    ----------
    tpbpcs : list
        Time per bandpass per channel for each star
    """

    # Calc wavelength range of channels
    deltas = []
    for channel in CHANNELS:
        l = default_luvoir(channel=channel)
        deltas.append(l.lammax - l.lammin)
    deltas = np.array(deltas)

    Nbs = len(np.hstack(tpbpcs[0]))
    Ndraw = len(tpbpcs)

    dtpb = np.zeros((Ndraw, Nbs))  # Time per band
    dcomp = np.zeros((Ndraw, Nbs)) # completeness
    dtdc = np.zeros((Ndraw, Nbs))  # d(time) / d(completeness)

    for i in range(Ndraw):

        # Create array of times per bandpass
        dtpb[i,:] = np.hstack(tpbpcs[i])

        for j in range(Nbs):

            # Calculate delta completeness: fractional increase in completeness from observing this band
            dcomp[i,j] = (specs[i][0][j][-1] - specs[i][0][j][0]) / np.sum(deltas)

    # Calculate d(exposure time) / d(completeness)
    dtdc = dtpb / dcomp

    return dtpb, dcomp, dtdc

def calc_dtdc_star(tpbpc0, spec):
    """
    Calculate partial derivatives of exposure time wrt completeness for each bandpass
    for a single star

    Parameters
    ----------
    tpbpc : list
        Time per bandpass per channel
    """

    # Calc wavelength range of channels
    deltas = []
    for channel in CHANNELS:
        l = default_luvoir(channel=channel)
        deltas.append(l.lammax - l.lammin)
    deltas = np.array(deltas)

    # Create array of times per bandpass
    dtpb = np.hstack(tpbpc0)

    Nbs = len(dtpb)

    dcomp = np.zeros(Nbs) # completeness
    dtdc = np.zeros(Nbs)  # d(time) / d(completeness)

    for j in range(Nbs):

        # Calculate delta completeness: fractional increase in completeness from observing this band
        dcomp[j] = (spec[0][j][-1] - spec[0][j][0]) / np.sum(deltas)

    # Calculate d(exposure time) / d(completeness)
    dtdc = dtpb / dcomp

    return dtpb, dcomp, dtdc

def calc_t_chan(tpbpc):
    """
    Calculate the exposure time per channel, given the exposure times in each bandpass
    """
    t_chan_new = []
    for j in range(len(tpbpc)):
        tmp = 0.0
        for k in range(len(tpbpc[j])):
            if np.isfinite(tpbpc[j][k]):
                tmp += tpbpc[j][k]
        t_chan_new.append(tmp)
    t_chan_new = np.array(t_chan_new)
    return t_chan_new

def remove_worst_bandpass(tpbpc0, spec):
    """
    """

    # Make a copy
    new_tpbpc = copy.deepcopy(tpbpc0)

    # Calculate exptime derivatives
    dtpb, dcomp, dtdc = calc_dtdc_star(new_tpbpc, spec)

    # Calculate the largest finite dt/dc derivative
    maxderiv = np.max(dtdc[np.isfinite(dtdc)])
    imax = np.argmin(np.fabs(dtdc - maxderiv))

    modcomp = dcomp[imax]
    modtime = dtpb[imax]

    # Set the time per band to nan to effectively remove it
    icount = 0
    for ix in range(len(new_tpbpc)):
        #print("ix", ix)
        for iy in range(len(new_tpbpc[ix])):
            #print("iy", iy)
            if icount == imax:
                #print(icount)
                new_tpbpc[ix][iy] = np.inf
            icount += 1

    return new_tpbpc, modtime, modcomp, maxderiv

def remove_N_worst_bandpasses(tpbpcs, specs, completeness, t_tots, N = 1, verbose = False):
    """
    """

    cur_tpbpcs = copy.deepcopy(tpbpcs)

    # Calculate exptime derivatives
    dtpb, dcomp, dtdc = calc_dtdc(cur_tpbpcs, specs)

    new_completeness = np.zeros((N, len(completeness)))
    new_t_tots = np.zeros((N, len(t_tots)))
    maxderivs = np.zeros((N, len(t_tots)))
    new_tpbpcs = []

    for i in range(len(dist)):

        if verbose:
            print("-- original: %.3f%% completeness in %.2f hours" %(completeness[i]*100, t_tots[i]))

        modcomp = 0.0
        modtime = 0.0

        for j in range(N):

            # Calculate the largest finite dt/dc derivative
            maxderiv = np.max(dtdc[i][np.isfinite(dtdc[i])])
            imax = np.argmin(np.fabs(dtdc[i] - maxderiv))
            maxderivs[j, i] = maxderiv

            # Set this index to nan for next iteration
            dtdc[i][imax] = np.nan

            if verbose:
                print("but it takes %.2f hours just to add %.3f%% completeness (%.2f - %.2f um)" %(dtpb[i,imax], 100*dcomp[i,imax], specs[0][0][imax][0], specs[0][0][imax][-1]))

            modcomp += dcomp[i,imax]
            modtime += dtpb[i,imax]

            # Set the time per band to nan to effectively remove it
            icount = 0
            for ix in range(len(cur_tpbpcs[i])):
                #print("ix", ix)
                for iy in range(len(cur_tpbpcs[i][ix])):
                    #print("iy", iy)
                    if icount == imax:
                        #print(icount)
                        cur_tpbpcs[i][ix][iy] = np.inf
                    icount += 1


            #t_chan_new = calc_t_chan(tpbpc_new)

            new_completeness[j, i] = completeness[i] - modcomp
            new_t_tots[j, i] = t_tots[i] - modtime

            # Save time per band if this is the last star in set
            if i == (len(dist) - 1):
                new_tpbpcs.append(cur_tpbpcs)

            if verbose:
                print("%i step down: %.3f%% completeness in %.2f hours" %(j+1, new_completeness[j, i]*100, new_t_tots[j, i]))

    return new_t_tots, new_completeness, cur_tpbpcs, maxderivs

if __name__ == "__main__":

    pass
