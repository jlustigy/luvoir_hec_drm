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
    print("Failed to import `coronagraph`. Please install: `pip install coronagraph`")


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

# Local imports
from luvoir_char import gen_candidate_catalog

NCPU = multiprocessing.cpu_count()
HERE = os.path.abspath(os.path.split(__file__)[0])

CHANNELS = ["UV", "vis", "NIR"]
LAMHR, AHR, FSTAR = cg.get_earth_reflect_spectrum()
ARCHITECTURES = ["APLC", "DMVC", "PIAA"]
ARCH_FILES = [
    "../inputs/LUVOIR-Architecture_A-LBTI_NOMINAL_2-NOMINAL_OCCRATES-APLC_3masks-AVC6-target_list.csv",
    "../inputs/LUVOIR-Architecture_B-LBTI_NOMINAL_2-NOMINAL_OCCRATES-DMVC6-target_list.csv",
    "../inputs/LUVOIR-Architecture_B-LBTI_NOMINAL_2-NOMINAL_OCCRATES-PIAA_mix-target_list.csv"
    ]


################################################################################
# HEC DRM CLASS
################################################################################

class HEC_DRM(object):
    """
    Object for running instances of the LUVOIR Habitable Exoplanet
    Characterization Design Reference Mission (DRM)

    Parameters
    ----------
    wantSNR : float
        Desired SNR in each spectral element
    wantexp : float
        Desired duration of exo-earth program [days]
    Ahr_flat : float
        Flat planet albedo to use for SNR/exposure time estimates
    eta_int : float
        Fraction of targets in biased sample that appear to be habitable/interesting
    bandwidth : float or list
        Coronagraph bandpass bandwidth (:math:`\Delta \lambda / \lambda`). Can be
        a single `float` value for the entire wavelength range, or a `list` of
        floats, with one bandwidth for each channel
        (where ``len(bandwidth) == len(CHANNELS)``).
    architecture : str
        LUVOIR architecture ("A" or "B")
    telescope_mods : dict
        Dictionary of telescope parameters/values to modify from defaults
    """
    def __init__(self, wantSNR=8.5, wantexp=365., Ahr_flat=0.20,
                 eta_int=0.1, bandwidth=0.2, architecture="A",
                 telescope_mods={}, catalog_seed = 1):

        # Set initial attributes
        self.wantSNR = wantSNR
        self.wantexp = wantexp
        self.Ahr_flat = Ahr_flat
        self.eta_int = eta_int
        self.bandwidth = bandwidth
        self.architecture = architecture
        self.telescope_mods = telescope_mods
        self.catalog_seed = catalog_seed

        # Read-in biased stellar catalog based on architecture
        if (architecture == "A") or (architecture == "B"):
            # Backwards compatibility with original results
            self.STARS = read_luvoir_stars(path = os.path.join(HERE, '../inputs/luvoir-%s_stars.txt' %architecture))
        else:
            for i, arch in enumerate(ARCHITECTURES):
                if arch.lower() in architecture.lower():
                    self.STARS = gen_candidate_catalog(os.path.join(HERE, ARCH_FILES[i]), seed = self.catalog_seed)
        #self.biased_sample = self.STARS
        #self.NBIAS = len(self.biased_sample["dist"])

        # Calculate the number of draws based on eta_int
        self.Ndraw = int(np.round(self.eta_int * self.NBIAS))

        # Create coronagraph noise object for calculations
        self.cn = cg.CoronagraphNoise(SILENT = True)

        # Set a fiducial modern earth spectrum for plotting
        self.LAMHR = LAMHR
        self.AHR = AHR
        self.FSTAR = FSTAR

        return

    @property
    def biased_sample(self):
        """
        Biased sample of exo-Earth candidates (this is for backwards
        compatibility and simply returns the attribute `STARS`)
        """
        return self.STARS

    @property
    def NBIAS(self):
        """
        Number of targets in the biased sample of exo-Earth candidates
        (this is for backwards compatibility and simply returns the
        attribute `len(STARS["dist"])`)
        """
        return len(self.STARS["dist"])

    def apply_telescope_mods(self):
        """
        Apply modifications to the default telescope parameters using the
        `telescope_mods` dictionary
        """
        for key, val in self.telescope_mods.items():
            self.cn.telescope.__dict__[key] = val
        return

    def prep_ith_star(self, i):
        """
        Takes an index and returns the object with parameters set for that system.

        Note
        ----
        Currently requires a bunch of arrays to exist outside of this functions scope,
        e.g. `stype`, `dist`, `temps`, `rads`, `lums`
        """

        # Get index with matching stellar type in stellar properties table
        imatch = match_stellar_type(self.STARS['stype'][i], verbose = True)

        # Set system distance
        self.cn.planet.distance = self.STARS['dist'][i]

        # Set stellar temperature
        self.cn.star.Teff = STARPROPS['temps'][imatch]

        # Set stellar radius
        self.cn.star.Rs = STARPROPS['rads'][imatch]

        # Calculate the Earth-equivalent insolation distance
        #a_eeq = np.sqrt(STARPROPS['lums'][imatch])

        # Calculate the semi-major axis for the inner edge (Kopparapu et al. 2013)
        a_in = calc_dist(STARPROPS['lums'][imatch],
                         calc_seff(STARPROPS['temps'][imatch], S0_inner, inner_edge))

        # Set semi-major axis
        self.cn.planet.a = a_in

        # Is the exozodi level defined for each star?
        if "nez" in self.STARS.keys():
            # Set exozodi level
            self.cn.planet.Nez = self.STARS['nez'][i]

        # Set stellar spectrum based on type
        # Calculate stellar flux at TOA assuming a blackbody
        Fs = cg.noise_routines.Fstar(self.LAMHR, self.cn.star.Teff, self.cn.star.Rs, self.cn.planet.a, AU=True)
        #Fs = fstar

        # Run count rates
        self.cn.run_count_rates(self.AHR, self.LAMHR, Fs)

        return

    def determine_exposure_time(self, bandlims, wantSNR = 10.0, wantetime = 5.0, ref_lam = 0.550,
                                plot_snr_curves = False, plot_spectrum = False,
                                title = ""):
        """
        Determine the exposure time needed to get a desired S/N

        Parameters
        ----------
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

        # Calculate various SNRs as a function of exposure time
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

    def complete_spectrum_time(self, plot = False, verbose = False):
        """
        Time for a complete spectrum

        Parameters
        ----------
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
            (lam, dlam, Cratio, Cobs, Csig)
        iwa : tuple
            (pct_obs_iwa, lammax_obs_iwa)

        """

        # If the coronagraph model has already been run...
        if self.cn._computed:
            # Use the existing stellar flux
            fstar = self.cn.solhr
        else:
            # Otherwise use the solar flux
            fstar = self.FSTAR

        if plot: fig, ax = plt.subplots()

        cc = ["C0", "C2", "C3"]
        t_chan = np.zeros(len(CHANNELS))
        Nbands_per_chan = np.zeros(len(CHANNELS))
        t_per_band_per_chan = []
        full_lam = []
        full_dlam = []
        full_Cobs = []
        full_Cratio = []
        full_Csig = []
        pct_obs_iwa = []
        lammax_obs_iwa = []
        lam_extrema = []

        ibp = 0

        # Loop over telescope channels
        for j, channel in enumerate(CHANNELS):

            # Channel dependent bandwidth?
            if type(self.bandwidth) is float:
                bandwidth = self.bandwidth
            else:
                assert len(self.bandwidth) == len(CHANNELS)
                bandwidth = self.bandwidth[j]

            t_tmp = []

            # Get the channel specific telescope parameters
            luvoir = default_luvoir(channel=channel, architecture = self.architecture)
            self.cn.telescope = luvoir

            self.apply_telescope_mods()

            if verbose: print(channel, luvoir.lammin, luvoir.lammax)

            lam_extrema.append(luvoir.lammin)
            lam_extrema.append(luvoir.lammax)

            # Calculate the bandpass edges
            edges = calculate_bandpass_edges(luvoir.lammin, luvoir.lammax, bandwidth = bandwidth)

            # Calculate the number of bandpasses
            Nbands = len(edges) - 1
            Nbands_per_chan[j] = Nbands

            # Calculate how much of the spectrum is observable
            cnc = copy.deepcopy(self.cn)
            cnc.run_count_rates(self.AHR, self.LAMHR, self.FSTAR)
            pct, lammax_obs = calc_observable_spectrum(cnc)
            pct_obs_iwa.append(pct)
            lammax_obs_iwa.append(lammax_obs)

            # Loop over bandpasses
            for i in range(Nbands):

                if (type(self.wantSNR) is float) or (type(self.wantSNR) is int):
                    wSNR = self.wantSNR
                else:
                    wSNR = self.wantSNR[ibp]

                # Get the max, min, and middle wavelenths for this bandpass
                lammin = edges[i]
                lammax = edges[i+1]
                lammid = 0.5*(lammax + lammin)

                # Set telescope wavelength range
                self.cn.telescope.lammin = lammin
                self.cn.telescope.lammax = lammax

                if channel == "UV":
                    self.cn.telescope.lam = np.array([lammid])
                    self.cn.telescope.dlam = np.array([lammax - lammin])

                # Set spectrum to use for exposure time calcs
                # Using flat spectrum so not biased by bottom of bands
                Ahr_flat  = self.Ahr_flat * np.ones(len(self.LAMHR))

                # Run count rates (necessary to generate new wavelength grid)
                self.cn.run_count_rates(Ahr_flat, self.LAMHR, fstar)

                # Calculate exposure times to wantSNR
                etimes = determine_exposure_time(self.cn, None, plot_snr_curves=False,
                            plot_spectrum=False, wantSNR=wSNR, ref_lam = lammid)
                t_ref_lam = etimes[-1]

                # Re-do count rate calcs for fiducial spectrum
                self.cn.run_count_rates(self.AHR, self.LAMHR, fstar)

                # Draw random samples of data for a plot
                self.cn.make_fake_data(texp=t_ref_lam)

                if verbose: print(lammid, t_ref_lam)

                # Plot
                if plot:
                    ax.axvspan(lammin, lammax, alpha = 0.2, color = cc[j])
                    #ax.plot(self.cn.lam, self.cn.Cratio, ls = "steps-mid", color = "grey", zorder = 100)
                    ax.plot(self.cn.lam, self.cn.Cobs, "o", ms = 3.0, alpha = 1.0, color = "w", zorder = 70)
                    ax.errorbar(self.cn.lam, self.cn.Cobs, yerr=self.cn.Csig, fmt = "o", ms = 2.0, alpha = 0.7, color = "k", zorder = 70)
                    ax.set_xlabel("Wavelength [$\mu$m]")
                    ax.set_ylabel("$F_p / F_s$")

                # Save values
                t_tmp.append(t_ref_lam)
                full_lam.append(self.cn.lam)
                full_dlam.append(self.cn.dlam)
                full_Cratio.append(self.cn.Cratio)
                full_Cobs.append(self.cn.Cobs)
                full_Csig.append(self.cn.Csig)

                # Add time
                if np.isfinite(t_ref_lam):
                    t_chan[j] += t_ref_lam

                ibp += 1

            # Save tmp times per band
            t_per_band_per_chan.append(t_tmp)

        # Deal with the "two channels at a time" thing
        t_tot = apply_two_channels(t_chan)

        spectrum = (np.array(full_lam),
                    np.array(full_dlam),
                    np.array(full_Cratio),
                    np.array(full_Cobs),
                    np.array(full_Csig))
        iwa = (pct_obs_iwa, lammax_obs_iwa)

        if plot:
            lam_extrema = np.array(lam_extrema)
            self.cn.telescope.lammin = np.min(lam_extrema)
            self.cn.telescope.lammax = np.max(lam_extrema)
            self.cn.telescope.resolution = 140.
            # Re-do count rate calcs for true Earth spectrum
            self.cn.run_count_rates(self.AHR, self.LAMHR, self.FSTAR)
            ax.plot(self.cn.lam, self.cn.Cratio, color = "grey", zorder = 80, lw = 3.0)
            ax.plot(self.cn.lam, self.cn.Cratio, color = "w", zorder = 80, lw = 2.0)

        return t_tot, t_per_band_per_chan, spectrum, iwa

    def generate_exptime_table(self, ):
        """
        Calculate the exposure times and spectra in each bandpass for each
        star in biased sample, make a Lookup Table of Exposure times for
        each star in sample, and calculate the spectral completeness.
        """

        # Perform calculation for all stars in biased sample
        Ndraw = self.NBIAS

        np.random.seed(seed=None)

        # Allocate memory for exposure times
        t_tots = np.zeros(Ndraw)
        tpbpcs = []
        pct_obs_iwas = []
        lammax_obs_iwas = []
        specs = []

        """
        Calculate the exposure times and spectra in each bandpass for each
        star in biased sample
        """

        # Loop over stars in this sample
        for i in range(Ndraw):
            #print("HIP %i, %.2f pc, %s " %(hip[i], dist[i], stype[i]))

            # Set system parameters for this star
            self.prep_ith_star(i)

            # Calculate the time to observe the complete spectrum
            t_tots[i], tpbpc, spectrum, iwa = self.complete_spectrum_time()

            tpbpcs.append(tpbpc)
            pct_obs_iwas.append(iwa[0])
            specs.append(spectrum)

        # Calculate channel widths
        deltas = []
        for channel in CHANNELS:
            l = default_luvoir(channel=channel)
            deltas.append(l.lammax - l.lammin)
        self.deltas = np.array(deltas)

        # Calculate channel fractional completeness
        self.channel_weights = (self.deltas / np.sum(self.deltas))

        # Calculate completeness for each star in sample
        self.completeness = np.sum(np.array(pct_obs_iwas) * self.channel_weights, axis = 1)

        """
        Make a Lookup Table of Exposure times for each star in sample
        """

        tpbpcs_rect = []    # Time per bandpass
        tpcs_rect = []      # Time per channel

        # Loop over all the stars in sample
        for idrew in range(self.NBIAS):

            tpbpcs_rect.append([])
            tpcs_rect.append([])
            bp_names = []
            bp_chan = []

            # Loop over all the LUVOIR channels
            for ichan in range(len(CHANNELS)):

                tpcs_rect[idrew].append(0.0)

                # Loop over all the bands in this channel
                for iband in range(len(tpbpcs[0][ichan])):

                    bp_names.append("%s %i" %(CHANNELS[ichan], iband+1))
                    bp_chan.append(ichan)
                    tpbpcs_rect[idrew].append(tpbpcs[idrew][ichan][iband])
                    tpcs_rect[idrew][ichan] += tpbpcs[idrew][ichan][iband]

        # Make np arrays
        tpbpcs_rect = np.array(tpbpcs_rect)
        tpcs_rect = np.array(tpcs_rect)
        bp_names = np.array(bp_names)
        bp_chan = np.array(bp_chan)

        # Make infs --> nans
        infmask = ~np.isfinite(tpbpcs_rect)
        tpbpcs_rect[infmask] = np.nan
        infmask = ~np.isfinite(tpcs_rect)
        tpcs_rect[infmask] = np.nan

        # Set attributes
        self.tpbpcs_rect = tpbpcs_rect
        self.tpcs_rect = tpcs_rect
        self.bp_names = bp_names
        self.bp_chan = bp_chan

        """
        New completeness calculations
        """

        bandpasses = []

        # Loop over telescope channels
        for j, channel in enumerate(CHANNELS):

            # Channel dependent bandwidth?
            if type(self.bandwidth) is float:
                bandwidth = self.bandwidth
            else:
                assert len(self.bandwidth) == len(CHANNELS)
                bandwidth = self.bandwidth[j]

            # Get the channel specific telescope parameters
            luvoir = default_luvoir(channel=channel)
            self.cn.telescope = luvoir

            # Calculate the bandpass edges
            edges = calculate_bandpass_edges(luvoir.lammin, luvoir.lammax, bandwidth = bandwidth)

            # Calculate the number of bandpasses
            Nbands = len(edges) - 1

            # Loop over bandpasses
            for i in range(Nbands):

                # Get the max, min, and middle wavelenths for this bandpass
                lammin = edges[i]
                lammax = edges[i+1]

                bandpasses.append([lammin, lammax])

        bandpasses = np.array(bandpasses)
        lmin, lmax = np.min(np.hstack(bandpasses)), np.max(np.hstack(bandpasses))

        # Fractional completeness of each bandpass
        bp_frac = ((bandpasses[:,1] - bandpasses[:,0]) / (lmax - lmin)) / np.sum((bandpasses[:,1] - bandpasses[:,0]) / (lmax - lmin))

        # Completeness by target
        tot_completeness = np.sum(np.isfinite(self.tpbpcs_rect) * bp_frac, axis=1)

        # Fraction of stars in biased sample that can completely observe each bandpass
        frac_bias_bp = np.sum(np.isfinite(tpbpcs_rect)*1.0, axis=0) / self.NBIAS

        # Set attributes
        self.bandpasses = bandpasses
        self.bp_frac = bp_frac
        self.tot_completeness = tot_completeness
        self.frac_bias_bp = frac_bias_bp

        self._make_pandas_table()

        return

    def _make_pandas_table(self):
        """
        """

        # Make a pandas table for lookup
        data = np.vstack([self.biased_sample["hip"],
                  self.biased_sample["stype"],
                  self.biased_sample["dist"],
                  self.tpbpcs_rect.T,
                  self.tot_completeness])
        columns = np.hstack(["HIP", "type", "d [pc]", self.bp_names, "Spec. Completeness"])
        isort = np.argsort(self.tpbpcs_rect[:,6])
        self.exptime_table = pd.DataFrame(data[:, isort].T, columns=columns)

        return

    def run_hec_drm(self, Ndraw = 5, wantexp_days = 365., verbose = True,
                    iremove = [], wantSNR_grid = None):
        """
        Run the LUVOIR Habitable Exoplanet Characterization (HEC) Design Reference Mission (DRM).

        Parameters
        ----------
        Ndraw : int
            Number of stars drawn out of the total biased sample of habitable Earth-like candidates
        wantexp_days : float
            Number of days willing to spend on science time for this program
        wantSNR_grid : list or numpy.array
            Desired SNR in each band; calculated via scaling from original ``wantSNR_grid``
            (must satisfy: ``len(wantSNR_grid) == len(bp_chan)``)
        iremove : list
            Indices of bandpasses to remove
        verbose : bool
            Use print statements? Good for a single example

        Returns
        -------
        """

        # Construct mask for bands we're keeping
        val = []
        for i in range(len(self.bp_names)):
            if i in iremove:
                val.append(False)
            else:
                val.append(True)
        val = np.array(val)

        # Randomly draw stellar sample indices
        idraw = np.random.choice(np.arange(self.NBIAS), size=Ndraw, replace=False)

        # Scale SNRs if need be
        if wantSNR_grid is not None:
            SNRfactor = (wantSNR_grid / self.wantSNR)**2
        else:
            SNRfactor = np.ones(self.tpbpcs_rect.shape[1])

        # Get exptimes for each star drawn
        tpbpcs_draws = SNRfactor*self.tpbpcs_rect[idraw, :]

        t_sci = np.zeros(Ndraw) # was t_tot
        c_tot = np.zeros(Ndraw)
        t_ovr = np.zeros(Ndraw)

        # Loop over targets
        for i in range(Ndraw):

            # Science exposure times in each channel
            t_uv = np.nansum(tpbpcs_draws[i, val & (self.bp_chan == 0)])
            t_vis = np.nansum(tpbpcs_draws[i, val & (self.bp_chan == 1)])
            t_nir = np.nansum(tpbpcs_draws[i, val & (self.bp_chan == 2)])

            # Total exposure time
            t_sci[i] = apply_two_channels(np.array([t_uv, t_vis, t_nir]))

            # OVERHEADS
            ## 1 hour for slew + dynamic settle + thermal settle
            slew_settle_time = 1.0
            ## 0.6 for A (1.25 for B) hours for digging initial dark hole
            if self.architecture.lower().startswith("a"):
                initial_dark_hole_tax = 0.6
            elif self.architecture.lower().startswith("b"):
                initial_dark_hole_tax = 1.25
            else:
                print("`architecture` is unknown value in `run_hec_drm`")
                initial_dark_hole_tax = np.nan
            ## GUESS: apply initial dark hole tax to each bandpass (flat tax * number of bandpasses used)
            dark_hole_time = initial_dark_hole_tax * np.sum(np.isfinite(tpbpcs_draws[i, val]))
            ## 10% science time tax per iteration of wavefront control
            Nwfc = 1.0   # Assuming one iteration
            fwfc = 0.1 * Nwfc
            ## GUESS: apply WFC science tax to total science time after accounting for simultaneous observations in two channels
            wfc_time = fwfc * t_sci[i]
            ## Sum up the overheads
            t_ovr[i] = slew_settle_time + dark_hole_time + wfc_time

            # Completeness = initial completeness - fraction contributed from removed bands
            mask = np.isfinite(tpbpcs_draws[i, :]) & ~val
            c_tot[i] = self.tot_completeness[idraw[i]] - np.sum(self.bp_frac[mask])

            # Print?
            if verbose:
                print("HIP%s - %s - %.2fpc" %(self.biased_sample["hip"][idraw[i]], self.biased_sample["stype"][idraw[i]], self.biased_sample["dist"][idraw[i]]))
                print("    - %.1f%% Complete Spectrum : %.2f days" %(c_tot[i] * 100., t_sci[i] / 24.))
                print("    - UV Spectrum : %.2f days" %(t_uv / 24.))
                print("    - Optical Spectrum : %.2f days" %(t_vis / 24.))
                print("    - NIR Spectrum : %.2f days" %(t_nir / 24.))
                print("    - Overhead %.2f days" %(t_ovr[i] / 24.))
                #print("    - O2 0.76 um : %.2f days" %(tpmbs[idraw[i],0,0] / 24.))
                #print("    - O3 0.6 um : %.2f days" %(tpmbs[idraw[i],1,0] / 24.))

        # Sum science time and overheads
        t_tot = t_sci + t_ovr

        # Calculate total science exposure time for all Ndraw targets
        t_sci_sum = np.sum(t_sci)

        # Calculate total overhead time for all Ndraw targets
        t_ovr_sum = np.sum(t_ovr)

        # Calculate total science + overhead time for all Ndraw targets
        t_tot_sum = np.sum(t_tot)

        # Prioritize targets for fixed desired exposure time
        isort = np.argsort(t_tot)
        t_tot_cumsum = np.cumsum(t_tot[isort])
        viable = (t_tot_cumsum / 24.) < wantexp_days
        count_in_texp = np.sum(viable)
        if count_in_texp > 0:
            texp_for_count = t_tot_cumsum[viable][-1] / 24
        else:
            texp_for_count = 0

        if verbose:
            print("---------------------FINAL TALLY---------------------")
            print("%.2f yrs for %i target's complete spectra with overheads (SNR=%.1f)" %(t_tot_sum / (24. * 356.), Ndraw, self.wantSNR))
            print("%.2f yrs for %i target's complete spectra just science time (SNR=%.1f)" %(t_sci_sum / (24. * 356.), Ndraw, self.wantSNR))
            print("%.2f yrs for %i target's complete spectra just overheads (SNR=%.1f)" %(t_ovr_sum / (24. * 356.), Ndraw, self.wantSNR))
            print("%.2f yrs for %i target's UV spectra (SNR=%.1f)" %(np.nansum(tpbpcs_draws[:, val & (self.bp_chan == 0)]) / (24. * 356.), Ndraw, self.wantSNR))
            print("%.2f yrs for %i target's optical spectra (SNR=%.1f)" %(np.nansum(tpbpcs_draws[:, val & (self.bp_chan == 1)]) / (24. * 356.), Ndraw, self.wantSNR))
            print("%.2f yrs for %i target's NIR spectra (SNR=%.1f)" %(np.nansum(tpbpcs_draws[:, val & (self.bp_chan == 2)]) / (24. * 356.), Ndraw, self.wantSNR))
            #print("%.2f yrs for %i target's O2 at 0.76 um (SNR=%i)" %(np.nansum(tpmbs[idraw,0,0]) / (24. * 356.), Ndraw, wantSNR_bands))
            #print("%.2f yrs for %i target's O3 at 0.6 um (SNR=%i)" %(np.nansum(tpmbs[idraw,1,0]) / (24. * 356.), Ndraw, wantSNR_bands))
            print("%i spectra in %i days (%i desired for program)" %(count_in_texp, texp_for_count, wantexp_days))

        return t_tot[isort], count_in_texp, c_tot[isort], tpbpcs_draws[isort, :], t_sci[isort], t_ovr[isort]

    def plot_bp_exptimes(self, plot_spectrum = True, title = None, ylims = (1.0, 1e7),
                         cc = ["C0", "C2", "C3"], iremove = []):
        """
        Plot the exposure time per bandpass for each bandpass
        """

        # Reshape exposure times
        tmp = self.tpbpcs_rect.T

        # Calculate clean spectrum
        output = self.complete_spectrum_time()
        spectrum = output[2]

        fig, ax2 = plt.subplots(figsize = (16,5))

        if title is not None:
            ax2.set_title(title)

        icount = 0
        for ichan in range(len(CHANNELS)):

            data = []
            positions = []
            widths = []

            for j in range(len(self.bp_names[self.bp_chan == ichan])):

                nanmask = np.isfinite(tmp[icount,:])

                data.append(tmp[icount,nanmask])
                positions.append(np.mean(spectrum[0][icount]))
                widths.append(spectrum[0][icount][-1] - spectrum[0][icount][0] + np.mean(spectrum[1][icount][:]))
                color1 = cc[ichan]

                comp_str = "$%i \%%$" %(100.*self.frac_bias_bp[icount])
                comp_str2 = "$\mathbf{%i \%%}$" %(100.*self.frac_bias_bp[icount])
                #ax2.text(positions[j], np.median(tmp[icount,:]) + 5.*np.std(tmp[icount,:]), comp_str2,
                #         ha = "center", va = "top", fontsize = 12, color = "w")
                q_l, q_50, q_h, q_m, q_p = nsig_intervals(tmp[icount,nanmask], intvls=[0.05, 0.5, 0.97])
                ax2.text(positions[j], ylims[1], comp_str2,
                         ha = "center", va = "top", fontsize = 12, color = color1)

                #ax2.plot(self.bandpasses[icount], [q_50, q_50], color = color1, zorder = 120, ls = "dashed")

                icount += 1

            positions = np.array(positions)
            widths = np.array(widths)
            bp1 = ax2.boxplot(data, sym = '', widths = widths, showfliers = False,
                              boxprops = {"color" : color1, "alpha" : 0.5},
                              whiskerprops = {"color" : color1, "linewidth" : 2.0},
                              capprops = {"color" : color1, "linewidth" : 0.0},
                              medianprops = {"color" : "w", "linewidth" : 2.0},
                              patch_artist=True, positions = positions, whis = [5, 95]);

            for patch in bp1['boxes']:
                patch.set_facecolor(color1)

        if plot_spectrum:

            ax = ax2.twinx()
            ax2.set_zorder(100)
            ax2.patch.set_visible(False)

            ax.set_xlabel("Wavelength [$\mu$m]")
            ax.set_ylabel(r"Planet-Star Flux Ratio ($\times 10^{-10}$)", rotation = 270, labelpad = 25)
            for i in range(len(self.bp_names)):
                if i not in iremove:
                    pass
                    #ax.plot(spectrum[0][i], 1e10*spectrum[3][i], "o", ms = 4.0, alpha = 0.65, color = "w", zorder = 80)
                    #ax.errorbar(spectrum[0][i], 1e10*spectrum[3][i], yerr=1e10*spectrum[4][i], fmt = "o", ms = 2.0, alpha = 0.65, color = "k", zorder = 80)
                    #ax.axvspan(drmA.bandpasses[i][0], drmA.bandpasses[i][1], alpha = 0.2, color = cc[drmA.bp_chan[i]])

            self.cn.telescope.lammin = 0.2
            self.cn.telescope.lammax = 2.0
            self.cn.telescope.resolution = 140.
            # Re-do count rate calcs for true Earth spectrum
            self.cn.run_count_rates(AHR, LAMHR, FSTAR)
            l1, = ax.plot(self.cn.lam, 1e10*self.cn.Cratio, color = "purple", zorder = 0, lw = 4.0, alpha = 1.)
            l2, = ax.plot(self.cn.lam, 1e10*self.cn.Cratio, color = "w", zorder = 0, lw = 2.0, alpha = 0.65)
            ax.set_ylim(bottom=0.0)
            ax.legend([(l1, l2)], [("Modern Earth")], framealpha = 0.0)

            # Label Molecules
            ax.text(0.27, 1.55, "O$_3$", ha = "center", va = "center")
            ax.text(0.6, 1.25, "O$_3$", ha = "center", va = "center")
            ax.text(0.68, 1.35, "O$_2$", ha = "center", va = "center")
            ax.text(0.76, 1.45, "O$_2$", ha = "center", va = "center")
            ax.text(0.96, 1.45, "H$_2$O", ha = "center", va = "center")
            ax.text(1.15, 1.45, "H$_2$O", ha = "center", va = "center")
            ax.text(1.4, 1.45, "H$_2$O", ha = "center", va = "center")
            ax.text(1.9, 1.45, "H$_2$O", ha = "center", va = "center")
            ax.text(1.6, 1.25, "CO$_2$", ha = "center", va = "center")

        ax2.set_ylabel("Science Time [hrs]")
        #ax2.set_title(r"All %i targets (S/N$\approx$%i)" %(Ndraw, wantSNR))
        ax2.set_yscale("log")

        ax2.set_xlabel("Wavelength [$\mu$m]")
        ax2.set_ylim(bottom = ylims[0], top = ylims[1])

        ax2.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
        ax2.set_xticklabels(["$0.2$", "$0.4$", "$0.6$", "$0.8$", "$1.0$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2.0$"])
        ax2.set_xlim(0.1, 2.1)
        #ax2.set_xlim(0.4, 1.0)

        #fig.savefig("/Users/Jake/Dropbox/Astronomy/UW/Astrobio/Research Rotation/LUVOIR/figures/drm_bp10_science_time_%s.pdf" %drm.architecture, bbox_inches = "tight")

        return fig

    def plot_observed_spectrum(self, iremove = [], cc = ["C0", "C2", "C3"],
                               yloc = 1.8, plot_boxes = False):
        """
        Plot the observed spectrum
        """

        bp_names = self.bp_names

        # Set planet and star parameters for an Earth-Sun analog at 6pc
        self.cn.planet.distance = 6.0
        self.cn.planet.a = 1.0
        self.cn.star.Rs = 1.0
        self.cn.star.Teff = 5780.

        """
        wantSNR_grid = np.array([wantSNR for i in range(len(bp_names))])
        wantSNR_grid[0] = 1.0
        wantSNR_grid[1] = 1.0
        """

        output = self.complete_spectrum_time()
        spectrum = output[2]

        fig, ax = plt.subplots(figsize = (16,5))
        ax.set_xlabel("Wavelength [$\mu$m]")
        ax.set_ylabel(r"Planet-Star Flux Ratio ($\times 10^{-10}$)")
        for i in range(len(bp_names)):
            if i not in iremove:
                ax.plot(spectrum[0][i], 1e10*spectrum[3][i], "o", ms = 4.0, alpha = 0.65, color = "w", zorder = 80)
                ax.errorbar(spectrum[0][i], 1e10*spectrum[3][i], yerr=1e10*spectrum[4][i], fmt = "o", ms = 2.0, alpha = 0.65, color = "k", zorder = 80)
                ax.axvspan(self.bandpasses[i][0], self.bandpasses[i][1], alpha = 0.2, color = cc[self.bp_chan[i]])

        self.cn.telescope.lammin = 0.2
        self.cn.telescope.lammax = 2.0
        self.cn.telescope.resolution = 140.
        # Re-do count rate calcs for true Earth spectrum
        self.cn.run_count_rates(AHR, LAMHR, FSTAR)
        ax.plot(self.cn.lam, 1e10*self.cn.Cratio, color = "purple", zorder = 70, lw = 4.0, alpha = 1.)
        ax.plot(self.cn.lam, 1e10*self.cn.Cratio, color = "w", zorder = 70, lw = 2.0, alpha = 0.65)
        ax.set_ylim(bottom=0.0)

        # Label Molecules
        ax.text(0.27, 1.55, "O$_3$",  ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(0.6, 1.45, "O$_3$",   ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(0.69, 1.35, "O$_2$",  ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(0.76, 1.65, "O$_2$",  ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(0.96, 1.65, "H$_2$O", ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(1.15, 1.45, "H$_2$O", ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(1.4, 1.45, "H$_2$O",  ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(1.9, 1.25, "H$_2$O",  ha = "center", va = "center", color = "k", zorder = 130)
        ax.text(1.6, 1.25, "CO$_2$",  ha = "center", va = "center", color = "k", zorder = 130)

        lammin_inst = self.bandpasses[self.bp_chan == 0][0][0]
        lammax_inst = self.bandpasses[self.bp_chan == 0][-1][1]
        xloc = (lammax_inst + lammin_inst) / 2
        name = "UV"
        color = cc[0]
        bbox_fc = "w"
        ax.annotate(s='', xy=(lammin_inst,yloc), xytext=(lammax_inst,yloc), arrowprops=dict(arrowstyle='<->', color=color, lw = 2.0), zorder=2)
        ax.text(xloc, yloc, name, ha="center", va="bottom", color=color, zorder=99)#, bbox=dict(boxstyle="square", fc=bbox_fc, ec="none", zorder=2))

        lammin_inst = self.bandpasses[self.bp_chan == 1][0][0]
        lammax_inst = self.bandpasses[self.bp_chan == 1][-1][1]
        xloc = (lammax_inst + lammin_inst) / 2
        name = "visible"
        color = cc[1]
        bbox_fc = "w"
        ax.annotate(s='', xy=(lammin_inst,yloc), xytext=(lammax_inst,yloc), arrowprops=dict(arrowstyle='<->', color=color, lw = 2.0), zorder=2)
        ax.text(xloc, yloc, name, ha="center", va="bottom", color=color, zorder=99)#, bbox=dict(boxstyle="square", fc=bbox_fc, ec="none", zorder=2))

        lammin_inst = self.bandpasses[self.bp_chan == 2][0][0]
        lammax_inst = self.bandpasses[self.bp_chan == 2][-1][1]
        xloc = (lammax_inst + lammin_inst) / 2
        name = "NIR"
        color = cc[2]
        bbox_fc = "w"
        ax.annotate(s='', xy=(lammin_inst,yloc), xytext=(lammax_inst,yloc), arrowprops=dict(arrowstyle='<->', color=color, lw = 2.0), zorder=2)
        ax.text(xloc, yloc, name, ha="center", va="bottom", color=color, zorder=99)#, bbox=dict(boxstyle="square", fc=bbox_fc, ec="none", zorder=2))

        if plot_boxes:

            ax2 = ax.twinx()

            """
            Xdraw = len(tpbpcs_draws_tots[0][0])

            # Transform quantities for boxplot
            tmp = [np.zeros((len(tpbpcs_draws_tots[i]), Xdraw)) for i in range(len(spectroscopy.CHANNELS))]
            for i in range(Xdraw):
                for j in range(tp):
                    for k in range(len(tpbpcs_draws_tots[j])):
                        tmp[j][k,i] = tpbpcs_draws_tots[j][k][i]
            """

            icount = 0
            for ichan in range(len(spectroscopy.CHANNELS)):

                data = []
                positions = []
                widths = []

                for j in range(len(bp_names[bp_chan == ichan])):

                    data.append(tmp[icount,:])
                    positions.append(np.mean(spectrum[0][icount]))
                    widths.append(spectrum[0][icount][-1] - spectrum[0][icount][0] + np.mean(spectrum[1][icount][:]))
                    color1 = cc[ichan]

                    comp_str = "$%i \%%$" %(100.*frac_bias_bp[icount])
                    comp_str2 = "$\mathbf{%i \%%}$" %(100.*frac_bias_bp[icount])
                    #ax2.text(positions[j], np.median(tmp[icount,:]) + 5.*np.std(tmp[icount,:]), comp_str2,
                    #         ha = "center", va = "top", fontsize = 12, color = "w")
                    ax2.text(positions[j], np.median(tmp[icount,:]) + 5.*np.std(tmp[icount,:]), comp_str2,
                             ha = "center", va = "top", fontsize = 12, color = color1)

                    icount += 1

                positions = np.array(positions)
                widths = np.array(widths)
                bp1 = ax2.boxplot(data, sym = '', widths = widths, showfliers = False,
                                  boxprops = {"color" : color1, "alpha" : 0.5},
                                  whiskerprops = {"color" : color1, "linewidth" : 2.0},
                                  capprops = {"color" : color1, "linewidth" : 0.0},
                                  medianprops = {"color" : "w", "linewidth" : 2.0},
                                  patch_artist=True, positions = positions, whis = [5, 95]);

                for patch in bp1['boxes']:
                    patch.set_facecolor(color1)

            ax2.set_ylabel("Science Time [hrs]", labelpad = 22, rotation = 270)
            #ax2.set_title(r"All %i targets (S/N$\approx$%i)" %(Ndraw, wantSNR))
            ax2.set_yscale("log")

            ax2.set_xlabel("Wavelength [$\mu$m]")
            ax2.set_ylim(bottom = 0.0)

            ax2.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
            ax2.set_xticklabels([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
            ax2.set_xlim(0.1, 2.0)
            #ax2.set_xlim(0.4, 1.0)

        return fig, ax


################################################################################
# SUPPORT FUNCTIONS
################################################################################
def default_luvoir(architecture = "A", channel = "vis"):
    """
    Returns the :class:`coronagraph.Telescope` for the `architecture` and
    `channel` specified by the user.
    """

    telescope = cg.Telescope()

    # Set paramaters for Architecture A
    if architecture.lower().startswith("a"):
        telescope.diameter = 15.0
        telescope.contrast = 1e-10
        telescope.A_collect = 155.2
        telescope.diam_inscribed = 13.5
        telescope.Tsys = 270.
        telescope.OWA = 64.
        telescope.qe = 0.9 * 0.75   # Detector QE * charge transfer term
        if channel.lower() == "vis".lower():
            # Visible channel parameters:
            telescope.IWA = 3.5
            telescope.resolution = 140.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.515
            telescope.lammax = 1.030
            telescope.Rc = 1.3e-3     # Clock induced charge [counts/pixel/photon]
        elif channel.lower() == "UV".lower():
            # UV channel parameters:
            telescope.IWA = 4.0
            telescope.resolution = 7.
            telescope.throughput = 0.18
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.2
            telescope.lammax = 0.525
            telescope.Rc = 1.3e-3     # Clock induced charge [counts/pixel/photon]
        elif channel.lower() == "NIR".lower():
            # NIR channel parameters:
            telescope.IWA = 3.5
            telescope.resolution = 70.
            telescope.throughput = 0.18
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.0
            telescope.lammax = 2.0
            telescope.Rc = 0.0       # Clock induced charge [counts/pixel/photon]
        else:
            print("Unknown `channel`")
            return None

    # Set paramaters for Architecture B
    elif architecture.lower().startswith("b"):
        telescope.diameter = 8.0
        telescope.contrast = 1e-10
        telescope.A_collect = 43.4
        telescope.diam_inscribed = 6.7
        telescope.Tsys = 270.
        telescope.OWA = 64.
        telescope.qe = 0.9 * 0.75   # Detector QE * charge transfer term
        if channel.lower() == "vis".lower():
            # Visible channel parameters:
            telescope.IWA = 2.0
            telescope.resolution = 140.
            telescope.throughput = 0.48
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.515
            telescope.lammax = 1.030
            telescope.Rc = 1.3e-3     # Clock induced charge [counts/pixel/photon]
        elif channel.lower() == "UV".lower():
            # UV channel parameters:
            telescope.IWA = 4.0
            telescope.resolution = 7.
            telescope.throughput = 0.48
            telescope.darkcurrent = 3e-5
            telescope.readnoise = 0.0
            telescope.lammin = 0.200
            telescope.lammax = 0.525
            telescope.Rc = 1.3e-3     # Clock induced charge [counts/pixel/photon]
        elif channel.lower() == "NIR".lower():
            # NIR channel parameters:
            telescope.IWA = 2.0
            telescope.resolution = 70.
            telescope.throughput = 0.48
            telescope.darkcurrent = 2e-3
            telescope.readnoise = 2.5
            telescope.lammin = 1.00
            telescope.lammax = 2.00
            telescope.Rc = 0.0       # Clock induced charge [counts/pixel/photon]
        else:
            print("Unknown `channel`")
            return None
    else:
        print("Unknown `architecture`")
        return None

    # Set wavelength-dependent throughput for the optics
    tpath = os.path.join(HERE, "../inputs/optical_throughput.txt")
    data = np.genfromtxt(tpath, skip_header=1)
    midlamt = 1e-3 * data[:,2]
    Tput_optics = data[:,3]
    telescope.Tput_lam = (midlamt, Tput_optics)

    # Specify type of coronagraph:
    # set separation (lam/D) dependent throughput and contrast based on
    if "aplc" in architecture.lower():
        try:
            # Extract the int following the coronagraph name
            imask = int(architecture.lower().split("aplc")[-1][:-1])
        except ValueError:
            imask = 1
        # Read-in APLC files
        c_aplc = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/APLC_Contrast%i.txt" %imask), skiprows=1)
        t_aplc = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/APLC_Throughput%i.txt" %imask), skiprows=1)
        # Set quantities for coronagraph
        telescope.C_sep = (c_aplc[:,0], c_aplc[:,1])
        telescope.Tput_sep = (t_aplc[:,0], t_aplc[:,1])
    elif "dmvc" in architecture.lower():
        try:
            # Extract the int following the coronagraph name
            imask = int(architecture.lower().split("dmvc")[-1][:-1])
        except ValueError:
            imask = 1
        # Read-in DMVC files
        c_dmvc = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/DMVC_Contrast%s.txt" %imask), skiprows=1)
        t_dmvc = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/DMVC_Throughput%s.txt" %imask), skiprows=1)
        # Set quantities for coronagraph
        telescope.C_sep = (c_dmvc[:,0], c_dmvc[:,1])
        telescope.Tput_sep = (t_dmvc[:,0], t_dmvc[:,1])
    elif "piaa" in architecture.lower():
        try:
            # Extract the int following the coronagraph name
            imask = int(architecture.lower().split("piaa")[-1][:-1])
        except ValueError:
            imask = 1
        # Read-in PIAA files
        c_piaa = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/PIAA_Contrast%i.txt" %imask), skiprows=1)
        t_piaa = np.loadtxt(os.path.join(HERE, "../inputs/LUVOIR_coronagraphs/PIAA_Throughput%i.txt" %imask), skiprows=1)
        # Set quantities for coronagraph
        telescope.C_sep = (c_piaa[:,0], c_piaa[:,1])
        telescope.Tput_sep = (t_piaa[:,0], t_piaa[:,1])
    else:
        pass

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

    # Case: between type (e.g. use M3V for M3.5V)
    for i, st in enumerate(stypes):
        if s.split(".")[0]+"V" == st:
            if verbose: print("Using %s for %s" %(st, s))
            return i

    print("ERROR for `%s` spectral type not matched" %s)

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

    # CAN Set system number of exo-zodis here

    # Set stellar temperature
    cn.star.Teff = STARPROPS['temps'][imatch]

    # Set stellar radius
    cn.star.Rs = STARPROPS['rads'][imatch]

    # Calculate the Earth-equivalent insolation distance
    #a_eeq = np.sqrt(STARPROPS['lums'][imatch])

    # Calculate the semi-major axis for the inner edge (Kopparapu et al. 2013)
    a_in = calc_dist(STARPROPS['lums'][imatch], calc_seff(STARPROPS['temps'][imatch], S0_inner, inner_edge))

    # Set semi-major axis
    cn.planet.a = a_in

    # Set stellar spectrum based on type
    # Calculate stellar flux at TOA assuming a blackbody
    Fs = cg.noise_routines.Fstar(LAMHR, cn.star.Teff, cn.star.Rs, cn.planet.a, AU=True)
    #Fs = fstar

    # Run count rates
    cn.run_count_rates(AHR, LAMHR, Fs)

    return cn

def set_fiducial_earth(cn, distance = 5.0, a = 1.0, Rs = 1.0, Teff = 5780.):
    """
    Set :class:``coronagraph.CornagraphNoise`` parameters for an Earth-Sun
    analog
    """
    cn.planet.distance = distance
    cn.planet.a = a
    cn.star.Rs = Rs
    cn.star.Teff = Teff
    return cn

def calculate_bandpass_edges(lammin, lammax, bandwidth = 0.2):
    """
    Calculate the wavelengths of the edges of bandpasses given a minimum and
    maximum wavelength for the channel and the bandwidth.

    Parameters
    ----------
    lammin : float
        Minimum wavelength
    lammax : float
        Maximum wavelength
    bandwidth : float
        Fractional bandwidth :math:`\Delta \lambda / \lambda`

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

def complete_spectrum_time(cn, Ahr_flat = 0.2, wantSNR = 10.0, bandwidth = 0.2, architecture = "A",
                           plot = False, verbose = False):
    """
    Time for a complete spectrum

    Parameters
    ----------
    Ahr_flat : float
        Flat albedo spectrum
    wanrSNR : float or array-like
        Desired SNR on spectrum (per bandpass if array-like)
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

    # If the coronagraph model has already been run...
    if cn._computed:
        # Use the existing stellar flux
        fstar = cn.solhr
    else:
        # Otherwise use the solar flux
        fstar = FSTAR

    if plot: fig, ax = plt.subplots()

    cc = ["C0", "C2", "C3"]
    t_chan = np.zeros(len(CHANNELS))
    Nbands_per_chan = np.zeros(len(CHANNELS))
    t_per_band_per_chan = []
    full_lam = []
    full_dlam = []
    full_Cobs = []
    full_Cratio = []
    full_Csig = []
    pct_obs_iwa = []
    lammax_obs_iwa = []
    lam_extrema = []

    ibp = 0

    # Loop over telescope channels
    for j, channel in enumerate(CHANNELS):

        t_tmp = []

        # Get the channel specific telescope parameters
        luvoir = default_luvoir(channel=channel, architecture = architecture)
        cn.telescope = luvoir

        if verbose: print(channel, luvoir.lammin, luvoir.lammax)

        lam_extrema.append(luvoir.lammin)
        lam_extrema.append(luvoir.lammax)

        # Calculate the bandpass edges
        edges = calculate_bandpass_edges(luvoir.lammin, luvoir.lammax, bandwidth = bandwidth)

        # Calculate the number of bandpasses
        Nbands = len(edges) - 1
        Nbands_per_chan[j] = Nbands

        # Run count rates (necessary to generate new wavelength grid)
        #cn.run_count_rates(spectroscopy.AHR, spectroscopy.LAMHR, spectroscopy.FSTAR)

        # Get new wavelength grid
        #l_grid, dl_grid = get_lam_dlam(cn)

        # Calculate how much of the spectrum is observable
        cnc = copy.deepcopy(cn)
        cnc.run_count_rates(AHR, LAMHR, FSTAR)
        pct, lammax_obs = calc_observable_spectrum(cnc)
        pct_obs_iwa.append(pct)
        lammax_obs_iwa.append(lammax_obs)

        # Loop over bandpasses
        for i in range(Nbands):

            if (type(wantSNR) is float) or (type(wantSNR) is int):
                wSNR = wantSNR
            else:
                wSNR = wantSNR[ibp]

            # Get the max, min, and middle wavelenths for this bandpass
            lammin = edges[i]
            lammax = edges[i+1]
            lammid = 0.5*(lammax + lammin)

            # Set telescope wavelength range
            cn.telescope.lammin = lammin
            cn.telescope.lammax = lammax

            if channel == "UV":
                cn.telescope.lam = np.array([lammid])
                cn.telescope.dlam = np.array([lammax - lammin])

            # Set spectrum to use for exposure time calcs
            # Using flat spectrum so not biased by bottom of bands
            Ahr_flat  = Ahr_flat * np.ones(len(LAMHR))

            # Run count rates (necessary to generate new wavelength grid)
            cn.run_count_rates(Ahr_flat, LAMHR, fstar)

            # Calculate exposure times to wantSNR
            etimes = determine_exposure_time(cn, None, plot_snr_curves=False,
                        plot_spectrum=False, wantSNR=wSNR, ref_lam = lammid)
            t_ref_lam = etimes[-1]

            # Re-do count rate calcs for true Earth spectrum
            cn.run_count_rates(AHR, LAMHR, fstar)

            # Draw random samples of data for a plot
            cn.make_fake_data(texp=t_ref_lam)

            if verbose: print(lammid, t_ref_lam)

            # Plot
            if plot:
                ax.axvspan(lammin, lammax, alpha = 0.2, color = cc[j])
                #ax.plot(cn.lam, cn.Cratio, ls = "steps-mid", color = "grey", zorder = 100)
                ax.plot(cn.lam, cn.Cobs, "o", ms = 3.0, alpha = 1.0, color = "w", zorder = 70)
                ax.errorbar(cn.lam, cn.Cobs, yerr=cn.Csig, fmt = "o", ms = 2.0, alpha = 0.7, color = "k", zorder = 70)
                ax.set_xlabel("Wavelength [$\mu$m]")
                ax.set_ylabel("$F_p / F_s$")

            # Save values
            t_tmp.append(t_ref_lam)
            full_lam.append(cn.lam)
            full_dlam.append(cn.dlam)
            full_Cratio.append(cn.Cratio)
            full_Cobs.append(cn.Cobs)
            full_Csig.append(cn.Csig)

            # Add time
            if np.isfinite(t_ref_lam):
                t_chan[j] += t_ref_lam

            ibp += 1

        # Save tmp times per band
        t_per_band_per_chan.append(t_tmp)

    # Deal with the "two channels at a time" thing
    t_tot = apply_two_channels(t_chan)

    spectrum = (np.array(full_lam),
                np.array(full_dlam),
                np.array(full_Cratio),
                np.array(full_Cobs),
                np.array(full_Csig))
    iwa = (pct_obs_iwa, lammax_obs_iwa)

    if plot:
        lam_extrema = np.array(lam_extrema)
        cn.telescope.lammin = np.min(lam_extrema)
        cn.telescope.lammax = np.max(lam_extrema)
        cn.telescope.resolution = 140.
        # Re-do count rate calcs for true Earth spectrum
        cn.run_count_rates(AHR, LAMHR, FSTAR)
        ax.plot(cn.lam, cn.Cratio, color = "grey", zorder = 80, lw = 3.0)
        ax.plot(cn.lam, cn.Cratio, color = "w", zorder = 80, lw = 2.0)

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
            dcomp[i,j] = (specs[i][0][j][-1] - specs[i][0][j][0] + np.mean(specs[i][1][j][:])) / np.sum(deltas)

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
        dcomp[j] = (spec[0][j][-1] - spec[0][j][0] + np.mean(spec[1][j][:])) / np.sum(deltas)

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
    Based on d(exposure time)/d(completeness)
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

def remove_worst_texp_bandpass(tpbpc0, spec):
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

def get_iremove(drm, Nremove_uv = 0, Nremove_nir = 0):
    """
    Convenience function for getting bandpass indices
    to remove from exposure time estimates. Removes the
    `Nremove_uv` shortest UV bandpasses, and the `Nremove_nir` longest
    NIR bandpasses.

    Parameters
    ----------
    drm : `HECDRM`
        DRM object
    Nremove_uv : int
    Nremove_nir : int
    """
    iremove = []

    ibp = np.arange(len(drm.bp_names))
    iremove.append(ibp[drm.bp_chan == 0][:Nremove_uv])
    if Nremove_nir > 0:
        iremove.append(ibp[drm.bp_chan == 2][-Nremove_nir:])
    iremove = np.hstack(iremove)

    return iremove

if __name__ == "__main__":

    pass
