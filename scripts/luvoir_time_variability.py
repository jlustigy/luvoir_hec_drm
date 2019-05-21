"""
luvoir_time_variability.py
==========================
Methods for calculating time variability yields for the habitable exoplanet
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
import spectroscopy
sys.path.insert(1, "../../exomapping/notebooks")
import ForwardMethods
from ForwardMethods import EarthModel, DataGenerator, bandwidth
from TimeTurner import TimeTurner

NCPU = multiprocessing.cpu_count()
HERE = os.path.abspath(os.path.split(__file__)[0])

__all__ = ["observe_earth_time_variability"]

def observe_earth_time_variability(drmx, Ntexp, istart, ibp, which_earth = "True_Earth",
                                   which_phase = 90.):
    """
    Make a mock observation of Earth's spectrum as a function of time that is consistent
    with the LUVOIR exoEarth DRM.

    Parameters
    ----------
    drmx : HEC_DRM
        DRM object
    Ntexp : int
        Number of consecutive hours to exposure
    istart : int
        Starting hour index
    ibp : int
        Bandpass Index
    which_earth : str
        Select from multiple VPL Earth Models:
            1. "True_Earth"
            2. "Cloudless"
            3. "Cirrus"
            4. "Land_Switch"
    which_phase : int or float
        Select the planet phase:
            1. 45 (gibbous)
            2. 90 (quadrature)
            3. 135 (crescent)

    Returns
    -------
    spectrum : dict
        "lam" : wavelength grid at native resolution
        "obs" : observations at native resolution
        "err" : errors at native resolution
        "t0" : observation time stamps
        "dt0" : observation time durations
    lightcurve :
        "time" : time grid for lightcurve
        "obs" : observations at native temporal sampling
        "err" : errors at native temporal sampling
        "bp_names" : bandpass names for each observation
        "lam0" : wavelength for lightcurve
        "dlam0" : wavelength width for lightcurve
    twod :
        "time" : native temporal sampling
        "lam" : native wavelength grid
        "obs" : 2D observations at native temporal and spectral sampling
        "err" : 2D observational errors at native temporal and spectral sampling
    """

    # Setup specific telescope
    channel = spectroscopy.CHANNELS[drmx.bp_chan[ibp]]
    telescope = spectroscopy.default_luvoir(architecture=drmx.architecture, channel=channel)
    telescope.lammin = drmx.bandpasses[ibp][0]
    telescope.lammax = drmx.bandpasses[ibp][1]

    # Set number of consecutive exposures
    #N = 100
    N =  Ntexp

    # Set the integration time per exposure
    iN = 1

    # Set number of neighboring pixels to coadd (int or "all")
    iM = "all"

    # Instantiate DataGenerator object with EarthModel data
    data = DataGenerator(earth_model = EarthModel())

    # Select the Earth Model dataset with HDF5 file
    data.select_dataset(which_earth = which_earth, which_phase = which_phase)

    # Set default telescope, planet and star
    #data.set_default_observing_system()
    data.telescope = telescope
    data.planet = drmx.cn.planet
    data.star = drmx.cn.star

    # Calculate wavelength bin for spectral coadd
    lamlr = np.array([0.5*(telescope.lammin - telescope.lammin/telescope.resolution + telescope.lammax + telescope.lammax/telescope.resolution)])
    dlamlr = np.array([(telescope.lammax + telescope.lammax/telescope.resolution) - (telescope.lammin - telescope.lammin/telescope.resolution)])

    x, y, yerr = [], [], []
    t2, x2, y2, yerr2 = [], [], [], []

    # Loop over time increments (hours)
    for i in range(N):

        # Get snapshot of Earth data
        data.create_mock_dataset(istart = istart+i, N = 1, iN = 1, iM = 1,
                                 imod = data.earth_model.hfile["True_Earth/geom_equatorial_90deg/reflectance/albedo"].shape[0])

        # Parse data
        t0 = data.times + istart + i
        x0 = data.lam[0,:]
        y0 = data.obs[0,:]
        yerr0 = data.sig[0,:]

        # Save 2D dataset
        t2.append(t0)
        x2.append(x0)
        y2.append(y0)
        yerr2.append(yerr0)

        # Coadd spectral pixels into lightcurve
        mf = np.isfinite(y0)
        specLR, errLR = cg.downbin_spec_err(y0[mf], yerr0[mf], x0[mf], lamlr, dlam = dlamlr)

        # Save lightcurve quantities
        x.append(t0)
        y.append(specLR)
        yerr.append(errLR)

    # Final lightcurve quantities
    t = np.hstack(x)
    y = np.hstack(y)
    dy = np.hstack(yerr)
    bps = np.array([drmx.bp_names[ibp] for _ in range(N)])
    lightcurve = {"time" : t,
                  "obs" : y,
                  "err" : dy,
                  "bp_names" : bps,
                  "lam0" : lamlr,
                  "dlam0" : dlamlr}

    # Final 2D dataset
    t2, x2, y2, yerr2 = np.hstack(t2), np.array(x2), np.array(y2), np.array(yerr2)
    twod = {"time" : t2,
            "lam" : x2,
            "obs" : y2,
            "err" : yerr2}

    # Create giant time window for giant coadd
    tlr = np.array([0.5*(t2[-1] + t2[0])])
    dtlr = np.array([(t2[-1] - t2[0])])

    # Loop over spectral elements
    tmpy = []
    tmpyerr = []
    for i in range(len(x2[0])):

        # Coadd in time for deep spectra
        specLR, errLR = cg.downbin_spec_err(y2[:,i], yerr2[:,i], t2, tlr, dlam = dtlr)
        tmpy.append(specLR)
        tmpyerr.append(errLR)

    # Final spectra quantites
    spectrum = {"lam" : x2[0],
                "obs" : np.array(tmpy),
                "err" : np.array(tmpyerr),
                "t0" : tlr,
                "dt0" : dtlr}

    return spectrum, lightcurve, twod
