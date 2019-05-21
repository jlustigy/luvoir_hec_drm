"""
luvoir_char.py
--------------
Code to select stars with candidate exoEarths for characterization
from 2-year survey with LUVOIR
"""

import numpy as np
import pandas as pd

__all__ = ["gen_candidate_catalog"]

def gen_candidate_catalog(path, seed = None, return_full_cat = False):
    """
    Read in file of observations from Chris Stark's LUVOIR DRM and use it to
    calculate the biased sample catalog of exo-Earth candidate targets for
    the LUVOIR habitable exoplanet characterization DRM.

    Parameters
    ----------
    path : str
        Path to nominal occurrence rates file
    seed : int or NoneType
        Seed for random number generator. Defaults to `None` for true randomness.
        Set to integer for repeatable randomness.
    return_full_cat : bool
        Set to return the entire target catalog without performing q biased
        draw from the exoEarth candidate yields.

    Returns
    -------
    dic : dict
        Dictionary with result:
            `"hip"` : Stellar HIP numbers
            `"dist"` : Stellar distances
            `"stype"` : Stellar Types
            `"nez"` : Exo-zodi level
            `"total count"` : Total targets in sample
            `"total deep time"` : Total deep time (days)

    """

    # Use pandas to read Chris Stark's observation CSV
    df = pd.read_csv(path)

    # Number of observations in CSV
    obs = len(df["HIP"])

    # Calculate the actual spectral characterization time for exoEarths
    spec_time = df["Total Spec Char Time (days)"].values / df["Total EEC Yield"].values

    """
    # Find "brightest" targets by sorting spect time
    ibrightest = np.argsort(spec_time)

    if architecture == "A":
        # Select all targets for LUVOIR-A
        iuse = ibrightest
    elif architecture == "B":
        # Select the brightest 3/5 of the targets for LUVOIR-B
        iuse = ibrightest[0:int(obs*(3./5.))]
    else:
        print("ERROR: Unknown `architecture` entered")
    """

    # Number of observations to actually use
    obs = len(df["HIP"])

    # Prep lists to append to
    deep_time_arr = []
    star_arr = []
    dist_arr = []
    type_arr = []
    nezodi_arr = []
    eec_yield = []

    # Set RNG seed
    np.random.seed(seed)

    # Draw random numbers from a uniform distribution between 0-1
    rands = np.random.rand(obs)

    # Start a counting variable
    count = 0

    # Loop over all targets in catalog
    for i in range(obs):

        # Get random number from a uniform distribution between 0-1
        rand = rands[i]

        # If the staistical yield exceeds the random number:
        if (df["Total EEC Yield"].iloc[i] >= rand) or return_full_cat:

            # Append quantities to lists
            deep_time_arr.append(spec_time[i])
            star_arr.append(df["HIP"].iloc[i])
            dist_arr.append(df["dist (pc)"].iloc[i])
            type_arr.append(df["Type"].iloc[i])
            nezodi_arr.append(df["nexozodis (zodis)"].iloc[i])
            eec_yield.append(df["Total EEC Yield"].iloc[i])

            # Increment count
            count += 1

    # Convert lists to arrays
    deep_time_arr = np.array(deep_time_arr)
    star_arr = np.array(star_arr)
    dist_arr = np.array(dist_arr)
    type_arr = np.array(type_arr)
    nezodi_arr = np.array(nezodi_arr)
    eec_yield = np.array(eec_yield)

    # Total results
    tot_deep_time_arr = np.sum(deep_time_arr)
    tot_count_arr = count

    # Construct dictionary to return
    dic = {
        "hip" : star_arr,
        "dist" : dist_arr,
        "stype" : type_arr,
        "nez" : nezodi_arr,
        "total count" : count,
        "total deep time" : tot_deep_time_arr,
        "eec yield" : eec_yield
    }

    return dic
