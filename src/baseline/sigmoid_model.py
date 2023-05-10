"""
Baseline using a sigmoid model fitting on the raw LAI values.
The baseline model is only applied on the validation set.

@author: Lukas Valentin Graf 
"""

import numpy as np
import pandas as pd

from pathlib import Path
from scipy.optimize import curve_fit


def sigmoid(
        x: np.ndarray,
        L: float = 1,
        k: float = 1,
        x0: float = 0,
        b: float = 0
) -> np.ndarray:
    """
    Sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input array.
     L : float
        scales the output range.
    k : float
        scales the input range.
    x0 : float
        is the sigmoid's midpoint.
    b : float
        is the bias.

    Returns
    -------
    np.ndarray
        Sigmoid of the input array.
    """
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def fit_sigmoid(x: np.ndarray, y: np.ndarray):
    """
    Fit a sigmoid function to the data.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.

    Returns
    -------
    np.ndarray
        Fitted sigmoid function.
    """
    # mandatory initial guess
    p0 = [max(y), np.median(x), 1, min(y)]
    popt, _ = curve_fit(sigmoid, x, y, p0, method='lm')
    return popt


def loop_pixels(parcel_lai_dir: Path):
    """
    Loop over pixels and apply the sigmoid model.

    Parameters
    ----------
    parcel_lai_dir : Path
        Directory with parcel LAI time series.
    """
    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):
        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        # loop over single pixels
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):
            lai_pixel_ts

            # plot the data
            # import matplotlib.pyplot as plt
            # plt.plot(lai_pixel_ts['time'], lai_pixel_ts['lai'], 'o')


if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('./results/test_sites_pixel_ts')

    loop_pixels(parcel_lai_dir=parcel_lai_dir)
