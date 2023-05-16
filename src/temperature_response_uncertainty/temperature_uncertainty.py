"""
Add random uncertainty to temperature readings required
for the temperature response function.
"""

import numpy as np


def add_noise_to_temperature(
        temperature: np.ndarray,
        noise_level: float
) -> np.ndarray:
    """
    Add Gaussian noise to temperature readings.

    Parameters
    ----------
    temperature : np.ndarray
        Temperature time series.
    noise_level : float
        Noise level in percent.

    Returns
    -------
    np.ndarray
        Temperature time series with added noise.
    """
    return temperature + np.random.normal(
        loc=0,
        scale=abs(noise_level / 100 * temperature)
    )
