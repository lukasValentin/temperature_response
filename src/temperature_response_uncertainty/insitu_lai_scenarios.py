"""
Assume that the in-situ LAI values have some uncertainty
because of random measurement errors.
"""

import numpy as np
import pandas as pd

from pathlib import Path

np.random.seed(42)


def loop_sites(
        in_situ_lai_dir: Path,
        n_sim: int,
        noise_level: float
) -> None:
    """
    Loop over all sites and add noise to the in-situ LAI time series.

    Parameters
    ----------
    in_situ_lai_dir : Path
        Directory with in-situ LAI time series.
    n_sim : int
        Number of simulations.
    noise_level : float
        Noise level in percent.
    """
    # loop over sites. These start with 'DE' or 'CH'
    for site in in_situ_lai_dir.glob('*'):
        if site.name.split('_')[0] in ['DE', 'CH']:
            # read the LAI data from csv
            for fpath_lai in site.glob('LAI_*_Raw-Data.csv'):

                df = pd.read_csv(fpath_lai)
                # output directory for the simulations
                output_dir = site.joinpath(fpath_lai.stem)
                output_dir.mkdir(parents=True, exist_ok=True)

                # loop over simulations
                for i in range(n_sim):
                    _df = df.copy()
                    output_dir_simulation = output_dir.joinpath(
                        f'simulation_{i}')
                    output_dir_simulation.mkdir(parents=True, exist_ok=True)

                    # add Gaussian noise to the LAI time series
                    _df['LAI_value'] = _df['LAI_value'] + \
                        np.random.normal(
                            loc=0,
                            scale=noise_level / 100 * _df['LAI_value'])

                    # write the noisy LAI time series to csv
                    fpath_lai_noisy = output_dir_simulation.joinpath(
                        fpath_lai.name)
                    _df.to_csv(fpath_lai_noisy, index=False)


if __name__ == '__main__':

    # directory with in-situ LAI time series
    in_situ_lai_dir = Path('results/dose_reponse_in-situ')

    # number of simulations
    n_sim = 1000
    # noise level (in percent)
    noise_level = 5  # percent

    loop_sites(
        in_situ_lai_dir=in_situ_lai_dir,
        n_sim=n_sim,
        noise_level=noise_level)
