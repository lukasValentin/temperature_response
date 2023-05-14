"""
Relationships between traits at the Strickhof and
the Witzwil test site over multiple years.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

plt.style.use('bmh')


def get_parcel_names(test_pixels_dir: Path) -> list:
    """
    Get the names of available parcels.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    return : list
        List of parcel names.
    """
    return [parcel.name.split('_')[1] for parcel
            in test_pixels_dir.iterdir()]


def get_parcel_data(test_pixels_dir: Path, parcel_name: str) -> pd.DataFrame:
    """
    Get the data of a specific parcel.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    parcel_name : str
        Name of the parcel.
    return : pd.DataFrame
        Data of the parcel.
    """
    res_parcel = []
    for parcel in test_pixels_dir.iterdir():
        if parcel_name == parcel.name.split('_')[1]:
            # read LAI and CCC data
            df_lai = pd.read_csv(parcel.joinpath('raw_lai_values.csv'))
            df_lai.time = pd.to_datetime(
                df_lai.time, format='%Y-%m-%d %H:%M:%S', utc=True)
            df_ccc = pd.read_csv(parcel.joinpath('raw_ccc_values.csv'))
            df_ccc.time = pd.to_datetime(
                df_ccc.time, format='%Y-%m-%d %H:%M:%S', utc=True)
            df = pd.merge(df_lai, df_ccc, on=['time', 'x', 'y'])
            # calculate CAB
            df['cab'] = df['ccc'] / df['lai'] * 100
            res_parcel.append(df)

    return pd.concat(res_parcel)


def main(test_pixels_dir: Path):
    """
    Get the relationships between traits.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    """
    # get the parcel names
    parcel_names = np.unique(get_parcel_names(test_pixels_dir))

    # loop over parcels and extract the data
    for parcel_name in parcel_names:
        df = get_parcel_data(test_pixels_dir, parcel_name)
        # check how many years are in the data. Some parcels
        # have 2 growing seasons (i.e., 4 years)
        df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S', utc=True)
        years = np.sort(np.unique(df.time.dt.year))

        for idx in range(0, len(years), 2):
            # get the data for the current year
            df_year = df[(df.time.dt.year == years[idx]) |
                         (df.time.dt.year == years[idx+1])].copy()
            # calculate the mean values of lai, ccc, and cab for those
            # records that have the same time stamp and x, y coordinates
            # but different pheno_phase values
            df_year = df_year.groupby(['time', 'x', 'y']).mean().reset_index()

            # plot the relationships
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
            # CCC - GLAI relationship
            ax[0].scatter(df_year['lai'], df_year['ccc'], s=1)
            ax[0].set_xlabel(r'GLAI [$m^2$ $m^{-2}$]')
            ax[0].set_ylabel(r'CCC [$g$ $m^{-2}$]')
            ax[0].set_xlim([0, 8])
            ax[0].set_ylim([0, 4])

            # CAB - GLAI relationship
            ax[1].scatter(df_year['lai'], df_year['cab'], s=1)
            ax[1].set_xlabel(r'GLAI [$m^2$ $m^{-2}$]')
            ax[1].set_ylabel(r'Cab [$\mu g$ $cm^{-2}$]')
            ax[0].set_xlim([0, 8])
            ax[0].set_ylim([0, 80])

            plt.suptitle(f'Parcel {parcel_name} - {years[idx]}-{years[idx+1]}')

            plt.show()


if __name__ == '__main__':

    test_pixels_dir = Path('results/test_sites_pixel_ts')
    main(test_pixels_dir)
