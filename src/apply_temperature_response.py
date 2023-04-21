'''
Created on Apr 21, 2023

@author: graflu
'''

import pandas as pd

from pathlib import Path

def apply_temperature_response(parcel_lai_dir: Path):
    """
    """
    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):
        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        # meteorological data
        fpath_meteo = parcel_dir.joinpath('hourly_mean_temperature.csv')
        meteo = pd.read_csv(fpath_meteo)

        # loop over single pixels
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):
            # pixel_coords are the coordinates of the pixel
            pixel_coords
            # the actual lai data. The meteo data can be joined on the time column
            lai_pixel_ts

if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('./results/test_sites_pixel_ts')

    apply_temperature_response(parcel_lai_dir=parcel_lai_dir)
