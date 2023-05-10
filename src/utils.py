'''
Some useful functions for preparing data
'''

import pandas as pd

from pathlib import Path


def from_meteoswiss(fpath: Path) -> pd.DataFrame:
    """
    read data from MeteoSwiss and convert it into CSV

    :param fpath:
        original data as delivered from MeteoSwiss (IDAWEB)
    :returns:
        DataFrame with internally used conventions
    """
    df = pd.read_csv(fpath, sep=';')
    df.rename(columns={'tre200h0': 'T_mean'}, inplace=True)
    df.drop(columns=['stn'], inplace=True)
    df['time'] = pd.to_datetime(df.time, format='%Y%m%d%H')
    return df


if __name__ == '__main__':

    fpath = Path('data/Meteo/order111046/order_111046_data.txt')
    df_meteo = from_meteoswiss(fpath=fpath)
    # save data
    fpath_out = fpath.parent.parent.joinpath(
        'SwissFutureFarm_Meteo_hourly.csv')
    df_meteo.to_csv(fpath_out)
