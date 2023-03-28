'''
Prepare data from MNI campaign

@author: Lukas Valentin Graf
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

plt.style.use('bmh')
mpl.rc('font', size=18)

if __name__ == '__main__':

    # prepare meteo data
    meteo_dir = Path('../../results/dose_reponse_in-situ/DE_MNI/DWD')
    fpath_meteo_akt = meteo_dir.joinpath('stundenwerte_TU_01262_akt').joinpath('produkt_tu_stunde_20210923_20230326_01262.txt')
    meteo_akt = pd.read_csv(fpath_meteo_akt, sep=';')
    fpath_meteo_hist = meteo_dir.joinpath('stundenwerte_TU_01262_19920517_20211231_hist').joinpath('produkt_tu_stunde_19920517_20211231_01262.txt')
    meteo_hist = pd.read_csv(fpath_meteo_hist, sep=';')
    meteo = pd.concat([meteo_hist, meteo_akt])
    meteo.MESS_DATUM = pd.to_datetime(meteo.MESS_DATUM, format='%Y%m%d%H')
    meteo.rename(columns={'MESS_DATUM': 'time', 'TT_TU': 'T_mean'}, inplace=True)
    meteo = meteo[['time', 'T_mean']].copy()
    meteo.index = meteo.time

    # prepare LAI data
    fpath_lai_data = Path('../../results/dose_reponse_in-situ/DE_MNI/MNI_LAI_WW_Lukas.xlsx')
    lai_data = pd.read_excel(fpath_lai_data)
    lai_data.date = pd.to_datetime(lai_data.date, format='%Y-%m-%d')

    # extract greenLAI
    lai_data.dropna(subset=['greenLAI'], inplace=True)
    lai_data.rename(columns={'greenLAI': 'LAI_value', 'phenology [BBCH]': 'BBCH'}, inplace=True)

    # plot LAI
    f, ax = plt.subplots(figsize=(10,8))
    ax.plot(lai_data.date, lai_data.LAI_value, marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'In-Situ Green Leaf Area Index [$m^2$ $m^{-2}$]')
    fname_plt = fpath_lai_data.parent.joinpath('LAI_MNI_Raw-Data_plot.png')
    f.savefig(fname_plt, bbox_inches='tight')
    plt.close(f)

    # split data by year to generate a single time series per growing season
    lai_data['year'] = lai_data.date.dt.year

    for year, lai_annual in lai_data.groupby('year'):
        fpath_annual = fpath_lai_data.parent.joinpath(f'LAI_MNI{year}_Raw-Data.csv')
        lai_annual[['date', 'LAI_value', 'BBCH']].to_csv(fpath_annual, index=False)
        min_date = lai_annual.date.min()
        max_date = lai_annual.date.max()
        # clip meteo data
        meteo_annual = meteo[min_date:max_date+pd.Timedelta('1d')]
        fpath_meteo_annual = fpath_lai_data.parent.joinpath(f'Meteo_MNI{year}.csv')
        meteo_annual.to_csv(fpath_meteo_annual, index=False)
