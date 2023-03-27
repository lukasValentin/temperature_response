'''
Prepare LAI data collected by Samuel Wildhaber for his
Master thesis
'''

import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from datetime import datetime
from pathlib import Path

plt.style.use('bmh')
mpl.rc('font', size=18)

if __name__ == '__main__':

    data_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/MA_Supervision/22_Samuel-Wildhaber/LAI_analysis_BW/data')
    fpath_glai = data_dir.joinpath('LAI_data_BW_final.gpkg')
    glai = gpd.read_file(fpath_glai)
    glai.date = pd.to_datetime(glai.date)

    out_dir = Path('../../results/dose_reponse_in-situ')
    out_dir.mkdir(exist_ok=True)
    out_dir_bw = out_dir.joinpath('CH_Bramenwies')
    out_dir_bw.mkdir(exist_ok=True)

    # plot data
    f, ax = plt.subplots(figsize=(16,12))
    for point_id, point_df in glai.groupby(by='Point_ID'):
        ax.plot(point_df.date, point_df.LAI_value, label=f'Point {point_id}', marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'In-Situ Green Leaf Area Index [$m^2$ $m^{-2}$]')
    ax.legend(loc='upper left', ncol=2)
    fname_plt = out_dir_bw.joinpath('LAI_BW_Raw-Data_plot.png')
    f.savefig(fname_plt, bbox_inches='tight')

    # save data
    glai.to_csv(out_dir_bw.joinpath('LAI_BW_Raw-Data.csv'), index=False)

    # extract meteo data from sowing to harvest
    sowing_date = datetime(2021,10,31)
    harvest_date = datetime(2022,7,25)
    fpath_meteo = Path('../../data/Meteo/Strickhof_Meteo_hourly.csv')
    meteo = pd.read_csv(fpath_meteo)
    meteo.time = pd.to_datetime(meteo.time, format='%d.%m.%Y %H:%M')
    meteo.index = meteo.time
    meteo_bw = meteo[sowing_date:harvest_date]
    meteo_bw.to_csv(out_dir_bw.joinpath('Meteo_BW.csv'), index=False)
    