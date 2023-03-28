'''
Created on Mar 28, 2023

@author: graflu
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

plt.style.use('bmh')
mpl.rc('font', size=18)

if __name__ == '__main__':

    fpath_lai_data = Path('../../results/dose_reponse_in-situ/DE_MNI/MNI_LAI_WW_Lukas.xlsx')
    lai_data = pd.read_excel(fpath_lai_data)
    lai_data.date = pd.to_datetime(lai_data.date, format='%Y-%m-%d')

    # extract greenLAI and save as CSV
    fpath_csv = fpath_lai_data.parent.joinpath('LAI_MNI_Raw-Data.csv')
    lai_data.dropna(subset=['greenLAI'], inplace=True)
    lai_data.rename(columns={'greenLAI': 'LAI_value', 'phenology [BBCH]': 'BBCH'}, inplace=True)
    lai_data[['date', 'LAI_value', 'BBCH']].to_csv(fpath_csv, index=False)

    # plot LAI
    f, ax = plt.subplots(figsize=(10,8))
    ax.plot(lai_data.date, lai_data.LAI_value, marker='x')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'In-Situ Green Leaf Area Index [$m^2$ $m^{-2}$]')
    fname_plt = fpath_lai_data.parent.joinpath('LAI_MNI_Raw-Data_plot.png')
    f.savefig(fname_plt, bbox_inches='tight')
    plt.close(f)

    # get date range
    # min_date = lai_data.date.min()
    # max_date = lai_data.date.max()