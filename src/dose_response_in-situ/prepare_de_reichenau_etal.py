'''
Prepare and extract data collected by Reichenau et al. (2020)
Paper: https://essd.copernicus.org/articles/12/2333/2020/
Data: https://www.tr32db.uni-koeln.de/data.php?dataID=1889
'''

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

plt.style.use('bmh')
mpl.rc('font', size=18)

def prepare_data(fpath_dataset: Path, out_dir: Path):
    """
    """
    # loop over locations
    for location_dir in fpath_dataset.glob('*'):
        location = location_dir.name
        # check the management files first to identify winter wheat samples
        management_dir = location_dir.joinpath('management')
        # check for winter wheat (WW)
        for ww_file in management_dir.glob('management_*WW*.csv'):
            # read management file to get the sowing and harvest date
            with open(ww_file, 'r') as src:
                lines = src.read().split('\n')
            # sowing date is more important than knowing the harvest date
            try:
                sowing = [x for x in lines if 'sowing' in x.lower()][0]
            except IndexError:
                print(f'{location} - no sowing date found -> skipping')

            ww_dataset = ww_file.name.split('.')[0].split('_')[-1]
            # extract trait data from vegetation file
            veg_file = next(location_dir.glob('vegetation_*.csv'))
            veg_df = pd.read_csv(veg_file, sep=';')
            # subset the current WW dataset
            veg_df_ww_dataset = veg_df[veg_df.dataset == ww_dataset].copy()
            veg_df_ww_dataset.date = pd.to_datetime(veg_df_ww_dataset.date).dt.tz_localize('CET')
            veg_df_ww_dataset.LAI_green = veg_df_ww_dataset.LAI_green.astype(float)
            veg_df_ww_dataset.sort_values(by='date', inplace=True)
            # aggregate values per date to represent the field parcel
            glai_agg = veg_df_ww_dataset[['date', 'LAI_green', 'bbch', 'canopy_height']].groupby(by=['date']).mean()
            glai_agg['date'] = glai_agg.index

            # plot GLAI
            f, ax = plt.subplots(figsize=(16,12))
            ax.plot(glai_agg.LAI_green, marker='x')
            ax.set_xlabel('Time')
            ax.set_ylabel(r'In-Situ Green Leaf Area Index [$m^2$ $m^{-2}$]')
            ax.set_title(f'{location} {ww_dataset}')
            f.savefig(out_dir.joinpath(f'LAI_{ww_dataset}.png'), bbox_inches='tight')
            plt.close(f)

            # get the corresponding meteo data
            # special case Merken site: it is suggested to use Selhausen weather station
            if location == 'merken':
                meteo_file = next(location_dir.parent.joinpath('selhausen').glob('meteo_*.csv'))
            else:
                meteo_file = next(location_dir.glob('meteo_*.csv'))
            _meteo = pd.read_csv(meteo_file, sep=';')
            # drop the line with the units
            meteo = _meteo.iloc[1:,:]
            # convert timestamps to Central European Time
            meteo.index = pd.to_datetime(meteo['Date & Time']).dt.tz_localize('utc').dt.tz_convert('CET')
            # extract data between sowing and harvest date
            if sowing == '# sowing date estimated; was between 2011-10-24 and 2011-11-03':
                sowing = pd.to_datetime('2011-10-31').tz_localize('CET')
            else:
                try:
                    sowing_date = pd.to_datetime(sowing.split(';')[0]).tz_localize('CET')
                except Exception:
                    print('aarg')
            last_date = veg_df_ww_dataset.date.iloc[-1]
            meteo_dataset = meteo[sowing_date:last_date][['Date & Time', 'AirTemp', 'Precip']].copy()
            meteo_dataset.rename(
                columns={'Date & Time': 'time', 'AirTemp': 'T_mean', 'Precip': 'precip_tot'},
                inplace=True
            )
            # convert temperature from K to deg C
            meteo_dataset['T_mean'] = meteo_dataset['T_mean'].astype(float) - 273.15
            # save Meteo data
            meteo_dataset.to_csv(
                out_dir.joinpath(f'Meteo_{ww_dataset}.csv'),
                index=False
            )
            
            # save GLAI data
            glai_agg.rename(
                columns={'LAI_green': 'LAI_value', 'bbch': 'BBCH'},
                inplace=True
            )
            glai_agg.to_csv(
                out_dir.joinpath(f'LAI_{ww_dataset}_Raw-Data.csv'),
                index=False
            )

if __name__ == '__main__':

    fpath_dataset = Path('/mnt/ides/Lukas/02_Research/PhenomEn/01_Data/04_inSitu_Data/reichenau_et_al_crop_and_ancillary_data_v1.0')
    out_dir = Path('../../results/dose_reponse_in-situ/DE_Rur')
    out_dir.mkdir(exist_ok=True)
    prepare_data(fpath_dataset, out_dir)
