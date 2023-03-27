'''
Created on Mar 21, 2023

@author: graflu
'''

import geopandas as gpd
import numpy as np
import pandas as pd

from eodal.core.raster import RasterCollection
from pathlib import Path

# make dateformats explicit to avoid formatting errors
date_formats_sites = {
    'Witzwil': '%Y-%m-%d %H:%M:%S',
    'Strickhof': '%d.%m.%Y %H:%M'
}

# AGDD windows for switching between phenological macro-stages
GDD_CRITICAL = {
    'SE': 800,
    'AN': 1490
}

def combine_lai_model_results(
    test_sites_dir: Path,
    s2_trait_dir: Path,
    meteo_dir: Path
):
    """
    """
    # loop over sites
    for fpath_site_management in test_sites_dir.glob('*.csv'):
        site = fpath_site_management.name.split('.')[0]
        fpath_site_parcels = test_sites_dir.joinpath(fpath_site_management.name.replace('.csv','.gpkg'))
        site_parcels = gpd.read_file(fpath_site_parcels)
        site_management = pd.read_csv(fpath_site_management)
        # open meteorological data
        meteo_site = pd.read_csv(meteo_dir.joinpath(f'{site}_Meteo_hourly.csv'))
        meteo_site.index = pd.to_datetime(meteo_site.time, format=date_formats_sites[site])
        # loop over parcels
        for _, record in site_parcels.iterrows():
            # get management data
            parcel_management = site_management[site_management['name'].astype(str) == str(record['name'])].copy()
            # there could be more than one season
            for _, schedule in parcel_management.iterrows():
                sowing_date = pd.to_datetime(schedule.sowing_date)
                harvest_date = pd.to_datetime(schedule.harvest_date)
                # get meteo data between these two dates
                meteo_parcel = meteo_site[sowing_date:harvest_date]
                # resample to daily values
                t_mean_daily = meteo_parcel['T_mean'].astype(float).resample('24H').mean()
                gdd = [x if x > 0 else 0 for x in t_mean_daily]
                gdd_cumsum = np.cumsum(gdd)
                # map dates on GDD
                dates = t_mean_daily.index.date
                agdd = dict(zip(list(dates), gdd_cumsum))

                # loop over S2 traits and select those observations located
                # between sowing and harvest date
                s2_trait_dir_site = s2_trait_dir.joinpath(site)
                for scene in s2_trait_dir_site.glob('*.SAFE'):
                    sensing_date = pd.to_datetime(scene.name.split('_')[2][0:8], format='%Y%m%d')
                    if not sowing_date <= sensing_date <= harvest_date:
                        continue
                    # lookup the AGDD of the current S2 scene
                    scene_agdd = agdd[sensing_date.date()]
                    # use AGDD threshold to determine the phenological macro-stage the parcel is in
                    if scene_agdd < GDD_CRITICAL['SE']:
                        phase = 'germination-endoftillering'
                    elif GDD_CRITICAL['SE'] <= scene_agdd < GDD_CRITICAL['AN']:
                        phase = 'stemelongation-endofheading'
                    elif scene_agdd >= GDD_CRITICAL['AN']:
                        phase = 'flowering-fruitdevelopment-plantdead'
                    else:
                        print('Could not determine phase')
                        continue
                    # open the corresponding file with trait values and read the data
                    fpath_trait = scene.joinpath(f'{phase}_lutinv_traits.tiff')
                    if not fpath_trait.exists():
                        print(f'{fpath_trait} not found')
                        continue
                    trait_ds = RaterCollection.from_m
                    

if __name__ == '__main__':
    
    test_sites_dir = Path('../data/Test_Sites')
    s2_trait_dir = Path('../S2_Traits')
    meteo_dir = Path('../data/Meteo')

    combine_lai_model_results(
        test_sites_dir=test_sites_dir,
        s2_trait_dir=s2_trait_dir,
        meteo_dir=meteo_dir
    )
