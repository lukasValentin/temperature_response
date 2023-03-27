'''
Created on Mar 21, 2023

@author: graflu
'''

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eodal.config import get_settings
from eodal.core.raster import RasterCollection
from pathlib import Path

mpl.rc('font', size=16)
plt.style.use('seaborn-dark')

logger = get_settings().logger

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

def plot_observation(trait_ds: RasterCollection) -> plt.Figure:
    """
    Plot traits as a map

    :param trait_ds:
        RasterCollection with traits from PROSAIL inversion
    :returns:
        maps as rendered figure
    """
    # plot traits on a map
    f, ax = plt.subplots(ncols=3, nrows=1, figsize=(30,10), sharey=True, sharex=True)
    labels = [
        r'Green LAI [$m^2$ $m^{-2}$]',
        r'CCC [$g$ $m^{-2}$]',
        'Median Cost Function Value [-]'
    ]
    for idx, trait in enumerate(['lai', 'ccc', 'median_error']):
        trait_ds[trait].plot(
            ax=ax[idx],
            colormap='viridis',
            colorbar_label=labels[idx],
            fontsize=16
        )
        ax[idx].set_title('')
        if idx > 0:
            ax[idx].set_ylabel('')
    return f

def combine_lai_model_results(
    test_sites_dir: Path,
    s2_trait_dir: Path,
    meteo_dir: Path
) -> None:
    """
    Combine model outputs from different phenological phases to get
    a single trait time series per field parcel.

    For each scene, the phenological phase is determined based on accumulated
    Growing Degree Days, and the statistics are extracted.

    :param test_sites_dir:
        directory with management information and parcel shapes per site
    :param s2_trait_dir:
        directory where the traits are stored (per site) from PROSAIL output
    :param meteo_dir:
        directory where to find the corresponding meteorological data
    """
    # loop over sites
    for fpath_site_management in test_sites_dir.glob('*.csv'):
        site = fpath_site_management.name.split('.')[0]
        fpath_site_parcels = test_sites_dir.joinpath(
            fpath_site_management.name.replace('.csv','.gpkg')
        )
        site_parcels = gpd.read_file(fpath_site_parcels)
        site_management = pd.read_csv(fpath_site_management)
        # open meteorological data
        meteo_site = pd.read_csv(meteo_dir.joinpath(f'{site}_Meteo_hourly.csv'))
        meteo_site.index = pd.to_datetime(meteo_site.time, format=date_formats_sites[site])
        # loop over parcels
        for _, record in site_parcels.iterrows():
            parcel_geom = site_parcels[site_parcels.name == record['name']].copy()
            if parcel_geom.empty:
                logger.error(f'Could not find geometry for {site} {record["name"]}')
                continue
            # reproject to Swiss coordinates
            parcel_geom.to_crs(epsg=2056, inplace=True)
            # buffer parcel geometry 20m inwards
            parcel_geom_buffered = parcel_geom.buffer(-20)
            # get management data
            parcel_management = site_management[
                site_management['name'].astype(str) == str(record['name'])
            ].copy()
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
                        logger.error(f'{site} {scene} Could not determine phase')
                        continue
                    # open the corresponding file with trait values and read the data
                    fpath_trait = scene.joinpath(f'{phase}_lutinv_traits.tiff')
                    if not fpath_trait.exists():
                        logger.error(f'{fpath_trait} not found')
                        continue
                    # read data. If the raster does not overlap the geometry of the parcel continue
                    try:
                        trait_ds = RasterCollection.from_multi_band_raster(
                            fpath_raster=fpath_trait,
                            vector_features=parcel_geom_buffered
                        )
                    except TypeError:
                        logger.info(f'{site} {record["name"]}: {scene} does not overlap parcel')
                        continue
                    # plot as map
                    f = plot_observation(trait_ds=trait_ds)
                    f.suptitle(
                        f'{site} - Parcel {record["name"]} {sensing_date.date()} ({phase})'
                    )
                    parcel_out_dir = scene.joinpath(record["name"])
                    parcel_out_dir.mkdir(exist_ok=True)
                    # save figure
                    fname_figure = parcel_out_dir.joinpath(f'parcel_{record["name"]}_lai-ccc_map.png')
                    f.savefig(fname_figure)
                    plt.close(f)
                    # save data as geoTiff
                    fname_tiff = parcel_out_dir.joinpath(f'parcel_{record["name"]}_lai-ccc_data.tiff')
                    trait_ds.to_rasterio(fname_tiff)
                    # save pheno-phase
                    with open(parcel_out_dir.joinpath(f'parcel_{record["name"]}_phase_{phase}'), 'w+') as src:
                        src.write('')
                    # extract statistics
                    stats = trait_ds.band_summaries(
                        method=['min', 'mean', 'median', 'max', 'percentile_10', 'percentile_90']
                    )
                    # update geometry column to represent the actual field boundaries
                    # (and not its bounding box)
                    stats.to_crs(crs=parcel_geom.crs, inplace=True)
                    stats.geometry = [parcel_geom.geometry.iloc[0] for x in range(stats.shape[0])]
                    # save as GeoPackage
                    fname_gpkg = parcel_out_dir.joinpath(f'{record["name"]}_lutinv_stats.gpkg')
                    stats.to_file(fname_gpkg)
                    logger.info(f'Successfully processed {site} {record["name"]} {scene}')

if __name__ == '__main__':
    
    test_sites_dir = Path('../data/Test_Sites')
    s2_trait_dir = Path('../S2_Traits')
    meteo_dir = Path('../data/Meteo')

    combine_lai_model_results(
        test_sites_dir=test_sites_dir,
        s2_trait_dir=s2_trait_dir,
        meteo_dir=meteo_dir
    )
