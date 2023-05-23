"""
Combine the LAI data over the test sites covering the period where
in-situ ratings are available.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.operators import Operator
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path

logger = get_settings().logger
plt.style.use('bmh')

formats = {
    'Witzwil': '%Y-%m-%d %H:%M:%S',
    'Strickhof': '%d.%m.%Y %H:%M',
    'SwissFutureFarm': '%Y-%m-%d %H:%M:%S'
}


def extract_raw_lai_timeseries(
        test_sites_dir: Path,
        s2_trait_dir: Path,
        meteo_dir: Path,
        out_dir: Path,
        farms: list[str],
        years: list[int]
) -> None:
    """
    Extract the raw LAI timeseries for the validation sites.

    :param test_sites_dir:
        Path to the directory containing the test sites.
    :param s2_trait_dir:
        Path to the directory containing the Sentinel-2 trait data.
    :param meteo_dir:
        Path to the directory containing the meteorological data.
    :param out_dir:
        Path to the directory where the results should be stored.
    :param farms:
        List of farms to be considered.
    :param years:
        List of years to be considered.
    """
    # loop over years
    for year in years:
        # loop over farms
        for farm in farms:
            # go through the scenes and only consider those that
            # are in the current year and have a subfolder called
            # 'validation'
            scene_dir_farm = s2_trait_dir.joinpath(farm)
            # open an empty SceneCollection
            scoll = SceneCollection()
            for scene in scene_dir_farm.glob('*.SAFE'):
                # get the date of the scene
                scene_date = pd.to_datetime(
                    scene.name.split('_')[2]).tz_localize(
                        'Europe/Zurich')
                if scene_date.year != year:
                    continue
                # check for a subfolder called 'validation'
                if not scene.joinpath('validation').exists():
                    continue
                # get the LAI data
                lai_file = scene.joinpath('validation', 'traits.tiff')
                # read results into RasterCollection
                s2_traits = RasterCollection.from_multi_band_raster(
                    fpath_raster=lai_file,
                    band_idxs=[1],
                    band_aliases=['lai'],
                    band_names_dst=['lai']
                )
                # check if the result contains actual data
                if np.isnan(s2_traits['lai'].values).all():
                    continue
                scene_props = SceneProperties(
                    acquisition_time=scene_date,
                    product_uri=scene.name
                )
                s2_traits.scene_properties = scene_props

                # try to add the scene. In some cases it might happen that a
                # timestamps occurs twice, e.g., because the data is stored
                # in multiple S2 tiles.
                # in this case we use the average LAI value (there are small
                # differences between the tiles because of the tile specific
                # atmospheric correction)
                try:
                    scoll.add_scene(s2_traits)
                except KeyError:
                    try:
                        new_vals = (scoll[scoll.timestamps[-1]]['lai'] +
                                    s2_traits['lai']).values * 0.5
                    # if the geometries do not align completely we keep
                    # the first scene
                    except Operator.BandMathError:
                        continue
                    new_product_uri = scoll[
                        scoll.timestamps[-1]].scene_properties.product_uri + \
                        '&&' + s2_traits.scene_properties.product_uri
                    new_scene_properties = SceneProperties(
                        acquisition_time=scoll[
                            scoll.timestamps[-1]
                            ].scene_properties.acquisition_time,
                        product_uri=new_product_uri
                    )
                    # get a "new" scene
                    new_scene = RasterCollection(
                        band_constructor=Band,
                        band_name='lai',
                        values=new_vals,
                        geo_info=s2_traits['lai'].geo_info,
                        scene_properties=new_scene_properties
                    )
                    # delete the "old scene" from the SceneCollection
                    del scoll[scoll.timestamps[-1]]
                    # and add the update scene
                    scoll.add_scene(new_scene)

            # make sure the scene are sorted chronologically
            scoll = scoll.sort('asc')
            # continue if no scenes were found
            if scoll.empty:
                logger.warn(
                    f'No scenes found for {farm} in {year}.')
                continue

            # extract the meteorological data (hourly)
            fpath_meteo_site = meteo_dir.joinpath(
                f'{farm}_Meteo_hourly.csv')
            meteo_site = pd.read_csv(fpath_meteo_site)
            meteo_site.time = pd.to_datetime(
                meteo_site.time, format=formats[farm])
            meteo_site.index = meteo_site.time
            # we only need to have meteorological data for the S2 observations
            # selected
            try:
                min_time = pd.to_datetime(scoll.timestamps[0].split('+')[0])
            except Exception as e:
                logger.error(e)
                continue
            max_time = pd.to_datetime(scoll.timestamps[-1].split('+')[0]) + \
                pd.Timedelta(days=1)   # add one day to include the last day
            meteo_site_parcel = meteo_site[
                min_time.date():max_time.date()].copy()[['time', 'T_mean']]
            meteo_site_parcel.index = [
                x for x in range(meteo_site_parcel.shape[0])]

            # save data to output directory
            out_dir_parcel = out_dir.joinpath(
                f'farm_{farm}_{min_time.date()}-{max_time.date()}')
            out_dir_parcel.mkdir(exist_ok=True)

            # save "raw" LAI values as pickle
            fname_raw_lai = out_dir_parcel.joinpath('raw_lai_values.pkl')
            with open(fname_raw_lai, 'wb+') as dst:
                dst.write(scoll.to_pickle())

            # save "raw" LAI values as table using all pixels in the parcels
            # so that it is easier to work with the data
            # we use xarray as an intermediate to convert it to a pandas
            # DataFrame
            xarr = scoll.to_xarray()
            df = xarr.to_dataframe(name='lai').reset_index()
            # drop nan's (these are the pixels outside the parcel)
            df.dropna(inplace=True)
            # drop the band name column since it is redundant
            df.drop('band', axis=1, inplace=True)
            # save the DataFrame as CSV file
            fname_csv = out_dir_parcel.joinpath('raw_lai_values.csv')
            df.to_csv(fname_csv, index=False)

            # plot data LAI as a map
            f = scoll.plot(
                band_selection='lai',
                figsize=(5*len(scoll), 5),
                max_scenes_in_row=len(scoll)
            )
            f.savefig(out_dir_parcel.joinpath('raw_lai_values.png'))
            plt.close(f)

            # save hourly meteo data
            meteo_site_parcel.to_csv(out_dir_parcel.joinpath(
                'hourly_mean_temperature.csv'), index=False)
            f, ax = plt.subplots(figsize=(8, 5))
            ax.plot(
                meteo_site_parcel.time, meteo_site_parcel['T_mean'].astype(
                    float))
            ax.set_ylabel('Mean Air Temperature 2m above ground [deg C]')
            plt.xticks(rotation=90)
            f.savefig(out_dir_parcel.joinpath(
                'hourly_mean_temperature.png'), bbox_inches='tight')
            plt.close(f)

            logger.info(
                f'Prepared LAI data for parcel {farm} in {year}')


if __name__ == '__main__':

    test_sites_dir = Path('./data/Test_Sites')
    s2_trait_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/S2_Traits')  # noqa: E501
    meteo_dir = Path('./data/Meteo')
    out_dir = Path('./results/validation_sites')
    out_dir.mkdir(exist_ok=True)

    farms = ['Witzwil', 'Strickhof', 'SwissFutureFarm']
    years = [2022]  # , 2023]

    extract_raw_lai_timeseries(
        test_sites_dir=test_sites_dir,
        s2_trait_dir=s2_trait_dir,
        meteo_dir=meteo_dir,
        out_dir=out_dir,
        farms=farms,
        years=years
    )
