'''
Generate raw LAI time series from PROSAIL inversion of S2 imagery
for each pixel.

@author: Lukas Valentin Graf
'''

from eodal.config import get_settings
from eodal.core.band import Band
from eodal.core.operators import Operator
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

formats = {
    'Witzwil': '%Y-%m-%d %H:%M:%S',
    'Strickhof': '%d.%m.%Y %H:%M'
}
traits = ['lai', 'ccc']

logger = get_settings().logger


def extract_raw_lai_timeseries(
    test_sites_dir: Path,
    s2_trait_dir: Path,
    meteo_dir: Path,
    out_dir: Path,
    relevant_phase: str
) -> None:
    """
    Extracts raw LAI values for all pixels of field parcels during
    the selected phenological phase.

    :param test_sites_dir:
        directory with management data of the test sites for selecting
        the time frames and parcels accordingly
    :param s2_trait_dir:
        directory where S2 inversion results (i.e., traits) are stored
        per site and S2 scene
    :param meteo_dir:
        directory with meteorological data per site and year
    :param out_dir:
        directory where to store the results
    :param relevant_phase:
        relevant phenological macro-phase to use for extracting the S2 data
    """
    # loop over test sites and field parcels thereof
    # look only at those observations for which the phase was set to
    # selected phenological macro phase
    for fpath_test_site in test_sites_dir.glob('*.csv'):
        site_name = fpath_test_site.name.split('.')[0]
        test_site_df = pd.read_csv(fpath_test_site)
        test_site = fpath_test_site.name.split('.')[0]
        s2_trait_dir_test_site = s2_trait_dir.joinpath(test_site)
        # get list of available S2 observations
        s2_obs_test_site = [x for x in s2_trait_dir_test_site.glob('*.SAFE')]
        # get the observation time stamp (UTC): 20170818T103021
        s2_obs_times = [
            pd.to_datetime(
                x.name.split('_')[2], format='%Y%m%dT%H%M%S'
            ).tz_localize(
                'UTC'
            ).tz_convert(
                'Europe/Zurich'
            ) for x in s2_obs_test_site
        ]
        # convert to dataframe for more convenient handling
        s2_obs = pd.DataFrame(
            {'sensing_time': s2_obs_times, 'fpath': s2_obs_test_site})
        s2_obs.sort_values(by='sensing_time', inplace=True)
        s2_obs.index = s2_obs.sensing_time
        # scene collection for storing trait values
        for _, parcel in test_site_df.iterrows():
            sowing_date = pd.to_datetime(parcel.sowing_date).tz_localize(
                'Europe/Zurich')
            harvest_date = pd.to_datetime(parcel.harvest_date).tz_localize(
                'Europe/Zurich')
            s2_obs_parcel = s2_obs[sowing_date:harvest_date].copy()
            parcel_name = parcel['name']
            # check which scenes have a <parcel_name> sub-folder. Due
            # to no-data it might happen that some scenes did not
            # cover the parcel. In this case, we do not consider
            # these scenes any further
            s2_obs_parcel['has_parcel_subfolder'] = \
                s2_obs_parcel['fpath'].apply(
                    lambda x, parcel_name=parcel_name:
                        x.joinpath(str(parcel_name)).exists()
                )
            s2_obs_parcel = s2_obs_parcel[
                s2_obs_parcel.has_parcel_subfolder].copy()
            if s2_obs_parcel.empty:
                logger.warn(
                    f'No data found for parcel {parcel_name} at {site_name}')

            # drop duplicates based on the acquisition time. These might
            # stem from different tiles
            s2_obs_parcel.drop_duplicates(
                subset=['sensing_time'], keep='first', inplace=True)

            # loop over observations and check for the selected
            # phenological phase
            s2_obs_parcel['pheno_phase'] = s2_obs_parcel['fpath'].apply(
                lambda x, relevant_phase=relevant_phase,
                parcel_name=parcel_name:
                    x.joinpath(str(parcel_name)).joinpath(
                        f'parcel_{parcel_name}_phase_{relevant_phase}')
            )

            # check if result exists -> this actually identifies the timing of
            # the phases
            s2_obs_parcel['result_exists'] =\
                s2_obs_parcel['pheno_phase'].apply(
                    lambda x: x.exists()
                )

            # also include the two scenes before and after the identified
            # period
            # to account for the uncertainty in the identification of the
            # period
            s2_obs_parcel['number'] = [
                x for x in range(s2_obs_parcel.shape[0])]

            # TODO: This clause can be removed once the issue with the
            # 2023 data is solved
            try:
                scene_before = s2_obs_parcel[
                    s2_obs_parcel.result_exists]['number'][0] - 1
            except Exception:
                continue
            if scene_before >= 0:
                scene_before_idx = s2_obs_parcel[
                    s2_obs_parcel.number == scene_before].index[0]
                s2_obs_parcel.loc[scene_before_idx, 'result_exists'] = True
            scene_after = s2_obs_parcel[
                s2_obs_parcel.result_exists]['number'][-1] + 1
            if scene_after < s2_obs_parcel.shape[0]:
                scene_after_idx = s2_obs_parcel[
                    s2_obs_parcel.number == scene_after].index[0]
                s2_obs_parcel.loc[scene_after_idx, 'result_exists'] = True

            s2_obs_parcel = s2_obs_parcel[s2_obs_parcel.result_exists].copy()
            # loop over remaining sensing dates and read data
            scoll = SceneCollection()
            for _, s2_obs_date in s2_obs_parcel.iterrows():
                fpath_s2_traits = s2_obs_date.fpath.joinpath(
                    str(parcel_name)
                ).joinpath(
                    f'parcel_{parcel_name}_lai-ccc_data.tiff'
                )
                # in case the parcel result could not be generated, continue
                if not fpath_s2_traits.exists():
                    continue
                # read parcel results
                s2_traits = RasterCollection.from_multi_band_raster(
                    fpath_raster=fpath_s2_traits
                )

                # check if the result contains actual data
                if np.isnan(s2_traits['lai'].values).all():
                    continue
                scene_props = SceneProperties(
                    acquisition_time=s2_obs_date.sensing_time,
                    product_uri=s2_obs_date.fpath.name
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
                        new_vals_traits = {}
                        for trait in traits:
                            new_vals = (scoll[scoll.timestamps[-1]][trait] +
                                        s2_traits[trait]).values * 0.5
                            new_vals_traits[trait] = new_vals

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
                    for idx, trait in enumerate(traits):
                        if idx == 0:
                            new_scene = RasterCollection(
                                band_constructor=Band,
                                band_name=trait,
                                values=new_vals_traits[trait],
                                geo_info=s2_traits[trait].geo_info,
                                scene_properties=new_scene_properties
                            )
                        else:
                            new_scene.add_band(
                                band_constructor=Band,
                                band_name=trait,
                                values=new_vals_traits[trait],
                                geo_info=s2_traits[trait].geo_info
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
                    f'No scenes found for parcel {parcel_name} at {site_name}')
                continue

            # extract the meteorological data (hourly)
            fpath_meteo_site = meteo_dir.joinpath(
                f'{site_name}_Meteo_hourly.csv')
            meteo_site = pd.read_csv(fpath_meteo_site)
            meteo_site.time = pd.to_datetime(
                meteo_site.time, format=formats[site_name])
            meteo_site.index = meteo_site.time
            # we only need to have meteorological data for the S2 observations
            # selected
            min_time = pd.to_datetime(scoll.timestamps[0].split('+')[0])
            max_time = pd.to_datetime(scoll.timestamps[-1].split('+')[0])
            meteo_site_parcel = meteo_site[
                min_time.date():max_time.date()].copy()[['time', 'T_mean']]
            meteo_site_parcel.index = [
                x for x in range(meteo_site_parcel.shape[0])]
            # convert T_mean to float in case it is a string and replace
            # '-0' with nan
            if meteo_site_parcel.T_mean.dtype == object:
                meteo_site_parcel.T_mean = \
                    meteo_site_parcel.T_mean.str.replace(
                        '-', '')
            meteo_site_parcel.T_mean = meteo_site_parcel.T_mean.astype(float)

            # save data to output directory
            out_dir_parcel = out_dir.joinpath(
                f'parcel_{parcel_name}_{min_time.date()}-{max_time.date()}')
            out_dir_parcel.mkdir(exist_ok=True)

            # save "raw" LAI values as pickle
            fname_raw_lai = out_dir_parcel.joinpath('raw_trait_values.pkl')
            with open(fname_raw_lai, 'wb+') as dst:
                dst.write(scoll.to_pickle())

            # save "raw" LAI values as table using all pixels in the parcels
            # so that it is easier to work with the data
            # we use xarray as an intermediate to convert it to a pandas
            # DataFrame
            xarr = scoll.to_xarray()
            for trait in traits:
                df = xarr.to_dataframe(name=trait).reset_index()
                # drop nan's (these are the pixels outside the parcel)
                df.dropna(inplace=True)
                # drop the band name column since it is redundant
                df.drop('band', axis=1, inplace=True)
                df['pheno_phase'] = relevant_phase
                # save the DataFrame as CSV file
                fname_csv = out_dir_parcel.joinpath(f'raw_{trait}_values.csv')
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
            # TODO: fix bug when there is a '-' in the meteo data
            # (happened in the last phase, so not too urgent to fix)
            ax.plot(
                meteo_site_parcel.time, meteo_site_parcel['T_mean'].astype(
                    float))
            ax.set_ylabel('Mean Air Temperature 2m above ground [deg C]')
            plt.xticks(rotation=90)
            f.savefig(out_dir_parcel.joinpath(
                'hourly_mean_temperature.png'), bbox_inches='tight')
            plt.close(f)

            logger.info(
                f'Prepared LAI data for parcel {parcel_name} at {site_name}'
                f' ({relevant_phase})')


if __name__ == '__main__':

    test_sites_dir = Path('./data/Test_Sites')
    s2_trait_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/S2_Traits')  # noqa: E501
    relevant_phases = [
        'germination-endoftillering',
        'stemelongation-endofheading',
        'flowering-fruitdevelopment-plantdead']
    meteo_dir = Path('./data/Meteo')
    out_dir = Path('./results/test_sites_pixel_ts')
    out_dir.mkdir(exist_ok=True)

    for relevant_phase in relevant_phases:
        extract_raw_lai_timeseries(
            test_sites_dir=test_sites_dir,
            s2_trait_dir=s2_trait_dir,
            relevant_phase=relevant_phase,
            meteo_dir=meteo_dir,
            out_dir=out_dir
        )
