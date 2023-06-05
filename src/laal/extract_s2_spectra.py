'''
Extract median S2 spectra per parcel as well as viewing and illumination
angles and store results in a GeoPackage (i.e., SQL database).

@author: Lukas Valentin Graf
'''

import geopandas as gpd
import numpy as np
import os
import pandas as pd
import planetary_computer
import tempfile
import urllib.request
import uuid

from datetime import datetime
from eodal.config import get_settings
from eodal.core.sensors import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from pathlib import Path
from typing import Dict, List


Settings = get_settings()
Settings.USE_STAC = True
logger = Settings.logger

cloudy_pixel_percentage: int = 70  # percent
parcel_cloudy_pixel_percentage: int = 20  # percent
snow_ice_percentage = 0  # percent
processing_level: str = 'Level-2A'
collection: str = 'sentinel2-msi'

# Sentinel-2 bands to extract and use for PROSAIL runs
band_selection: List[str] = [
    'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
# number of days per month
days_per_month: Dict[int, int] = {
    1: 31, 2: 28, 3: 31, 4: 30, 5: 31,
    6: 30, 7: 31, 8: 31, 9: 30, 10: 31,
    11: 30, 12: 31}


def angles_from_mspc(url: str) -> Dict[str, float]:
    """
    Extract viewing and illumination angles from MS Planetary Computer
    metadata XML (this is a work-around until STAC provides the angles
    directly)

    :param url:
        URL to the metadata XML file
    :returns:
        extracted angles as dictionary
    """
    response = urllib.request.urlopen(planetary_computer.sign_url(url)).read()
    temp_file = os.path.join(tempfile.gettempdir(), f'{uuid.uuid4()}.xml')
    with open(temp_file, 'wb') as dst:
        dst.write(response)

    from eodal.metadata.sentinel2.parsing import parse_MTD_TL
    metadata = parse_MTD_TL(in_file=temp_file)
    # get sensor zenith and azimuth angle
    sensor_angles = ['SENSOR_ZENITH_ANGLE', 'SENSOR_AZIMUTH_ANGLE']
    sensor_angle_dict = {
        k: v for k, v in metadata.items() if k in sensor_angles}
    return sensor_angle_dict


def preprocess_sentinel2_scenes(
    ds: Sentinel2,
    target_resolution: int,
) -> Sentinel2:
    """
    Resample Sentinel-2 scenes and mask clouds, shadows, and snow
    based on the Scene Classification Layer (SCL).

    NOTE:
        Depending on your needs, the pre-processing function can be
        fully customized using the full power of EOdal and its
        interfacing libraries!

    :param target_resolution:
        spatial target resolution to resample all bands to.
    :returns:
        resampled, cloud-masked Sentinel-2 scene.
    """
    # resample scene
    ds.resample(inplace=True, target_resolution=target_resolution)
    # mask clouds, shadows, and snow
    ds.mask_clouds_and_shadows(inplace=True)
    return ds


def config_mapper(
    time_start: datetime,
    time_end: datetime,
    bbox: gpd.GeoSeries
) -> MapperConfigs:
    """
    Setup EOdal mapper configurations

    :param time_start:
        start of the time window
    :param time_end:
        end of the time window
    :param bbox:
        bounding box of the area of interest
    :returns:
        EOdal mapper configurations
    """
    metadata_filters = [
        Filter('cloudy_pixel_percentage', '<', cloudy_pixel_percentage),
        Filter('processing_level', '==', processing_level)
    ]
    feature = Feature.from_geoseries(bbox)
    mapper_configs = MapperConfigs(
        collection=collection,
        feature=feature,
        time_start=time_start,
        time_end=time_end,
        metadata_filters=metadata_filters
    )
    return mapper_configs


def parcel_cloud_cover(raster: np.ma.MaskedArray) -> float:
    """
    Check how much percent of the pixels in a parcel
    are cloudy

    :param raster:
        raster with SCL values
    :returns:
        percentage of valid SCL pixels (classes 4 and 5) in %
    """
    # get the non-masked SCL values
    _raster = raster.copy()
    _raster = _raster.compressed()
    scl_classes, counts = np.unique(_raster, return_counts=True)
    scl_counts = dict(zip(scl_classes, counts))
    # count the occurence of SCL classes 4 and 5
    valid_counts = scl_counts.get(4, 0) + scl_counts.get(5, 0)
    # in case the entire parcel is masked return 100
    if _raster.size == 0:
        return 100
    # else calculate the percentage of valid pixel
    return (1 - valid_counts / _raster.size) * 100


def extract_s2_spectra(
    ww_polys_dir: Path,
    months_of_interest: List[int],
    out_dir: Path
):
    """
    Get the median spectral reflectance per parcel and
    Sentinel-2 scene for the months of interest alongside
    the viewing and illumination angles required to run
    PROSAIL

    :param ww_polys_dir:
        directory with the winter wheat polygons
    :param months_of_interest:
        months of interest
    :param out_dir:
        output directory
    """
    # specify the processing of Sentinel-2 scenes
    # (resample them to 10 m, apply cloud masking)
    scene_kwargs = {
        'scene_constructor': Sentinel2.from_safe,
        'scene_constructor_kwargs': {'band_selection': band_selection},
        'scene_modifier': preprocess_sentinel2_scenes,
        'scene_modifier_kwargs': {'target_resolution': 10}
    }

    # loop over years and tiles
    for ww_polys_year_dir in ww_polys_dir.glob('*'):
        if not ww_polys_year_dir.is_dir():
            continue
        year = int(ww_polys_year_dir.name)
        # loop over tiles
        for ww_polys_tile in ww_polys_year_dir.glob('*.gpkg'):
            s2_tile = ww_polys_tile.name.split('_')[0]

            logger.info(
                f'Processing {s2_tile} for {year} ' +
                f'(months: {months_of_interest})')

            # set time frame for S2 data extraction
            time_start = datetime(year, months_of_interest[0], 1)
            time_end = datetime(
                year, months_of_interest[-1],
                days_per_month[months_of_interest[-1]])

            # check if the results exist already
            out_file_scoll = out_dir.joinpath(
                f'{s2_tile}_{year}_{months_of_interest[0]}-' +
                f'{months_of_interest[-1]}_srf.pkl')
            if out_file_scoll.exists():
                logger.info(
                    f'{out_file_scoll} exists already. ' +
                    'Skipping this tile.')
                continue

            # get the dissolved bounds of the polygons in
            # geographic coordinates
            polys = gpd.read_file(ww_polys_tile)
            bbox = polys.to_crs(4326).dissolve().geometry
            # setup EOdal mapper configurations
            mapper_configs = config_mapper(
                time_start=time_start,
                time_end=time_end,
                bbox=bbox
            )
            # query available scene
            mapper = Mapper(mapper_configs)
            try:
                mapper.query_scenes()
            except Exception as e:
                logger.error(
                    f'Error while querying scenes for {s2_tile} ' +
                    f'between {time_start} and {time_end}: {e}')
                continue

            # drop all records from metadata where the S2 tile does not
            # equal the selected one
            # this will speed up the retrieval process as there will be
            # no mosaicing required
            mapper.metadata = mapper.metadata[
                mapper.metadata.tile_id == s2_tile
            ].copy()

            # if no scenes were found continue
            if mapper.metadata.empty:
                logger.warning(
                    f'No scenes found for {s2_tile} between ' +
                    f'{time_start} and {time_end}'
                )

            # get the angular data
            if Settings.USE_STAC:
                # this step is only required when the data is read from
                # STAC and can be removed once STAC supports the
                # provision of the sensor angles
                updated_metadata = []
                for _, metadata_record in mapper.metadata.iterrows():
                    url_md_xml = metadata_record.assets.get(
                        'granule-metadata').get('href')
                    sensor_angles = angles_from_mspc(url_md_xml)
                    sensor_angles = {
                        k.lower(): v for k, v in sensor_angles.items()}
                    metadata_record = pd.concat(
                        [metadata_record, pd.Series(sensor_angles)])
                    updated_metadata.append(metadata_record)
                updated_metadata_df = pd.concat(updated_metadata, axis=1).T
                updated_metadata_gdf = gpd.GeoDataFrame(
                    updated_metadata_df,
                    geometry=updated_metadata_df.geom,
                    crs=mapper.metadata.crs)
                mapper.metadata = updated_metadata_gdf

            # load scenes
            mapper.load_scenes(scene_kwargs)
            # calculate zonal statistics of the SCL layer. If more than XX% of
            # the parcel pixel is cloudy remove the record
            scl_stats = mapper.data.get_feature_timeseries(
                vector_features=polys,
                band_selection=['SCL'],
                method=[parcel_cloud_cover]
            )
            scl_stats = scl_stats[
                scl_stats.cloudy_pixel_percentage <=
                parcel_cloudy_pixel_percentage].copy()
            # save the metadata
            out_file_metadata = out_dir.joinpath(
                f'{s2_tile}_{year}_{months_of_interest[0]}-' +
                f'{months_of_interest[-1]}_metadata.csv')
            mapper.metadata.to_csv(out_file_metadata, index=False)

            # save the SceneCollection so also the original pixels
            # are available
            with open(out_file_scoll, 'wb+') as dst:
                dst.write(mapper.data.to_pickle())

            logger.info(f'Finished {s2_tile} for {year}' +
                        f' (month(s): {months_of_interest})')


if __name__ == '__main__':

    # base directory
    base_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn')

    # directory with parcel geometries by year and S2 tile
    ww_polys_dir = base_dir.joinpath('00_GIS_Basedata/Crop_Maps/WW_CH')

    # directory where to save results by year and S2 tile
    out_dir = base_dir.joinpath('04_LaaL').joinpath('CH')
    out_dir.mkdir(exist_ok=True)

    for month in [3, 4, 5, 6]:
        months_of_interest = [month]
        extract_s2_spectra(
            ww_polys_dir=ww_polys_dir,
            months_of_interest=months_of_interest,
            out_dir=out_dir)
