'''
Extract median S2 spectra per parcel as well as viewing and illumination
angles and store results in a GeoPackage (i.e., SQL database).

@author: Lukas Valentin Graf
'''

import geopandas as gpd

from datetime import datetime
from eodal.config import get_settings
from eodal.core.sensors import Sentinel2
from eodal.mapper.feature import Feature
from eodal.mapper.filter import Filter
from eodal.mapper.mapper import Mapper, MapperConfigs
from pathlib import Path
from typing import Any, Dict, List

Settings = get_settings()
Settings.USE_STAC = True

months_of_interest: List[int] = [3,4,5,6]
cloudy_pixel_percentage: int = 50 # percent
snow_ice_percentage = 0 # percent
processing_level: str = 'Level-2A'
collection: str = 'sentinel2-msi'

# Sentinel-2 bands to extract and use for PROSAIL runs
band_selection: List[str] = ['B02','B03','B04','B05','B06','B07','B8A','B11','B12']

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

# specify the processing of Sentinel-2 scenes (resample them to 10 m, apply cloud masking)
scene_kwargs: Dict[str,Any] = {
    'scene_constructor': Sentinel2.from_safe,
    'scene_constructor_kwargs': {'band_selection': band_selection},
    'scene_modifier': preprocess_sentinel2_scenes,
    'scene_modifier_kwargs': {'target_resolution': 10}
}

def config_mapper(
    time_start: datetime,
    time_end: datetime,
    bbox: gpd.GeoSeries
) -> MapperConfigs:
    """
    Setup EOdal mapper configurations
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

def extract_s2_spectra(
    ww_polys_dir: Path,
    out_dir: Path
):
    """
    """
    # loop over years and tiles
    for ww_polys_year_dir in ww_polys_dir.glob('*'):
        if not ww_polys_year_dir.is_dir(): continue
        year = int(ww_polys_year_dir.name)
        # loop over tiles
        for ww_polys_tile in ww_polys_year_dir.glob('*.gpkg'):
            s2_tile = ww_polys_tile.name.split('_')[0]
            # set time frame for S2 data extraction
            time_start = datetime(year, months_of_interest[0], 1)
            time_end = datetime(year, months_of_interest[-1], 15)
            # get the dissolved bounds of the polygons in geographic coordinates
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
            mapper.query_scenes()
            # drop all records from metadata where the S2 tile does not
            # equal the selected one
            mapper.metadata = mapper.metadata[
                mapper.metadata.tile_id == s2_tile
            ].copy()

if __name__ == '__main__':

    # directory with parcel geometries by year and S2 tile
    ww_polys_dir = Path('/mnt/ides/Lukas/00_GIS_Basedata/WW_CH')

    # directory where to save results by year and S2 tile
    out_dir = Path('/mnt/ides/Lukas/04_Work/LaaL')

    extract_s2_spectra(ww_polys_dir, out_dir)
