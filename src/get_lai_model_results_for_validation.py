"""
This module extracts S2 derived LAI values for those farms
and parcels with in-situ data available. We focus on the
stem elongation phase, i.e., we take all S2 scenes between
BBCH 31 and 59 rated in-situ.
"""

import geopandas as gpd
import pandas as pd

from eodal.config import get_settings
from eodal.core.raster import RasterCollection
from pathlib import Path
from typing import List


logger = get_settings().logger

# define BBCH range for stem elongation
BBCH_MIN = 31
BBCH_MAX = 59


def get_lai_model_results_for_validation(
        trait_dir: Path,
        insitu_dir: Path,
        test_site_dir: Path,
        years: List[int]
) -> None:
    """
    Extract LAI model results for validation

    :param trait_dir:
        directory with S2 model results
    :param insitu_dir:
        directory with in-situ data
    :param test_site_dir:
        directory with test site data
    :param years:
        years with in-situ data
    """
    # loop over years
    for year in years:
        # the in-situ data is stored by year
        insitu_year_dir = insitu_dir.joinpath(f'{year}')
        # read the data into a GeoDataFrame
        fpath_lai = insitu_year_dir.joinpath('in-situ_glai.gpkg')
        insitu_lai = gpd.read_file(fpath_lai)
        fpath_bbch = insitu_year_dir.joinpath('in-situ_bbch.gpkg')
        insitu_bbch = gpd.read_file(fpath_bbch)
        # convert date to pd.to_datetime
        insitu_lai['date'] = pd.to_datetime(insitu_lai['date'])
        insitu_bbch['date'] = pd.to_datetime(insitu_bbch['date'])
        # join LAI and BBCH data on date and parcel
        insitu = pd.merge(
            insitu_lai, insitu_bbch,
            on=['date', 'location', 'parcel', 'point_id'])
        # filter for stem elongation phase
        insitu = insitu[
            (insitu['BBCH Rating'] >= BBCH_MIN) &
            (insitu['BBCH Rating'] <= BBCH_MAX)].copy()

        # convert insitu data to GeoDataFrame
        insitu = gpd.GeoDataFrame(
            insitu, geometry=insitu.geometry_x, crs=insitu_bbch.crs)
        # drop unnecessary columns
        insitu.drop(columns=['geometry_x', 'geometry_y'], inplace=True)
        # project to swiss coordinates (LV95)
        insitu = insitu.to_crs(2056)

        # save the GeoDataFrame to a GeoPackage in the trait_dir directory
        fpath_gpkg = trait_dir.joinpath(f'{year}_insitu_lai_bbch_se.gpkg')
        insitu.to_file(fpath_gpkg, driver='GPKG')
        logger.info(f'Saved in-situ data for stem elongation phase {year}')

        # loop over farms in in-situ data
        for farm in insitu['location'].unique():
            # get the farm directory
            farm_dir = trait_dir.joinpath(farm)
            if not farm_dir.exists():
                continue
            # get parcel geometries for the farm
            fpath_gpkg = test_site_dir.joinpath(f'{farm}.gpkg')
            test_site_gdf = gpd.read_file(fpath_gpkg)
            # get only those parcels whose growth period is in the
            # range of the insitu data. The sowing date must be in
            # the year before the in-situ data was taken.
            test_site_gdf['sowing_year'] = pd.to_datetime(
                test_site_gdf['sowing_date']).dt.year
            test_site_gdf = test_site_gdf[
                test_site_gdf['sowing_year'] == year - 1].copy()

            # get the in-situ data for the farm
            farm_insitu = insitu[insitu['location'] == farm].copy()
            # loop over scenes in farm
            for scene_dir in farm_dir.glob('*.SAFE'):
                # get the date of the scene
                date = pd.to_datetime(scene_dir.name.split('_')[2])
                # check if the scene is in the stem elongation phase
                if not (
                    date >= farm_insitu['date'].min() and
                        date <= farm_insitu['date'].max()):
                    continue
                # check if an inversion result for the stem-elongation phase
                # has been generated already
                fpath_traits = scene_dir.joinpath(
                    'stemelongation-endofheading_lutinv_traits.tiff')
                if not fpath_traits.exists():
                    continue

                # read the trait data for the parcel geometry using eodal
                ds = RasterCollection.from_multi_band_raster(
                    fpath_raster=fpath_traits, vector_features=test_site_gdf)
                # save the result in a validation sub-folder
                fpath_validation = scene_dir.joinpath('validation')
                fpath_validation.mkdir(exist_ok=True)
                # save ds to the validation folder as a geoTiff
                ds.to_rasterio(fpath_validation.joinpath('traits.tiff'))

                logger.info(f'Saved traits for {farm}: {scene_dir.name}')


if __name__ == '__main__':

    # directory with S2 model results
    trait_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/S2_Traits')  # noqa: E501

    # directory with test site data
    test_site_dir = Path('./data/Test_Sites')

    # directory with in-situ data
    insitu_dir = Path('./data/in-situ')

    # years with in-situ data
    years = [2022]  # [2022, 2023]

    get_lai_model_results_for_validation(
        trait_dir=trait_dir, insitu_dir=insitu_dir, years=years,
        test_site_dir=test_site_dir)
