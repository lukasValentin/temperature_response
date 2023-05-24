'''
Created on Apr 21, 2023

@author: Flavian Tschurr and Lukas Valentin Graf
'''

import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path
from typing import List

from ensemble_kalman_filter import EnsembleKalmanFilter
from temperature_response import Response

logger = get_settings().logger
warnings.filterwarnings('ignore')

# set seed to make results reproducible
np.random.seed(42)

# noise level for temperature data
noise_level = 5  # in percent
# uncertainty in LAI data (relative)
lai_uncertainty = 5  # in percent


def prepare_lai_ts(lai_pixel_ts: pd.Series) -> pd.Series:
    """
    Prepare LAI time series for the temperature response function.

    Parameters
    ----------
    lai_pixel_ts : pd.Series
        LAI time series.
    return : pd.Series
        Prepared LAI time series.
    """
    lai_pixel_ts.sort_values(by='time', inplace=True)
    lai_pixel_ts.index = [x for x in range(len(lai_pixel_ts))]

    # apply a simple outlier filtering
    # values smaller than one standard deviation are removed
    # we look in negative direction, only
    # the exception is the first value
    lai_values = lai_pixel_ts['lai'].values.copy()
    mean, std = np.mean(lai_values), np.std(lai_values)
    lai_values[1:] = np.where(
        lai_values[1:] < mean - std,
        np.nan,
        lai_values[1:]
    )
    # get indices of nan values
    nan_indices = np.argwhere(np.isnan(lai_values)).flatten()
    # remove nan values from lai_pixel_ts
    lai_pixel_ts = lai_pixel_ts[
        ~lai_pixel_ts.index.isin(nan_indices)].copy()

    return lai_pixel_ts


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def interpolate_between_assimilated_points(
        measurement_index: List[int],
        meteo_pixel: pd.DataFrame,
        response: Response
) -> pd.DataFrame:
    """
    Interpolate assimilated LAI values between satellite observations.
    """
    model_sims_between_points = []
    # loop over measurement points
    for i in range(len(measurement_index)-1):
        meteo_time_window = meteo_pixel.loc[
            measurement_index[i]:measurement_index[(i+1)]].copy()
        # calculate the temperature response
        meteo_time_window['temp_response'] = \
            response.get_response(
                meteo_time_window['T_mean'])
        # get cumulative sum of temperature response
        meteo_time_window['temp_response_cumsum'] = \
            meteo_time_window['temp_response'].cumsum()
        # scale values between lai_value_start and lai_value_end
        in_min = meteo_time_window['temp_response_cumsum'].iloc[0]
        in_max = meteo_time_window['temp_response_cumsum'].iloc[-1]

        for measure in ['mean', 'std']:
            out_min = meteo_time_window[
                f'reconstructed_lai_{measure}'].iloc[0]
            out_max = meteo_time_window[
                f'reconstructed_lai_{measure}'].iloc[-1]
            # our assumption here is that LAI MUST increase between
            # two observations
            out_range = out_max - out_min
            if out_range < 0:
                continue
            meteo_time_window[f'reconstructed_lai_{measure}'] = \
                meteo_time_window['temp_response_cumsum'].apply(
                    lambda x: rescale(x, in_min, in_max, out_min, out_max))

        model_sims_between_points.append(meteo_time_window)

    model_sims_between_points = pd.concat(
        model_sims_between_points, axis=0)
    return model_sims_between_points


def apply_temperature_response(
        parcel_lai_dir: Path,
        dose_response_parameters: Path,
        response_curve_type,
        covariate_granularity,
        n_sim=50
) -> None:
    """
    Apply the temperature response function to the LAI time series.

    Parameters
    ----------
    parcel_lai_dir : Path
        Path to the directory containing the LAI time series.
    dose_response_parameters : Path
        Path to the dose response parameters.
    response_curve_type : str
        Type of the response curve.
    covariate_granularity : str
        Granularity of the covariate.
    n_sim : int
        Number of simulations for the ensemble Kalman filter.
    """
    # read in dose response paramters
    path_paramters = Path.joinpath(
        dose_response_parameters,
        response_curve_type,
        f'{response_curve_type}_granularity_{covariate_granularity}' +
        '_parameter_T_mean.csv')

    params = pd.read_csv(path_paramters)
    params = dict(zip(params['parameter_name'], params['parameter_value']))

    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):

        # make an output dir
        output_dir = parcel_dir.joinpath(response_curve_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        lai['time'] = pd.to_datetime(
            lai['time'], format='ISO8601', utc=True).dt.floor('H')

        # meteorological data
        # TODO: check what the time zone of the meteo data is
        fpath_meteo = parcel_dir.joinpath('hourly_mean_temperature.csv')
        meteo = pd.read_csv(fpath_meteo)
        # ensure timestamp format
        meteo['time'] = pd.to_datetime(
            meteo['time'], utc=True).dt.floor('H')
        # sort
        meteo = meteo.sort_values(by='time')

        # calculate temperature response and write into
        # the meteo df
        Response_calculator = Response(
            response_curve_type=response_curve_type,
            response_curve_parameters=params)

        # loop over pixels
        interpolated_pixel_results = []
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):

            lai_pixel_ts = prepare_lai_ts(lai_pixel_ts)
            # merge with meteo
            meteo_pixel = pd.merge(meteo, lai_pixel_ts, on='time', how='left')

            # STEP 1: Data Assimilation using Ensemble Kalman Filter
            # setup Ensemble Kalman Filter
            enskf = EnsembleKalmanFilter(
                state_vector=meteo_pixel,
                response=Response_calculator,
                n_sim=n_sim)
            # run the filter to assimilate data
            enskf.run()

            # STEP 2: Interpolate between the assimilated points
            # get assimilated results at the measurement values
            # and interpolate between them using scaled temperature
            # response to get a continuous LAI time series without
            # breaks resulting from the assimilation
            measurement_indices = meteo_pixel[
                meteo_pixel['lai'].notnull()]['time'].tolist()

            meteo_pixel['reconstructed_lai_mean'] = np.nan
            meteo_pixel['reconstructed_lai_std'] = np.nan
            meteo_pixel.index = meteo_pixel['time']

            # get the assimilated LAI values
            # ignore the last element as we do not have an uncertainty
            # estimate for it
            for measurement_index in measurement_indices[:-1]:
                assimilated_lai_values = \
                    enskf.new_states.loc[measurement_index].iloc[-1]
                # get mean and standard deviation of the ensemble
                # at the measurement point for which an S2 observation
                # is available
                assimilated_lai_value_mean = \
                    np.mean(assimilated_lai_values)
                assimilated_lai_value_std = \
                    np.std(assimilated_lai_values)
                meteo_pixel.loc[measurement_index,
                                'reconstructed_lai_mean'] = \
                    assimilated_lai_value_mean
                meteo_pixel.loc[measurement_index,
                                'reconstructed_lai_std'] = \
                    assimilated_lai_value_std

            # interpolate between the assimilated points
            # using the scaled temperature response
            # between the first and last interpolated point
            measurement_indices = [
                measurement_indices[0], measurement_indices[-2]]
            model_sims_between_points = \
                interpolate_between_assimilated_points(
                    measurement_index=measurement_indices,
                    meteo_pixel=meteo_pixel,
                    response=Response_calculator)

            # save results to DataFrame
            lai_interpolated_df = pd.DataFrame({
                'time': model_sims_between_points['time'],
                'lai': model_sims_between_points[
                    'reconstructed_lai_mean'],
                'lai_minus_std': model_sims_between_points[
                    'reconstructed_lai_mean'] - model_sims_between_points[
                        'reconstructed_lai_std'],
                'lai_plus_std': model_sims_between_points[
                    'reconstructed_lai_mean'] + model_sims_between_points[
                        'reconstructed_lai_std'],
                'y': pixel_coords[0],
                'x': pixel_coords[1]
            })
            interpolated_pixel_results.append(lai_interpolated_df)

        # concatenate the results for all pixels
        interpolated_pixel_results_parcel = pd.concat(
            interpolated_pixel_results, ignore_index=True)
        # correct the coordinates as xarray shifts them to the center
        # we fix the pixel resolution to 10 meters (S2 resolution)
        interpolated_pixel_results_parcel['y'] = \
            interpolated_pixel_results_parcel['y'] - 5  # meters
        interpolated_pixel_results_parcel['x'] = \
            interpolated_pixel_results_parcel['x'] + 5  # meters

        sc = SceneCollection()
        for time_stamp in interpolated_pixel_results_parcel.time.unique():
            # get the data for the current time stamp
            data = interpolated_pixel_results_parcel[
                interpolated_pixel_results_parcel.time == time_stamp].copy()

            # convert to eodal RasterCollection
            # reconstruct geoinfo
            geo_info = GeoInfo(
                epsg=32632,
                ulx=data.x.min(),
                uly=data.y.max(),
                pixres_x=10,
                pixres_y=-10
            )

            data_gdf = gpd.GeoDataFrame(
                data,
                geometry=gpd.points_from_xy(data.x, data.y),
                crs=geo_info.epsg
            )
            rc = RasterCollection()
            for band_name in ['lai',
                              'lai_minus_std',
                              'lai_plus_std']:
                band = Band.from_vector(
                    vector_features=data_gdf,
                    geo_info=geo_info,
                    band_name_src=band_name,
                    band_name_dst=band_name,
                    nodata_dst=np.nan
                )
                rc.add_band(band)
            rc.scene_properties = SceneProperties(
                acquisition_time=time_stamp
            )
            sc.add_scene(rc)

        # save the SceneCollection as pickled object
        sc = sc.sort()
        fname_pkl = output_dir.joinpath(
            f'{covariate_granularity}_lai.pkl')
        with open(fname_pkl, 'wb') as dst:
            dst.write(sc.to_pickle())

        logger.info(
            f'Interpolated {parcel_dir.name} to ' +
            f'{covariate_granularity} LAI values' +
            f'using {response_curve_type} response curve')


if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('results/validation_sites')

    dose_response_parameters = Path(
        'results/dose_reponse_in-situ/output/parameter_model')  # noqa: E501

    response_curve_types = ['asymptotic', 'non_linear']  # 'WangEngels'
    covariate_granularities = ["hourly", "daily"]

    for response_curve_type in response_curve_types:
        for covariate_granularity in covariate_granularities:
            apply_temperature_response(
                parcel_lai_dir=parcel_lai_dir,
                dose_response_parameters=dose_response_parameters,
                response_curve_type=response_curve_type,
                covariate_granularity=covariate_granularity)
