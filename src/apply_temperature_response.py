'''
Created on Apr 21, 2023

@author: Flavian Tschurr and Lukas Valentin Graf
'''

import geopandas as gpd
import pandas as pd
import numpy as np

from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path
from typing import List

from temperature_response_uncertainty import add_noise_to_temperature


logger = get_settings().logger

# set seed to make results reproducible
np.random.seed(42)

# noise level for temperature data
noise_level = 5  # in percent
# uncertainty in LAI data (relative)
lai_uncertainty = 5  # in percent


class Response:
    def __init__(self, response_curve_type, response_curve_parameters):
        self.response_cruve_type = response_curve_type
        self.params = response_curve_parameters

    def non_linear_response(self, env_variate):
        '''
        env_variate: value of an environmental covariate
        base_value: estimated value, start of the linear growing phase
        slope: estimated value, slope of the linear phase
        description: broken stick model according to an env variable
        '''

        base_value = self.params.get('base_value', 0)
        slope = self.params.get('slope_value', 0)

        y = (env_variate - base_value) * slope
        y = y if env_variate > base_value else 0.
        return y

    def asymptotic_response(self, env_variate):
        """
        Calculates the asymptotic response for a given input variable.

        Args:
        env_variate: input variable
        Asym: a numeric parameter representing the horizontal asymptote on
        the right side (very large values of input).
        lrc: a numeric parameter representing the natural logarithm of the
        rate constant.
        c0: a numeric parameter representing the env_variate for which the
        response is zero.

        Returns:
        A numpy array containing the asymptotic response values.
        """
        Asym = self.params.get('Asym_value', 0)
        lrc = self.params.get('lrc_value', 0)
        c0 = self.params.get('c0_value', 0)

        y = Asym * (1. - np.exp(-np.exp(lrc) * (env_variate - c0)))
        y = np.where(y > 0., y, 0.)  # no negative growth
        return y

    def wang_engels_response(self, env_variate):
        """
        Calculates the Wang-Engels response for a given input variable.

        Args:
            env_variate: effective env_variable value

        Returns:
            A numpy array containing the Wang-Engels response values.
        """
        xmin = self.params['xmin_value']
        xopt = self.params['xopt_value']
        xmax = self.params['xmax_value']

        alpha = np.log(2.) / np.log((xmax - xmin) / (xopt - xmin))

        if xmin <= env_variate <= xmax:
            y = (2. * (env_variate - xmin) ** alpha *
                 (xopt - xmin) ** alpha - (env_variate - xmin) **
                 (2. * alpha)) / \
                ((xopt - xmin) ** (2. * alpha))
        else:
            y = 0

        return y

    def get_response(self, env_variates):
        response_fun = getattr(
            Response, f'{self.response_cruve_type}_response')
        response = []
        for env_variate in env_variates:
            response.append(response_fun(self, env_variate))
        return response


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


def ensemble_kalman_filtering(
        measurement_index: List[int],
        meteo_pixel: pd.DataFrame,
        n_sim: int,
        Response_calculator: Response
) -> pd.DataFrame:
    """
    Ensemble Kalman filtering for assimilating satellite-derived
    LAI values into a temperature response function.

    Parameters
    ----------
    measurement_index : List[int]
        Indices of the LAI time series where measurements are available.
    meteo_pixel : pd.DataFrame
        Meteo data for the pixel with joined LAI values from satellite
        observations.
    n_sim : int
        Number of simulations.
    Response_calculatior : Response
        Response calculator.
    return : pd.DataFrame
        assimilated LAI time series.
    """
    model_sims_between_points = []
    lai_value_sim_start = None
    Aa = None
    for i in range(len(measurement_index)-1):
        meteo_time_window = meteo_pixel.loc[
            measurement_index[i]:measurement_index[(i+1)]].copy()

        # set starting LAI value for simulation
        # for the first observation this will be the measured value
        if i == 0:
            lai_value_sim_start = np.zeros((1, n_sim), dtype=float)
        # for the other observations this will be the assimilated
        # LAI value
        else:
            lai_value_sim_start = Aa[0, :]

        # loop over meteo data to run ensembles of temperature response
        # required for the ensemble Kalman filter
        lai_modelled = []
        model_sim = []
        for sim in range(n_sim):
            _meteo = meteo_time_window.copy()
            # add noise to temperature
            _meteo['T_mean'] = add_noise_to_temperature(
                _meteo['T_mean'].values, noise_level=noise_level)
            _meteo['temp_response'] = \
                Response_calculator.get_response(_meteo['T_mean'])
            # set response to lai_value_sim_start
            _meteo['temp_response_cumsum'] = \
                _meteo['temp_response'].cumsum()
            _meteo['temp_response_cumsum'] = \
                _meteo['temp_response_cumsum'] + \
                lai_value_sim_start[0, sim]
            # the cumulative sum of the temperature response at
            # i+1 is the modelled LAI stage at i+1
            lai_modelled.append(
                _meteo['temp_response_cumsum'].values[-1])
            # zip temperature response and time
            model_sim_time = dict(
                zip(_meteo.time, _meteo.temp_response_cumsum))
            model_sim.append(model_sim_time)

        model_sim_df = pd.DataFrame(model_sim).T
        model_sim_df.index = meteo_time_window['time']
        model_sims_between_points.append(model_sim_df)

        # calculate the updates for the next model run
        # using the ensemble Kalman filter

        # get the model stage A
        A_df = pd.DataFrame(lai_modelled)
        A = np.matrix(np.asarray(lai_modelled))  # .T
        # compute the variance within the ensemble A, P_e
        P_e = np.matrix(A_df.cov())

        # get the mean and covariance of the satellite observations
        lai_value = meteo_time_window['lai'].iloc[-1]
        lai_std = lai_uncertainty * 0.01 * lai_value
        perturbed_lai = np.random.normal(lai_value, lai_std, (n_sim))
        D = np.matrix(perturbed_lai)  # .T
        R_e = np.matrix(pd.DataFrame(perturbed_lai).cov())

        # Here we compute the Kalman gain
        H = np.identity(1)  # len(obs) in the original code
        K1 = P_e * (H.T)
        K2 = (H * P_e) * H.T
        K = K1 * ((K2 + R_e).I)

        # Here we compute the analysed states that will be used to
        # reinitialise the model at the next time step
        Aa = A + K * (D - (H * A))

    model_sims_between_points = pd.concat(
        model_sims_between_points, axis=0)

    return model_sims_between_points


def rescale(val, in_min, in_max, out_min, out_max):
    return out_min + (val - in_min) * ((out_max - out_min) / (in_max - in_min))


def interpolate_lai(
        measurement_index: List[int],
        meteo_pixel: pd.DataFrame,
        Response_calculator: Response
) -> pd.DataFrame:
    """
    Interpolate LAI values between satellite observations.
    """
    model_sims_between_points = []
    # loop over measurement points
    for i in range(len(measurement_index)-1):
        meteo_time_window = meteo_pixel.loc[
            measurement_index[i]:measurement_index[(i+1)]].copy()
        # calculate the temperature response
        meteo_time_window['temp_response'] = \
            Response_calculator.get_response(
                meteo_time_window['T_mean'])
        # get cumulative sum of temperature response
        meteo_time_window['temp_response_cumsum'] = \
            meteo_time_window['temp_response'].cumsum()
        # scale values between lai_value_start and lai_value_end
        in_min = meteo_time_window['temp_response_cumsum'].iloc[0]
        in_max = meteo_time_window['temp_response_cumsum'].iloc[-1]
        out_min = meteo_time_window['lai'].iloc[0]
        out_max = meteo_time_window['lai'].iloc[-1]
        out_range = out_max - out_min
        if out_range < 0:
            continue
        meteo_time_window['interpolated'] = \
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
        n_sim=50):
    """
    """
    # read in dose response paramters
    # TODO: change back once bug has been fixed
    # path_paramters = Path.joinpath(
    #     dose_response_parameters,
    #     response_curve_type,
    #     f'{response_curve_type}_granularity_{covariate_granularity}' +
    #     '_parameter_T_mean.csv')
    path_paramters = Path.joinpath(
        dose_response_parameters,
        response_curve_type,
        f'{response_curve_type}_variable_delta_LAI' +
        '_parameter_T_mean_location_CH_Bramenwies.csv')

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
            lai['time'], format='%Y-%m-%d %H:%M:%S', utc=True).dt.floor('H')

        # meteorological data
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
            measurement_index = meteo_pixel['lai'].notna()
            measurement_index = meteo_pixel[
                measurement_index].index.tolist()
            meteo_pixel['interpolation'] = pd.Series(dtype=float)

            if (len(measurement_index) <= 1):
                continue

            # calculate cumulative dose response between two
            # consecutive measurement timepoints
            model_sims_between_points = interpolate_lai(
                measurement_index=measurement_index,
                meteo_pixel=meteo_pixel,
                Response_calculator=Response_calculator)

            lai_interpolated_df = pd.DataFrame({
                'time': model_sims_between_points['time'],
                'lai': model_sims_between_points['interpolated'].values,
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
            band = Band.from_vector(
                vector_features=data_gdf,
                geo_info=geo_info,
                band_name_src='lai',
                band_name_dst=str(time_stamp),
                nodata_dst=np.nan
            )
            rc = RasterCollection()
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
            f'{covariate_granularity} LAI values')


if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('results/validation_sites')

    dose_response_parameters = Path(
        'results/dose_reponse_in-situ/output/parameter_model')  # noqa: E501

    response_curve_type = "non_linear"   # non_linear, wangengels
    # response_curve_type = "asymptotic"

    covariate_granularity = "hourly"

    apply_temperature_response(
        parcel_lai_dir=parcel_lai_dir,
        dose_response_parameters=dose_response_parameters,
        response_curve_type=response_curve_type,
        covariate_granularity=covariate_granularity)
