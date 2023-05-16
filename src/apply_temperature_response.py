'''
Created on Apr 21, 2023

@author: Flavian Tschurr and Lukas Valentin Graf
'''

import pandas as pd
import numpy as np

from pathlib import Path
from temperature_response_uncertainty import add_noise_to_temperature


# set seed to make results reproducible
np.random.seed(42)

# ensemble size for meteo data
n_sim = 100

# noise level for temperature data
noise_level = 2  # in percent


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


def apply_temperature_response(
        parcel_lai_dir: Path,
        dose_response_parameters: Path,
        response_curve_type,
        covariate_granularity):
    """
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
        temp_response_results = []
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):

            lai_pixel_ts = prepare_lai_ts(lai_pixel_ts)

            # calculate cumulative response between two satelite
            # measurements
            meteo_pixel = pd.merge(meteo, lai_pixel_ts, on='time', how='left')
            measurement_index = meteo_pixel['lai'].notna()
            measurement_index = meteo_pixel[
                measurement_index].index.tolist()
            meteo_pixel['interpolation'] = pd.Series(dtype=float)

            if (len(measurement_index) <= 1):
                continue

            # calculate cumulative dose response between two
            # consecutive measurement timepoints

            for i in range(len(measurement_index)-1):
                meteo_time_window = meteo_pixel.loc[
                    measurement_index[i]:measurement_index[(i+1)]].copy()
       
                # meteo_pixel.loc[
                #     measurement_index[i]:measurement_index[(i+1)],
                #     'interpolation'] = \
                #         np.cumsum(meteo_pixel.loc[
                #             measurement_index[i]: measurement_index[(i+1)],
                #             'temp_response'])

                # loop over meteo data to run ensembles of temperature response
                # required for the ensemble Kalman filter
                meteo_response_cumsum = []
                for sim in range(n_sim):
                    _meteo = meteo_time_window.copy()
                    # add noise to temperature
                    _meteo['T_mean'] = add_noise_to_temperature(
                        _meteo['T_mean'].values, noise_level=noise_level)
                    _meteo['temp_response'] = \
                        Response_calculator.get_response(_meteo['T_mean'])
                    meteo_response_cumsum.append(
                        np.cumsum(_meteo['temp_response']).values[-1])

                # get the uncertainty in the temperature response
                # due to the uncertainty in the temperature
                response_cumsum_unc = np.std(meteo_response_cumsum)

                    


        # outlier selection? -> yes
        # scaling of interpolation?
        # temperature timing --> last day is cut of --> round to day?

        # import matplotlib.pyplot as plt
        #
        # # Plot the time series
        # plt.plot(meteo_pixel['time'], meteo_pixel['lai'],marker='o', label='LAI')
        # plt.plot(meteo_pixel['time'], meteo_pixel['T_mean'], label='T_mean')
        # plt.plot(meteo_pixel['time'], meteo_pixel['temp_response'], label='Response')
        # plt.plot(meteo_pixel['time'], meteo_pixel['interpolation'], label='interpolation')
        #
        #
        # # Add legend and axis labels
        # plt.legend()
        # plt.xlabel('Time')
        # plt.ylabel('Values')
        #
        # # Show the plot
        # plt.show()

        # # loop over single pixels
        # for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):
        #     # pixel_coords are the coordinates of the pixel
        #     pixel_coords
        #     # the actual lai data. The meteo data can be joined on the time column
        #     lai_pixel_ts


if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('results/validation_sites')

    dose_response_parameters = Path(
        'results/dose_reponse_in-situ/output/parameter_model')
    # dose_response_parameters = Path('./results/dose_reponse_in-situ/output/parameter_model').resolve()

    response_curve_type = "non_linear"
    # response_curve_type = "asymptotic"

    covariate_granularity = "hourly"

    apply_temperature_response(
        parcel_lai_dir=parcel_lai_dir,
        dose_response_parameters=dose_response_parameters,
        response_curve_type=response_curve_type,
        covariate_granularity=covariate_granularity)
