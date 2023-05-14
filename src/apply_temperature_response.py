'''
Created on Apr 21, 2023

@author: graflu
'''

import pandas as pd
import numpy as np

import os
native_os_path_join = os.path.join
def modified_join(*args, **kwargs):
    return native_os_path_join(*args, **kwargs).replace('\\', '/')
os.path.join = modified_join

import datetime
from pathlib import Path

class Response:
    def __init__(self, response_curve_type,response_curve_parameters):
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
        Asym: a numeric parameter representing the horizontal asymptote on the right side (very large values of input).
        lrc: a numeric parameter representing the natural logarithm of the rate constant.
        c0: a numeric parameter representing the env_variate for which the response is zero.

        Returns:
        A numpy array containing the asymptotic response values.
        """
        Asym = self.params.get('Asym_value', 0)
        lrc = self.params.get('lrc_value', 0)
        c0 = self.params.get('c0_value', 0)

        y = Asym * (1. - np.exp(-np.exp(lrc) * (env_variate - c0)))
        y = np.where(y > 0., y, 0.) # no negative growth
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
            y = (2. * (env_variate - xmin) ** alpha * (xopt - xmin) ** alpha - (env_variate - xmin) ** (2. * alpha)) / \
                ((xopt - xmin) ** (2. * alpha))
        else:
            y = 0

        return y

    def get_response(self, env_variates):
        response_fun = getattr(Response, f'{self.response_cruve_type}_response')
        response = []
        for env_variate in env_variates:
            response.append(response_fun(self, env_variate))
        return response


def apply_temperature_response(parcel_lai_dir: Path,
                               dose_response_parameters: Path,
                                response_curve_type,
                               covariate_granularity):
    """
    """
    # read in dose response paramters
    path_paramters = os.path.join(dose_response_parameters,response_curve_type,f'{response_curve_type}_granularity_hourly_parameter_T_mean.csv')
    # dose_response_parameters.glob('*')
    # path_paramters = dose_response_parameters.joinpath(response_curve_type,f'{response_curve_type}_variable_delta_LAI_smooth_parameter_T_mean_location_CH_Bramenwies.csv')

    params = pd.read_csv(path_paramters)
    params = dict(zip(params['parameter_name'], params['parameter_value']))
    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):
        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        lai['coords_combined'] = lai['x'].astype(str).str.cat(lai['y'].astype(str),sep="_")
        lai['time'] = pd.to_datetime(lai['time'], utc=True).dt.floor('H')

        # meteorological data
        fpath_meteo = parcel_dir.joinpath('hourly_mean_temperature.csv')
        meteo = pd.read_csv(fpath_meteo)
        # ensure timestamp format
        meteo['time'] = pd.to_datetime(meteo['time'], utc=True).dt.floor('H')
        # sort
        meteo = meteo.sort_values(by='time')

        # calculate temperature response and write into the meteo df
        Response_calculator = Response(response_curve_type = response_curve_type,
                                       response_curve_parameters = params)
        meteo['temp_response'] = Response_calculator.get_response(meteo['T_mean'])


        # loop over pixels
        for coords in lai['coords_combined'].unique():
            onePixel = lai[lai['coords_combined'] == coords]
            onePixel = onePixel.sort_values(by='time')
            # merge meteo df
            meteo_pixel = pd.merge(meteo, onePixel, on='time', how='left')
            # calculate cumulative response between two satelite measurements
            measurement_index = meteo_pixel['lai'].notna()
            measurement_index = meteo_pixel[measurement_index].index.tolist()
            meteo_pixel['interpolation'] = pd.Series(dtype=float)
            #calculate cumulative doiseresponse between two measurement timepoints
            if(len(measurement_index) <= 1):
                continue
            for i in range(len(measurement_index)-1):
                meteo_pixel.loc[measurement_index[i]:measurement_index[(i+1)], 'interpolation'] = np.cumsum(meteo_pixel.loc[measurement_index[i]:measurement_index[(i+1)], 'temp_response'])

        # outlier selection?
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
    parcel_lai_dir = Path('./results/test_sites_pixel_ts')

    dose_response_parameters = os.path.abspath('./results/dose_reponse_in-situ/output/parameter_model')
    # dose_response_parameters = Path('./results/dose_reponse_in-situ/output/parameter_model').resolve()

    response_curve_type = "non_linear"
    # response_curve_type = "asymptotic"


    covariate_granularity = "hourly"


    apply_temperature_response(parcel_lai_dir=parcel_lai_dir,
                               dose_response_parameters = dose_response_parameters,
                               response_curve_type = response_curve_type,
                               covariate_granularity = covariate_granularity)
