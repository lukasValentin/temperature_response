"""
Baseline using a sigmoid model fitting on the raw LAI values.
The baseline model is only applied on the validation set.

@author: Lukas Valentin Graf
"""

import geopandas as gpd
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from eodal.config import get_settings
from eodal.core.band import Band, GeoInfo
from eodal.core.raster import RasterCollection, SceneProperties
from eodal.core.scene import SceneCollection
from pathlib import Path
from scipy.optimize import curve_fit
from shapely.geometry import Polygon

warnings.filterwarnings('ignore')
logger = get_settings().logger


def sigmoid(
        x: np.ndarray,
        L: float = 1,
        k: float = 1,
        x0: float = 0,
        b: float = 0
) -> np.ndarray:
    """
    Sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        Input array.
     L : float
        scales the output range.
    k : float
        scales the input range.
    x0 : float
        is the sigmoid's midpoint.
    b : float
        is the bias.

    Returns
    -------
    np.ndarray
        Sigmoid of the input array.
    """
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)


def fit_sigmoid(x: np.ndarray, y: np.ndarray):
    """
    Fit a sigmoid function to the data.

    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    y : np.ndarray
        Dependent variable.

    Returns
    -------
    np.ndarray
        Fitted sigmoid function.
    """
    # mandatory initial guess
    p0 = [max(y), np.median(x), 1, min(y)]
    popt, _ = curve_fit(sigmoid, x, y, p0, method='lm')
    return popt


def to_doy(time: pd.Series) -> pd.Series:
    """
    Convert time stamps to day of year
    """
    int_time = time.dt.day_of_year
    # substract the start time from all time stamps
    int_time = int_time - int_time.min()
    return int_time


def loop_pixels(parcel_lai_dir: Path):
    """
    Loop over pixels and apply the sigmoid model.

    Parameters
    ----------
    parcel_lai_dir : Path
        Directory with parcel LAI time series.
    """
    # loop over parcels and read the data
    for parcel_dir in parcel_lai_dir.glob('*'):

        # make an output directory
        output_dir = parcel_dir.joinpath('sigmoid_model')
        output_dir.mkdir(exist_ok=True)

        # leaf area index data
        fpath_lai = parcel_dir.joinpath('raw_lai_values.csv')
        lai = pd.read_csv(fpath_lai)
        lai['time'] = pd.to_datetime(
            lai['time'], format='%Y-%m-%d %H:%M:%S').dt.floor('H')
        # convert time to doy
        lai['doys'] = to_doy(lai['time'])
        # get the maximum extent of the site to make sure
        # all dates have the same extent
        min_x, min_y = lai[['x', 'y']].min()
        max_x, max_y = lai[['x', 'y']].max()
        # construct the reference shape
        reference_shape = Polygon(
            [(min_x, min_y), (min_x, max_y),
             (max_x, max_y), (max_x, min_y)])

        # loop over single pixels
        interpolated_pixel_results = []
        for pixel_coords, lai_pixel_ts in lai.groupby(['y', 'x']):

            lai_pixel_ts.sort_values(by='time', inplace=True)

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
            # remove nan values
            lai_values = np.delete(lai_values, nan_indices)

            # get doy
            doys = np.delete(
                lai_pixel_ts['doys'].values.copy(), nan_indices)
            n_days = doys.max()
            # normalize doy values between 0 and 1
            doys = doys / doys.max()

            # fit the sigmoid model
            try:
                popt = fit_sigmoid(
                    x=doys,
                    y=lai_values
                )
            except RuntimeError:
                continue

            # apply the sigmoid model to obtain LAI values in the desired
            # granularity
            time_stamps = np.arange(0, 1 + 1/n_days, 1/n_days)
            lai_interpolated = sigmoid(
                x=time_stamps,
                L=popt[0],
                k=popt[1],
                x0=popt[2],
                b=popt[3])

            # scale the doys back to the original dates
            first_date = lai_pixel_ts['time'].min()
            last_date = first_date + pd.Timedelta(days=n_days)
            daily_dates = pd.date_range(first_date, last_date, freq='D')

            # create a dataframe with the interpolated LAI values
            lai_interpolated_df = pd.DataFrame({
                'time': daily_dates,
                'lai': lai_interpolated,
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
                band_name_dst=str(time_stamp.date()),
                nodata_dst=np.nan,
                snap_bounds=reference_shape
            )
            rc = RasterCollection()
            rc.add_band(band)
            rc.scene_properties = SceneProperties(
                acquisition_time=time_stamp
            )
            sc.add_scene(rc)

            # plot the current scene
            f, ax = plt.subplots(figsize=(10, 10))
            band.plot(
                vmin=0,
                vmax=8,
                colormap='viridis',
                colorbar_label=r'GLAI [$m^2$ $m^{-2}$]',
                fontsize=18,
                ax=ax)
            f.savefig(
                output_dir.joinpath(f'{time_stamp.date()}.png'),
                dpi=300)
            plt.close(f)

        # save the SceneCollection as pickled object
        fname_pkl = output_dir.joinpath('daily_lai.pkl')
        with open(fname_pkl, 'wb') as dst:
            dst.write(sc.to_pickle())

        logger.info(f'Interpolated {parcel_dir.name} to daily LAI values')


if __name__ == '__main__':

    # directory with parcel LAI time series
    parcel_lai_dir = Path('./results/validation_sites')

    loop_pixels(parcel_lai_dir=parcel_lai_dir)
