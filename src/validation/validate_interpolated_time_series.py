"""
Plot interpolated LAI time series
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings

from eodal.core.scene import SceneCollection
from pathlib import Path


models = ['non_linear', 'sigmoid']
bbch_range = (30, 59)
black_listed_parcels = ['Bramenwies']

plt.style.use('bmh')
warnings.filterwarnings('ignore')


def validate_interpolated_time_series(
        model_output_dir: Path,
        validation_data_dir: Path,
) -> None:
    """
    Validate the interpolated LAI time series.

    Parameters
    ----------
    model_output_dir : Path
        Path to the directory containing the model output.
    validation_data_dir : Path
        Path to the directory containing the validation data.
    """
    # get the validation data
    join_cols = ['date', 'location', 'parcel', 'point_id']
    lai = gpd.read_file(validation_data_dir / 'in-situ_glai.gpkg')
    lai.date = pd.to_datetime(lai.date).dt.tz_localize('Europe/Zurich')
    lai_cols = ['lai', 'geometry'] + join_cols
    bbch = gpd.read_file(validation_data_dir / 'in-situ_bbch.gpkg')
    bbch.date = pd.to_datetime(bbch.date).dt.tz_localize('Europe/Zurich')
    bbch_cols = ['BBCH Rating'] + join_cols
    # join the dataframes on date, location, parcel and point_id
    val_df = lai[lai_cols].merge(bbch[bbch_cols], on=join_cols, how='left')
    val_df.rename(columns={'lai': 'lai_in-situ'}, inplace=True)

    # loop over directories in the model output directory
    for site_dir in model_output_dir.glob('*'):
        site = site_dir.name.split('_')[1]
        site_val_df = val_df[val_df.location == site].copy()
        # filter by BBCH range
        site_val_df = site_val_df[
            (site_val_df['BBCH Rating'] >= bbch_range[0]) &
            (site_val_df['BBCH Rating'] <= bbch_range[1])
        ].copy()
        # filter black listed parcels
        site_val_df = site_val_df[
            ~site_val_df.parcel.isin(black_listed_parcels)
        ].copy()
        for model in models:
            model_dir = site_dir.joinpath(model)
            # read the scene collection from pickle (thre could be more than
            # one with different levels of temporal granularity)
            pixel_vals_list = []
            for fpath_scoll in model_dir.glob('*lai.pkl'):
                granularity = fpath_scoll.name.split('_')[0]
                round_to = '1H' if granularity == 'hourly' else '1D'

                scoll = SceneCollection.from_pickle(fpath_scoll)
                min_date = pd.to_datetime(scoll.timestamps[0], utc=True)
                max_date = pd.to_datetime(scoll.timestamps[-1], utc=True)
                # loop over dates for which in-situ data is available
                # and extract the interpolated LAI values
                for date, site_val_df_date in site_val_df.groupby('date'):
                    date_rounded = date.round(round_to).tz_convert('UTC')
                    # continue if date is not between min and max date
                    if not (min_date <= date_rounded <= max_date):
                        continue

                    # get the interpolated LAI value by timestamp
                    if granularity == 'hourly':
                        scene = scoll[date_rounded.__str__()]
                    elif granularity == 'daily':
                        date_date = date_rounded.date()
                        # get the corresponding date in the scene collection
                        for timestamp in scoll.timestamps:
                            if pd.to_datetime(timestamp).date() == date_date:
                                scene = scoll[timestamp]
                                break

                    # get the pixel values at the in-situ points
                    pixel_vals = scene.get_pixels(
                        vector_features=site_val_df_date)
                    pixel_vals = pixel_vals.rename(
                        columns={'lai': f'lai_{model}'})
                    pixel_vals_list.append(pixel_vals)

                # concatenate the pixel values
                if len(pixel_vals_list) == 0:
                    continue
                pixel_vals_df = pd.concat(pixel_vals_list)
                pixel_vals_df['model'] = model
                pixel_vals_df['granularity'] = granularity

                # save the pixel values
                fpath_out = model_dir / f'{granularity}_lai_validation.csv'
                pixel_vals_df.to_csv(fpath_out, index=False)

                # scatter plot lai vs. interpolated lai
                f, ax = plt.subplots(figsize=(8, 8), ncols=1, sharex=True,
                                     sharey=True)
                sns.scatterplot(
                    data=pixel_vals_df,
                    x='lai_in-situ',
                    y=f'lai_{model}',
                    ax=ax)
 
                ax.set_xlabel('in-situ LAI' + r' [$m^2$ $m^{-2}$]')
                ax.set_ylabel('S2-derived interpolated LAI' +
                              r' [$m^2$ $m^{-2}$]')
                ax.set_xlim(0, 8)
                ax.set_ylim(0, 8)
                ax.plot([0, 8], [0, 8], color='black', linestyle='--')
                ax.set_title(
                    f'{site} {model} {granularity}\n' +
                    f'N={len(pixel_vals_df)}')
                plt.tight_layout()
                f.savefig(
                    model_dir / f'{granularity}_lai_validation.png',
                    dpi=300)
                plt.close(f)

                print(f'{site} {model} {granularity} --> done')


if __name__ == '__main__':

    model_output_dir = Path('results/validation_sites')

    # in-situ validation data
    years = [2022]

    # go through the years
    for year in years:
        validation_data_dir = Path('data/in-situ') / str(year)
        validate_interpolated_time_series(
            model_output_dir=model_output_dir,
            validation_data_dir=validation_data_dir)
