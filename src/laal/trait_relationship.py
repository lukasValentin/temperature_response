"""
Relationships between traits at the Strickhof and
the Witzwil test site over multiple years.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

plt.style.use('bmh')
test_sites = ['Strickhof', 'Witzwil']


def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_


def get_parcel_names(test_pixels_dir: Path) -> list:
    """
    Get the names of available parcels.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    return : list
        List of parcel names.
    """
    return [parcel.name.split('_')[1] for parcel
            in test_pixels_dir.iterdir()]


def get_field_calendars(sites_dir: Path) -> pd.DataFrame:
    """
    Get the field calendars of the test sites.

    Parameters
    ----------
    sites_dir : Path
        Path to the directory with the test site field calendar data.
    return : pd.DataFrame
        Field calendars of the test sites.
    """
    res = []
    for test_site in test_sites:
        df = pd.read_csv(sites_dir.joinpath(f'{test_site}.csv'))
        df['test_site'] = test_site
        res.append(df)
    return pd.concat(res)


def get_parcel_data(test_pixels_dir: Path, parcel_name: str) -> pd.DataFrame:
    """
    Get the data of a specific parcel.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    parcel_name : str
        Name of the parcel.
    return : pd.DataFrame
        Data of the parcel.
    """
    res_parcel = []
    for parcel in test_pixels_dir.iterdir():
        if parcel_name == parcel.name.split('_')[1]:
            # read LAI and CCC data
            df_lai = pd.read_csv(parcel.joinpath('raw_lai_values.csv'))
            df_lai.time = pd.to_datetime(
                df_lai.time, format='%Y-%m-%d %H:%M:%S', utc=True)
            df_ccc = pd.read_csv(parcel.joinpath('raw_ccc_values.csv'))
            df_ccc.time = pd.to_datetime(
                df_ccc.time, format='%Y-%m-%d %H:%M:%S', utc=True)
            df = pd.merge(df_lai, df_ccc, on=['time', 'x', 'y'])
            # calculate CAB
            df['cab'] = df['ccc'] / df['lai'] * 100
            res_parcel.append(df)

    return pd.concat(res_parcel)


def plot_yearly_relationships(
        df: pd.DataFrame,
        df_field_calendars: pd.DataFrame,
        parcel_name: str,
        years: list,
        idx: int,
        output_dir: Path) -> None:
    """
    Plot the relationships between LAI, CCC, and CAB for a specific
    parcel for a single growing season.

    Parameters
    ----------
    df : pd.DataFrame
        S2 derived trait data of the parcel.
    df_field_calendars : pd.DataFrame
        Field calendars of the test sites.
    parcel_name : str
        Name of the parcel.
    years : list
        List of years.
    idx : int
        Index of the current year.
    output_dir : Path
        Path to the output directory.
    """

    # get the data for the current year
    # make sure to avoid index errors
    # it might happen that we have no data from the previous year
    # when the sowing was late or it was very cloudy
    if len(years) == 1:
        years = [years[idx] - 1, years[0]]
    df_year = df[
        (df.time.dt.year == years[idx]) |
        (df.time.dt.year == years[idx+1])].copy()
    # get the sowing and harvest dates for parcel
    df_parcel = df_field_calendars[
        df_field_calendars.name.astype(str) == parcel_name].copy()
    sowing_date = pd.to_datetime(df_parcel[
        pd.to_datetime(
            df_parcel.sowing_date).dt.year == years[idx]
        ].iloc[0].sowing_date).date()
    harvest_date = pd.to_datetime(df_parcel[
        pd.to_datetime(
            df_parcel.harvest_date).dt.year == years[idx+1]
        ].iloc[0].harvest_date).date()

    # calculate the mean values of lai, ccc, and cab for those
    # records that have the same time stamp and x, y coordinates
    # but different pheno_phase values
    df_year = df_year.groupby(['time', 'x', 'y']).mean().reset_index()

    # calculate days after sowing (das)
    df_year['das'] = (df_year.time.dt.date - sowing_date).dt.days

    # plot the relationships
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # CCC - GLAI relationship
    ax[0].scatter(df_year['lai'], df_year['ccc'], s=1)
    ax[0].set_xlabel(r'S2 derived GLAI [$m^2$ $m^{-2}$]')
    ax[0].set_ylabel(r'S2 derived CCC [$g$ $m^{-2}$]')
    ax[0].set_xlim([0, 8])
    ax[0].set_ylim([0, 4])

    # CAB - GLAI relationship
    ax[1].scatter(df_year['lai'], df_year['cab'], s=1)
    ax[1].set_xlabel(r'S2 derived GLAI [$m^2$ $m^{-2}$]')
    ax[1].set_ylabel(r'S2 derived Cab [$\mu g$ $cm^{-2}$]')
    ax[1].set_xlim([0, 8])
    ax[1].set_ylim([0, 80])

    plt.suptitle(
        f'{df_parcel.test_site.iloc[0]} {parcel_name}'
        f': {sowing_date} - {harvest_date}'
        f'\n(Variety: {df_parcel.genotype.iloc[0]})')

    fname_scatter = output_dir.joinpath(
        f'{df_parcel.test_site.iloc[0]}_{parcel_name}_'
        f'scatter_traits.png')
    fig.savefig(fname_scatter, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # plot the time series of the traits in das
    fig, ax = plt.subplots(
        nrows=1, ncols=3, figsize=(15, 5),
        sharex=True)
    # use the percentile values for the time series
    df_year_median = df_year[['das', 'lai', 'ccc', 'cab']].groupby(
        'das').quantile(q=0.5).reset_index()
    df_year_q05 = df_year[['das', 'lai', 'ccc', 'cab']].groupby(
        'das').quantile(q=0.05).reset_index()
    df_year_q95 = df_year[['das', 'lai', 'ccc', 'cab']].groupby(
        'das').quantile(q=0.95).reset_index()

    # LAI time series
    ax[0].plot(
        df_year_median['das'], df_year_median['lai'], lw=0.5,
        label='median')
    ax[0].set_xlabel('Days after sowing')
    ax[0].set_ylabel(r'S2 derived GLAI [$m^2$ $m^{-2}$]')
    ax[0].set_ylim([0, 8])
    # plot the 5th and 95th percentile spread
    ax[0].fill_between(
        df_year_q05['das'], df_year_q05['lai'],
        df_year_q95['lai'], alpha=0.5,
        label='5th and 95th percentile')
    # CCC time series
    ax[1].plot(
        df_year_median['das'], df_year_median['ccc'], lw=0.5,
        label='median')
    ax[1].set_xlabel('Days after sowing')
    ax[1].set_ylabel(r'S2 derived CCC [$g$ $m^{-2}$]')
    ax[1].set_ylim([0, 4])
    # plot the 5th and 95th percentile spread
    ax[1].fill_between(
        df_year_q05['das'], df_year_q05['ccc'],
        df_year_q95['ccc'], alpha=0.5,
        label='5th and 95th percentile')
    # CAB time series
    ax[2].plot(
        df_year_median['das'], df_year_median['cab'], lw=0.5,
        label='median')
    ax[2].set_xlabel('Days after sowing')
    ax[2].set_ylabel(r'S2 derived Cab [$\mu g$ $cm^{-2}$]')
    ax[2].set_ylim([0, 80])
    # plot the 5th and 95th percentile spread
    ax[2].fill_between(
        df_year_q05['das'], df_year_q05['cab'],
        df_year_q95['cab'], alpha=0.5,
        label='5th and 95th percentile')
    ax[2].legend(loc='upper left', frameon=False)

    plt.suptitle(
        f'{df_parcel.test_site.iloc[0]} {parcel_name}'
        f': {sowing_date} - {harvest_date}'
        f'\n(Variety: {df_parcel.genotype.iloc[0]})')
    fname_ts = output_dir.joinpath(
        f'{df_parcel.test_site.iloc[0]}_{parcel_name}_'
        f'ts_traits.png')
    fig.savefig(fname_ts, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(test_pixels_dir: Path, sites_dir: Path,
         output_dir: Path):
    """
    Get the relationships between traits.

    Parameters
    ----------
    test_pixels_dir : Path
        Path to the directory with the test site pixel time series.
    sites_dir : Path
        Path to the directory with the test site field calendar data.
    output_dir : Path
        Path to the directory where the output should be saved.
    """
    # get the parcel names
    parcel_names = np.unique(get_parcel_names(test_pixels_dir))

    # get the sowing and harvest dates by parcel
    df_field_calendars = get_field_calendars(sites_dir)

    # loop over parcels and extract the data
    for parcel_name in parcel_names:
        df = get_parcel_data(test_pixels_dir, parcel_name)
        # check how many years are in the data. Some parcels
        # have 2 growing seasons (i.e., 4 years)
        df.time = pd.to_datetime(df.time, format='%Y-%m-%d %H:%M:%S', utc=True)
        years = np.sort(np.unique(df.time.dt.year))

        for idx in range(0, len(years), 2):
            try:
                plot_yearly_relationships(
                    df=df,
                    df_field_calendars=df_field_calendars,
                    parcel_name=parcel_name,
                    years=years,
                    idx=idx,
                    output_dir=output_dir)
            except (IndexError) as e:
                print(f'Could not plot {parcel_name} for {years[idx]} '
                      f'and {years[idx+1]}: {e}')
                continue


if __name__ == '__main__':

    test_pixels_dir = Path('results/test_sites_pixel_ts')
    sites_dir = Path('data/Test_Sites')
    output_dir = Path('analysis/trait_relationships')
    output_dir.mkdir(exist_ok=True, parents=True)

    main(
         test_pixels_dir=test_pixels_dir,
         sites_dir=sites_dir,
         output_dir=output_dir)
