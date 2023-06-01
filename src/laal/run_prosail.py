"""
Run PROSAIL RTM simulations in forward mode to generate look-up tables
for the stem elongation phase for those scenes extracted in
`laal.extract_s2_spectra.py`.
"""

import pandas as pd

from pathlib import Path


def run_prosail(
        fpath_lut_params: Path,
        sat_data_dir: Path
) -> None:
    """
    """
    # loop over the *metadata.csv files in the directory
    # these contain the scene metadata required to run PROSAIL
    for fpath_metadata in sat_data_dir.glob('*metadata.csv'):
        # read the data
        df_metadata = pd.read_csv(fpath_metadata)
        # loop over the rows in the metadata dataframe
        for _, row in df_metadata.iterrows():
            # extract the angles
            


if __name__ == '__main__':

    fpath_lut_params = Path(
        'src/lut_params/prosail_danner-etal_stemelongation-endofheading.csv')
    sat_data_dir = Path(
        '/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/CH')  # noqa: E501