'''
This module pre-calculates PROSAIL LUTs for different phenological macro-stages
of winter wheat at a set of discrete solar and viewing angles to speed up the
trait retrieval by means of RTM inversion later on (i.e., the LUTs are loaded
and interpolated if necessary from an archive of LUTs).

The sets of angles are based on >6 metadata analysis from Sentinel-2 scenes over
Switzerland.
'''

import numpy as np

from copy import deepcopy
from eodal.config import get_settings
from itertools import product
from pathlib import Path
from rtm_inv.core.lookup_table import generate_lut
from typing import Dict, List, Optional

solar_zenith_angles: np.ndarray = np.arange(start=20, stop=80, step=5)
sensor_zenith_angles: np.ndarray = np.arange(start=0, stop=15, step=3)
relative_azimuth_angles: np.ndarray = np.arange(start=0, stop=225, step=45)
angles: List[np.ndarray] = [solar_zenith_angles, sensor_zenith_angles, relative_azimuth_angles]

angle_combinations = list(product(*angles))
n_angles = len(angle_combinations)

# lookup table sizes
lut_sizes: Dict[str, int] = {
    'germination-endoftillering': 10000,
    'stemelongation-endofheading': 50000,
    'flowering-fruitdevelopment-plantdead': 50000
}

# setup logger
logger = get_settings().logger

def precalculate_luts(
    storage_dir: Path,
    lut_params_dir: Optional[Path] = Path('./lut_params'),
    sensors: Optional[List[str]] = ['Sentinel2A', 'Sentinel2B'],
    fpath_srf: Optional[Path] = Path('../data/S2-SRF_COPE-GSEG-EOPG-TN-15-0007_3.1.xlsx')
):
    """
    Precalculate PROSAIL LUTs for a set of discrete viewing and illumination
    angles for different phenological macro-stages of winter wheat.

    :param storage_dir:
        directory where to store the precalculated LUTs as pickled numpy objects
    :param lut_params_dir:
        directory with PROSAIL input parameters. Default is to use the parameters
        provided in this package.
    :param sensors:
        name of the sensors for which to precalculate the LUTs (i.e., the PROSAIL 1nm
        output is spectrally resampled to the spectral response of a sensor).
        Default is ['Sentinel2A', 'Sentinel2B'].
    :param fpath_srf:
        file-path to the spectral response function of the sensor(s). Default is the
        SRF of Sentinel2A and B provided by ESA. 
    """
    macro_stages = [x.name.split('_')[-1].split('.')[0] for x in lut_params_dir.glob('*.csv')]
    # the LUTs are stored by sensor and phenological macro-stage
    for sensor in sensors:
        storage_dir_sensor = storage_dir.joinpath(sensor)
        storage_dir_sensor.mkdir(exist_ok=True)
        for macro_stage in macro_stages:
            storage_dir_sensor_macrostage = storage_dir_sensor.joinpath(macro_stage)
            storage_dir_sensor_macrostage.mkdir(exist_ok=True)
            lut_params_macro_stage = next(lut_params_dir.glob(f'*{macro_stage}.csv'))
            # setup RTM configuration
            rtm_lut_config = {
                'lut_size': lut_sizes[macro_stage],
                'fpath_srf': fpath_srf,
                'remove_invalid_green_peaks': True,
                'sampling_method': 'FRS',
                'linearize_lai': False,
                'sensor': sensor,
                'lut_params': lut_params_macro_stage
            }
            # generate the LUTs for each angle combination
            for idx, angle_combination in enumerate(angle_combinations):
                sza, vza, psi = angle_combination
                lut_inp = deepcopy(rtm_lut_config)
                fpath = storage_dir_sensor_macrostage.joinpath(f'{sza}_{vza}_{psi}.pkl')
                # do not overwrite existing files
                if fpath.exists():
                    logger.info(f'{fpath.name} already exists -> skipping')
                    continue
                angle_dict = {
                    'solar_zenith_angle': sza,
                    'viewing_zenith_angle': vza,
                    'relative_azimuth_angle': psi
                }
                lut_inp.update(angle_dict)
                # actual LUT generation
                lut = generate_lut(**lut_inp)
                # save LUT as pickled object. The file-name is <sza>_<vza>_<psi>.pkl
                lut.to_pickle(fpath)
                logger.info(f'Processed {macro_stage} {fpath.name} ({idx+1}/{n_angles})')

if __name__ == '__main__':
    import sys
    sensors = [sys.argv[1]]

    storage_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/PROSAIL_LUTs')
    precalculate_luts(storage_dir=storage_dir, sensors=sensors)
