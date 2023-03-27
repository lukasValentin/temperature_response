'''
Load pre-calculated PROSAIL LUTs for inversion
'''

import multiprocessing
import numpy as np
import pandas as pd

from dask.distributed import Client
from numba import float32, int8, jit
from pathlib import Path
from scipy.interpolate import interpn
from typing import Dict, List, Tuple

def _get_weight(actual_angle: float, matching_angle: float, angle_spacing: float) -> float:
    """
    Get weight of the distance between actual and matching angle
    for the interpolation
    """
    return 1 - abs(actual_angle - matching_angle) / angle_spacing

def find_closest_angles(
    actual_angle: float,
    available_angles: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the two closest available angles for a given angle. The
    "distance" to angles is reported as a weight to be used for the
    interpolation.

    :param actual_angle:
        the actual (satellite) angle
    :param available_angles:
        angles for which the LUTs were pre-calculated
    :returns:
        closest two angles and weights for interpolation
    """
    matching_angles = available_angles[np.argsort(abs(actual_angle - available_angles))[0:2]]
    # assess weights of the matching angles. The closer the distance the higher
    # the weight. If an angle is exactly matched it's weight becomes 1 and the weight of the
    # other angle zero.
    angle_spacing = abs(matching_angles[0] - matching_angles[1])
    weights = []
    for matching_angle in matching_angles:
        weight = _get_weight(actual_angle, matching_angle, angle_spacing)
        weights.append(weight)
    weights = np.array(weights)
    return matching_angles, weights

@jit(
    float32[:,:,:](float32[:,:,:], float32[:,:], float32[:,:], int8),
    nopython=True,
    cache=True
)
def _interp_3d_numba(
    record: np.ndarray,
    record_df: np.ndarray,
    points_arr: np.ndarray,
    spectral_band_idx: int
):
    # fill the cube's edges with spectral values
    ii, kk, jj, tt = 0, 0, 0, 0
    for ii in range(2):
        for jj in range(2):
            for kk in range(2):
                for tt in range(8):
                    res = np.sign(
                        record_df[tt,-3:] - np.array([points_arr[0,ii], points_arr[1,jj], points_arr[2,kk]])
                    )
                    if (res == np.array([0,0,0])).all():
                        record[ii,jj,kk] = record_df[tt,spectral_band_idx]
                        break
    return record

def interp_3d(
    luts_to_interpolate: pd.DataFrame,
    ii: int,
    spectral_bands: List[str],
    traits: List[str],
    tts_angles: List[float],
    tto_angles: List[float],
    psi_angles: List[float],
    point: Tuple[float,float,float]
) -> Dict[str, float]:
    """
    The actual 3D interpolation of spectral values.

    :param record_df:
        DataFrame with spectral values to interpolate
    :param spectral_band:
        the spectral band to interpolate
    :param tts_angles:
        sun zenith angles between which to interpolate
    :param tto_angles:
        observer zenith angles between which to interpolate
    :param psi_angles:
        relative azimuth angles between which to interpolate
    :param point:
        actual tts, tto, psi angle combination
    :returns:
        interpolated values
    """
    # get all 8 records (angles and values)
    records = []
    sel_columns = traits + spectral_bands + ['tts', 'tto', 'psi']
    for jj in range(8):
        records.append(luts_to_interpolate.iloc[jj].data.loc[ii, sel_columns].copy())
    record_df = pd.concat(records, axis=1).T
    # reindex record_df
    record_df.index = [x for x in range(record_df.shape[0])]

    points = (
        np.linspace(min(tts_angles), max(tts_angles), num=2),
        np.linspace(min(tto_angles), max(tto_angles), num=2),
        np.linspace(min(psi_angles), max(psi_angles), num=2)
    )
    points_arr = np.array(list(points)).astype(np.float32)

    # actual interpolation per spectral band
    interpolated = {}
    for spectral_band in spectral_bands:
        spectral_band_idx = list(record_df.columns).index(spectral_band)
        record = np.empty(shape=(2,2,2), dtype=np.float32)
        _record_df = record_df.values.astype(np.float32)
        record = _interp_3d_numba(record, _record_df, points_arr, spectral_band_idx)
        # actual interpolation
        interpolated_value = interpn(points, record, point)[0]
        
        interpolated.update({spectral_band: interpolated_value})

    # append trait values
    for trait in traits:
        interpolated.update({trait: record_df[trait].iloc[0]})
    return interpolated
    

def load_lut(
    lut_dir: Path,
    tts: float,
    tto: float,
    psi: float,
    sensor: str,
    pheno_phase: str,
    sel_columns: List[str],
    client: Client
) -> pd.DataFrame:
    """
    Load LUT for a scene by interpolating between pre-calculated LUTs at discrete
    viewing and illumination angles.

    :param lut_dir:
        directory where the precalculated LUTs are stored (top-level)
    :param tts:
        actual solar zenith angle of the scene
    :param tto:
        actual observer zenith angle of the scene
    :param psi:
        actual relative azimuth angle of the scene
    :param sensor:
        name of the sensor (e.g., Sentinel2A) to select the correct LUTs
    :param pheno_phase:
        phenological phase to select the correct LUTs
    :param sel_columns:
        spectral bands and traits to load
    :param client:
        dask client for parallelization to speed-up interpolation
    """
    actual_lut_dir = lut_dir.joinpath(sensor).joinpath(pheno_phase)

    # extract spectral bands from sel_columns
    spectral_bands = [x for x in sel_columns if x.startswith('B')]
    traits = [x for x in sel_columns if x in sel_columns and x not in spectral_bands]
    # angle combination to interpolate column values to
    point = np.array([tts, tto, psi])

    # get available LUTs and their angles
    available_luts = pd.DataFrame([x for x in actual_lut_dir.glob('*.pkl')], columns=['fpath'])
    available_luts['tts'] = available_luts.fpath.apply(lambda x: float(x.name.split('_')[0]))
    available_luts['tto'] = available_luts.fpath.apply(lambda x: float(x.name.split('_')[1]))
    available_luts['psi'] = available_luts.fpath.apply(lambda x: float(x.name.split('_')[2].split('.')[0]))

    # closest angles and weights for interpolation
    sun_angles_lut = available_luts['tts'].unique()
    tts_angles, _ = find_closest_angles(actual_angle=tts, available_angles=sun_angles_lut)
    obs_angles_lut = available_luts['tto'].unique()
    tto_angles, _ = find_closest_angles(actual_angle=tto, available_angles=obs_angles_lut)
    psi_angles_lut = available_luts['psi'].unique()
    psi_angles, _ = find_closest_angles(actual_angle=psi, available_angles=psi_angles_lut)

    # find corresponding LUT files (we need 2**3, i.e., 8 LUTs)
    luts_to_interpolate = available_luts[
        (available_luts.tts.isin(tts_angles)) &
        (available_luts.tto.isin(tto_angles)) &
        (available_luts.psi.isin(psi_angles))
    ].copy()

    # read LUT data
    luts_to_interpolate['data'] = luts_to_interpolate.fpath.apply(
        lambda x, pd=pd: pd.read_pickle(x) 
    )
    luts_to_interpolate['data_shape'] = luts_to_interpolate.data.apply(
        lambda x: x.shape[0]
    )

    # loop over entries in the LUT and interpolate them one by one
    # NaNs records are skipped
    sel_columns += ['tts', 'tto', 'psi']
    interpolated_lut_entries = []
    luts_to_interpolate_scattered = client.scatter(luts_to_interpolate)
    for ii in range(luts_to_interpolate.data_shape.unique()[0]):
        # interpolated = interp_3d(luts_to_interpolate, ii, spectral_bands, traits, tts_angles, tto_angles, psi_angles, point)
        interpolated = client.submit(
            interp_3d,
            luts_to_interpolate_scattered,
            ii,
            spectral_bands,
            traits,
            tts_angles,
            tto_angles,
            psi_angles,
            point
        )
        interpolated_lut_entries.append(interpolated.result())

    lut = pd.DataFrame(interpolated_lut_entries)
    return lut            
    # from plot_lut_spectra import plot_spectra
    # plot_spectra(lut, out_dir=Path('.'), n_spectra=50) 

if __name__ == '__main__':

    cpu_count = multiprocessing.cpu_count() - 1 # leave one CPU for other operations
    client: Client = Client(n_workers=cpu_count)

    lut_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/PROSAIL_LUTs')
    sensor = 'Sentinel2B'
    pheno_phase = 'stemelongation-endofheading'

    tts = 22.8
    tto = 6.8
    psi = 56.2

    sel_columns = ['cab', 'lai', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    lut = load_lut(lut_dir, tts, tto, psi, sensor, pheno_phase, sel_columns=sel_columns, client=client)
    lut.to_csv(lut_dir.parent.joinpath('test.csv'))
