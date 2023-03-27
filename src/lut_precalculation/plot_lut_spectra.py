'''
Created on Mar 3, 2023

@author: graflu
'''

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from pathlib import Path

mpl.rc('font', size=16)
plt.style.use('bmh')
bands = ['B02','B03','B04','B05','B06','B07','B8A','B11','B12']
wvls = [490, 560, 665, 705, 740, 783, 842, 1610, 2190]

def plot_spectra(df: pd.DataFrame, out_dir: Path, n_spectra: int = 2000):

    f, ax = plt.subplots(figsize=(16,10))
    _df = df.sample(n=n_spectra)
    for _, spectrum in _df.iterrows():
        ax.plot(wvls, spectrum[bands])
    ax.set_ylabel('Bi-Directional Reflectance (PROSAIL) [-]')
    ax.set_xlabel('Spectral Wavelength [nm]')
    ax.set_title(f'Sentinel-2 Spectra; N={_df.shape[0]} (out of {df.shape[0]})')

    f.savefig(out_dir.joinpath('S2_spectra.png'))
    plt.close(f)

if __name__ == '__main__':

    fpath_lut = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/PROSAIL_LUTs/Sentinel2A/stemelongation-endofheading/25_6_45.pkl')
    out_dir = fpath_lut.parent
    lut = pd.read_pickle(fpath_lut)

    plot_spectra(df=lut, out_dir=out_dir)
