'''
Generate raw LAI time series from PROSAIL inversion of S2 imagery.

@author: Lukas Valentin Graf
'''

from pathlib import Path


if __name__ == '__main__':

    test_sites_dir = Path('../data/Test_Sites')
    s2_trait_dir = Path('/home/graflu/public/Evaluation/Projects/KP0031_lgraf_PhenomEn/04_LaaL/S2_Traits')
