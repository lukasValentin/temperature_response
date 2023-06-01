"""
The yield data can be obtained from
https://www.research-collection.ethz.ch/handle/20.500.11850/581023.
The data should be unzipped and copied into the `data/yield` folder.
"""

from pathlib import Path


if __name__ == '__main__':

    fpath_yield = Path(
        'data/yield/Data_Code_Yieldmapping_MS/cloudy_data/WW_yield_tot.csv')
