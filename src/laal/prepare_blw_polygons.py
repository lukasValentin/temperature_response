'''
Created on Apr 4, 2023

@author: graflu
'''

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from pathlib import Path

target_epsg = 2056 # Swiss coordinate system
min_parcel_size = 0.5  # hectar

plt.style.use('bmh')

def prepare_blw_polygons(
    fpath_s2_tiles: Path,
    fpath_ww_polys: Path
) -> None:
    """
    Prepare winter wheat polygon data for trait retrieval.

    :param fpath_s2_tiles:
        path to directory with selected Sentinel-2 tiles
    :param fpath_ww_polys:
        path to directory with WW polygon data from Swiss Federal
        Office of Agriculture 8BLW)
    """
    s2_tiles = gpd.read_file(next(fpath_s2_tiles.glob('*.gpkg')))
    s2_tiles.to_crs(epsg=target_epsg, inplace=True)

    # loop over WW polygon files and clip these to the extent of
    # the S2 tiles. This means some parcels will appear in more than
    # one tile. We'll solve this when we have the final LAI TS
    for ww_polys in fpath_ww_polys.glob('*.gpkg'):
        polys = gpd.read_file(ww_polys)
        year = ww_polys.name.split('_')[1].split('.')[0]
        out_dir_year = fpath_ww_polys.joinpath(year)
        out_dir_year.mkdir(exist_ok=True)
        # loop over S2 tiles
        stats_tile_list = []
        for _, s2_tile_geom in s2_tiles.iterrows():
            s2_tile = s2_tile_geom['Name']
            # clip to tile
            polys_clipped = polys.clip(s2_tile_geom.geometry)
            # discard parcels smaller than min_parcel_size in ha
            polys_clipped['area_ha'] = polys_clipped.area
            polys_clipped.area_ha = polys_clipped.area_ha * (1/100**2)
            polys_clipped_eq = polys_clipped[polys_clipped.area_ha >= min_parcel_size].copy()
            stats_tiles_parcel = {
                'total': polys_clipped.shape[0],
                'total_mean_area_ha': polys_clipped.area_ha.mean(),
                'total_median_area_ha': polys_clipped.area_ha.median(),
                'after_size_constraint': polys_clipped_eq.shape[0],
                'after_size_constraint_mean_area_ha': polys_clipped_eq.area_ha.mean(),
                'after_size_constraint_median_area_ha': polys_clipped_eq.area_ha.median(),
                's2_tile': s2_tile
            }
            stats_tile_list.append(stats_tiles_parcel)
            # save clipped parcels to directory
            polys_clipped_eq.to_file(out_dir_year.joinpath(f'{s2_tile}_WW.gpkg'))
            # plot histogram
            f, ax = plt.subplots(figsize=(6,6))
            polys_clipped_eq.area_ha.hist(ax=ax, bins=50)
            ax.set_xlabel('Field parcel area [ha]')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{s2_tile}: Fields >= {min_parcel_size} ha (N={polys_clipped_eq.shape[0]})')
            f.savefig(out_dir_year.joinpath(f'{s2_tile}_WW.png'))
            plt.close(f)

        # save parcel statistics
        parcel_stats = pd.DataFrame(stats_tile_list)
        parcel_stats.to_csv(out_dir_year.joinpath(f'WW_parcel_stats.csv'), index=False)



if __name__ == '__main__':

    gis_dir = Path('/mnt/ides/Lukas/00_GIS_Basedata')
    fpath_s2_tiles = gis_dir.joinpath('S2_CH')
    fpath_ww_polys = gis_dir.joinpath('WW_CH')
