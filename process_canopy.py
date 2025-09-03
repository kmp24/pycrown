import sys
from datetime import datetime
from osgeo import gdal
from pycrown import PyCrown
import time
import sys
from constants import INFILE

start_time = time.time()

infile = INFILE

F_CHM = infile.split("\\")[-1].replace('.las','_chm_output.tif') #dsm-dem
F_DTM = infile.split("\\")[-1].replace('.las','_dem_output.tif') #(DEM)
F_DSM = infile.split("\\")[-1].replace('.las','_dsm_output.tif')
F_LAS = infile

PC = PyCrown(F_CHM, F_DTM, F_DSM, F_LAS, outpath=infile.split("\\")[-1].replace('.las',''))

PC.filter_chm(7, ws_in_pixels=True, circular=False)
PC.chm = PC.chm.astype('float32')
PC.dtm = PC.dtm.astype('float32')
PC.dsm = PC.dsm.astype('float32')

PC.tree_detection(PC.chm, ws=5, hmin=3, ws_in_pixels=True)
PC.crown_delineation(algorithm='dalponteCIRC_numba', th_tree=6, # seed starting point for crown delin
                     th_seed=0.65, th_crown=0.5, max_crown=25.)

PC.correct_tree_tops()

PC.get_tree_height_elevation(loc='top')
PC.get_tree_height_elevation(loc='top_cor')

PC.screen_small_trees(hmin=2., loc='top')

from shapely.geometry import mapping, Point, Polygon
from rasterio.features import shapes as rioshapes
polys = []
for feature in rioshapes(PC.crowns, mask=PC.crowns.astype(bool)):

    # Convert pixel coordinates to lon/lat
    edges = feature[0]['coordinates'][0].copy()
    for i in range(len(edges)):
        edges[i] = PC._to_lonlat(*edges[i], PC.resolution)

    # poly_smooth = PC.smooth_poly(Polygon(edges), s=None, k=9)
    polys.append(Polygon(edges))
PC.trees.crown_poly_raster = polys

import geopandas as gpd
import numpy as np
thin_perc = None
if thin_perc:
    thin_size = np.floor(len(PC.las) * (1 - thin_perc))
    lidar_geodf = PC.las.sample(n=thin_size)
else:
    lidar_geodf = PC.las

print('Converting LAS point cloud to shapely points')
geometry = [Point(xy) for xy in zip(lidar_geodf.x, lidar_geodf.y)]
lidar_geodf = gpd.GeoDataFrame(lidar_geodf, crs=f'epsg:{PC.epsg}',
                                geometry=geometry)

print('Converting raster crowns to shapely polygons')

import numpy as np
import pandas as pd

print('Converting raster crowns to shapely polygons')
polys = []
for feature in rioshapes(PC.crowns, mask=PC.crowns.astype(bool)):
    edges = np.array(list(zip(*feature[0]['coordinates'][0])))
    edges = np.array(PC._to_lonlat(edges[0], edges[1],
                                        PC.resolution)).T
    polys.append(Polygon(edges))
crown_geodf = gpd.GeoDataFrame(
    pd.DataFrame(np.arange(len(PC.trees))),
    crs=f'epsg:{PC.epsg}', geometry=polys
)

print('Attach LiDAR points to corresponding crowns')
lidar_in_crowns = gpd.sjoin(lidar_geodf, crown_geodf,
                            predicate='within', how="inner")

lidar_tree_class = np.zeros(lidar_in_crowns['index_right'].size)
lidar_tree_mask = np.zeros(lidar_in_crowns['index_right'].size,
                            dtype=bool)

from skimage.filters import threshold_otsu
print('Create convex hull around first return points')
polys = []
first_return = None
indx_list = sorted(list(lidar_in_crowns['index_right'].unique()))

for tidx in range(len(PC.trees)):
    bool_indices = lidar_in_crowns['index_right'] == tidx
    lidar_tree_class[bool_indices] = tidx
    points = lidar_in_crowns[bool_indices]
    # check that not all values are the same
    if len(points.z) > 1 and not np.allclose(points.z,
                                    points.iloc[0].z):
        threshold = threshold_otsu(points.z.values)
        points = points[points.z >= threshold]
        if first_return:
            points = points[points.return_num == 1]
    if tidx in indx_list:        
        hull = points.unary_union.convex_hull
        polys.append(hull)
        lidar_tree_mask[bool_indices] = \
            lidar_in_crowns[bool_indices].within(hull)
    else:
        polys.append(None)
PC.trees.crown_poly_smooth = polys

PC.quality_control()
print(f"Number of trees detected: {len(PC.trees)}")

PC.export_raster(PC.chm, PC.outpath / 'canopy_dalponte.tif', 'CHM')
PC.export_tree_locations(loc='top')
PC.export_tree_locations(loc='top_cor') # corrected tree tops?
PC.export_tree_crowns(crowntype='crown_poly_raster')

# faced an unexpected issue with non-polygon geometry types
# update the dataframe as a workaround, for my sample 9 were points or linestrings out of 1057
import geopandas as gpd
PC.trees = PC.trees.loc[PC.trees.crown_poly_smooth.astype(str).str.contains('POLYGON')]
PC.export_tree_crowns(crowntype='crown_poly_smooth')

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Canopy processing time: {elapsed_time:.4f} seconds")