import laspy
import numpy as np
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds
import xarray
from rasterio.enums import Resampling
from constants import EPSG, INFILE
import time

start_time = time.time()

infile = INFILE
epsg = EPSG

def create_tif(ground_points, outname, kind, crs="EPSG:5070"):
    resolution=1
    if kind == 'DEM':
        ground_points = points.points[(points.classification == 2) & (points.return_number == 1) ]
        print(len(ground_points))
    elif kind == 'DSM':
        ground_points = points.points[(points.classification != 18)& (points.return_number == 1)]
        print(len(ground_points))
    
    x_ground = ground_points.x
    y_ground = ground_points.y
    z_ground = ground_points.z
    
    min_x, max_x = np.min(x_ground), np.max(x_ground)
    min_y, max_y = np.min(y_ground), np.max(y_ground)
    
    x_grid = np.arange(min_x, max_x, resolution)
    y_grid = np.arange(min_y, max_y, resolution)

    # this data has reversed y coords?
    grid_x, grid_y = np.meshgrid(x_grid, y_grid[::-1])
    dem_data = griddata((x_ground, y_ground), z_ground, (grid_x, grid_y), method='nearest')

    dem_data[np.isnan(dem_data)] = -9999  # Assign a NoData value
    
    transform = from_bounds(min_x, min_y, max_x, max_y, dem_data.shape[1], dem_data.shape[0])
    
    with rasterio.open(
        outname,
        "w",
        driver="GTiff",
        height=dem_data.shape[0],
        width=dem_data.shape[1],
        count=1,
        dtype=dem_data.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(dem_data, 1)
        

with laspy.open(infile) as inFile:
    print(f'Reading {infile}')
    points = inFile.read()
    print(f'Processing {infile}')

    dem_outname = infile.split("\\")[-1].replace('.las','_dem.tif')
    create_tif(points, dem_outname, 'DEM', crs=epsg)
    print(f'{dem_outname} complete')

    dsm_outname = infile.split("\\")[-1].replace('.las','_dsm.tif')
    create_tif(points, dsm_outname , 'DSM', crs=epsg)
    print(f'{dsm_outname} complete')


dem_xds = xarray.open_dataarray(dem_outname)
dsm_xds = xarray.open_dataarray(dsm_outname)
chm_xds = dsm_xds - dem_xds
chm_outname = infile.split("\\")[-1].replace('.las','_chm.tif')
print(f'{chm_outname} complete')

print(f'CHM res {chm_xds.rio.resolution()}, DEM res {dem_xds.rio.resolution()}, DSM res {dsm_xds.rio.resolution()}')

#resample to 1ft for integer resolution needed for pycrown
print('Resampling to 1ft')
chm_xds = chm_xds.rio.reproject(
        chm_xds.rio.crs,
        resolution=(1, 1),  
        resampling=Resampling.nearest)

dem_xds = dem_xds.rio.reproject(
        dem_xds.rio.crs,
        resolution=(1, 1),  
        resampling=Resampling.nearest)

dsm_xds = dsm_xds.rio.reproject(
        dsm_xds.rio.crs,
        resolution=(1, 1),  
        resampling=Resampling.nearest)

dem_xds = dem_xds.rio.reproject_match(dem_xds)

dsm_xds = dsm_xds.rio.reproject_match(dsm_xds)

print(f'CHM res {chm_xds.rio.resolution()}, DEM res {dem_xds.rio.resolution()}, DSM res {dsm_xds.rio.resolution()}')

dem_xds.rio.to_raster(dem_outname.replace(".tif","_output.tif"))
dsm_xds.rio.to_raster(dsm_outname.replace(".tif","_output.tif"))
chm_xds.rio.to_raster(chm_outname.replace(".tif","_output.tif"))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"LiDAR pre-processing time: {elapsed_time:.4f} seconds")