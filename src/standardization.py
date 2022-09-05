import geopandas as gpd
import numpy as np

from data import read_geodata

from shapely.geometry import Polygon

# Generate rectangular grids using a bounding box
# Edited from user Mativane's code
# https://gis.stackexchange.com/questions/269243/creating-polygon-grid-using-geopandas
def gen_grid(bounds, cell_length, cell_width):
    
    # Get bounds or extract if gdf
    xmin, ymin, xmax, ymax = bounds.total_bounds if isinstance(bounds, gpd.GeoDataFrame) else bounds

    # Calculate rows and cols
    cols = list(np.arange(xmin, xmax + cell_width, cell_width))
    rows = list(np.arange(ymin, ymax + cell_length, cell_length))

    # Generate grid based on bounds, cols, and rows
    polygons = []
    for x in cols[:-1]:
        for y in rows[:-1]:
            polygons.append(Polygon([(x,y), (x + cell_width, y), (x + cell_width, y+cell_length), (x, y+cell_length)]))

    # Return grid as gdf
    crs = bounds.crs if isinstance(bounds, gpd.GeoDataFrame) else None
    out = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
    out.index = list(out.index)
    return out

def geobin(geodata, bins, *args, **kwargs):

    # Read geodata if str and convert to list of gdf
    gdfl = geodata if isinstance(geodata, list) else [geodata]
    gdfl = [read_geodata(gdf) if isinstance(gdf, str) else gdf for gdf in gdfl]

    # Call func if bins is not a gdf
    if not isinstance(bins, gpd.GeoDataFrame):
        bins = bins(*args, **kwargs)

    # 
    return out