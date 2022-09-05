import geopandas as gpd

def read_geodata(file_path, *args, **kwargs):
    out = gpd.read_file(filename=file_path, *args, **kwargs)
    return out

def write_geodata(gdf, file_path, *args, **kwargs):
    gdf.to_file(filename=file_path, *args, **kwargs)