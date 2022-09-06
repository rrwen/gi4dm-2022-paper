import geopandas as gpd

from data import read_geodata

def geobin(
    geodata,
    bins,
    num_stats=['sum', 'mean', 'min', 'max', 'median', 'var', 'skew', 'std', 'sem', 'mad'],
    str_stats=[],
    ignore_cols=['geometry'],
    join_kwargs={'predicate': 'intersects'},
    *args, **kwargs):

    # Read geodata if str and convert to list of gdf
    gdfl = geodata if isinstance(geodata, list) else [geodata]
    gdfl = [read_geodata(gdf) if isinstance(gdf, str) else gdf for gdf in gdfl]

    # Call func if bins is not a gdf
    if not isinstance(bins, gpd.GeoDataFrame):
        bins = bins(*args, **kwargs)
        
    # Aggregate data by bins
    for gdf in gdfl:
        
        # Spatially join to bins
        join = bins.sjoin(gdf, **join_kwargs)
        group = join.groupby(join.index)
        
        # Aggregate count
        counts = join.groupby(join.index).size().fillna(0)
        counts.name = f'{name}_count'
        bins = bins.join(counts)
        
        # Aggregate by stats
        # TODO: Need to aggregate depending on data type (str or numeric)
        agg = group.agg({c: num_stats for c in gdf.columns if c not in ignore_cols})
        agg.columns = ['_'.join(c).strip() for c in agg.columns]
        bins = bins.join(agg)

    # Return binned aggregate data
    out = bins
    return out