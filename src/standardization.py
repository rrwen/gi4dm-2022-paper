import geopandas as gpd

from aggregation import *

from pandas.api.types import is_numeric_dtype, is_string_dtype

def geobin(
    geodict,
    bins,
    stats=['sum', 'mean', 'min', 'max', 'median', 'var', 'skew', 'std', 'sem', 'mad', mode, quantile25, quantile50, quantile75],
    ucount_threshold=100,
    ignore_cols=['geometry'],
    join_kwargs={'predicate': 'intersects'},
    *args, **kwargs):
    
    # Convert to general dict if single gdf
    geodict = {'data': geodict} if isinstance(geodict, gpd.GeoDataFrame) else geodict

    # Call func if bins is not a gdf
    if not isinstance(bins, gpd.GeoDataFrame):
        bins = bins(*args, **kwargs)
        
    # Aggregate data by bins
    for name, gdf in geodict.items():
        
        # Spatially join to bins
        join = bins.sjoin(gdf, **join_kwargs)
        group = join.groupby(join.index)
        
        # Aggregate count
        counts = join.groupby(join.index).size().fillna(0)
        counts.name = f'{name}_count'
        bins = bins.join(counts)
        
        # Aggregate by stats if numeric
        num_columns = [c for c in gdf.columns if is_numeric_dtype(gdf[c]) and c not in ignore_cols]
        if len(num_columns) > 0:
            agg = group.agg({c: stats for c in num_columns})
            agg.columns = [f'{name}_{"_".join(c).strip()}' for c in agg.columns]
            bins = bins.join(agg)
        
        # Aggregate unique count if str and unique values under threshold
        ufreq = []
        str_columns = [c for c in gdf.columns if is_string_dtype(gdf[c]) and c not in ignore_cols]
        str_columns = [c for c in str_columns if gdf[c].unique().size <= ucount_threshold]
        if len(str_columns) > 0:
            for c in str_columns:

                # Get possible unique values in col
                possible = gdf[c].unique()

                # Count freq for each unique value
                f = group.apply(ucount, c=c, possible=possible)

                # Rename freq cols
                prefix = c.replace(f'{name}_', '')
                f.columns = [f'{name}_{c}_count' for c in f.columns]
                ufreq.append(f)

            # Combine count freq
            ufreq = pd.concat(ufreq, axis=1)
            ufreq.index = ufreq.index.get_level_values(0)
            bins = bins.join(ufreq)
        
    # Return binned aggregate data
    out = bins
    return out