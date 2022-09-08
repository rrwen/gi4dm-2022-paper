import standardization
import data
import geopandas as gpd

x = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
grid = data.gen_grid(x, 10)
x['numbas'] = 1
bins = standardization.geobin({'cities': x}, grid)