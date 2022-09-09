import preprocess
import data
import models
import geopandas as gpd

x = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
x['numbas'] = 1

grid = data.gen_grid(x, 10)
bins = preprocess.geobin({'cities': x}, grid)

aml = models.AutoML()
aml.train()