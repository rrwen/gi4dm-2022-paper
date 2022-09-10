from src import preprocess, data, models
import geopandas as gpd
import pandas as pd

x = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
x['numbas'] = 1

grid = data.gen_grid(x, 10)
bins = preprocess.geobin({'cities': x}, grid)

x = pd.DataFrame(bins).drop('geometry', axis=1)
x = x.fillna(0)

aml = models.AutoMLModel('AutoSklearnRegressor')
aml.fit(x, 'cities_count')