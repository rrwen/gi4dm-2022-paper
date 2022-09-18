import preprocess, data, models, optimization
import geopandas as gpd
import pandas as pd

x = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
x['numbas'] = 1

grid = data.gen_grid(x, 10)
bins = preprocess.geobin({'cities': x}, grid)

x = pd.DataFrame(bins).drop('geometry', axis=1)
x = x.fillna(0)

ask = models.AutoMLModel('AutoSklearnRegressor')
ask.fit(x=x, y=x.cities_count)

tpt = models.AutoMLModel('TPOTRegressor')
tpt.fit(x=x, y=x.cities_count)

aopt = optimization.Optimizer(ask, 'BayesianOptimization')
#topt = optimization.Optimizer(tpt, 'BayesianOptimization')

aopt.optimize(n_iter=2)
#topt.optimize(n_iter=2)