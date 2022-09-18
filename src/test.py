import preprocess, data, models, optimization
import geopandas as gpd
import pandas as pd

x = gpd.read_file('../downloads/geogrid_to_10.geojson')
x = pd.DataFrame(x.drop(columns='geometry'))

startswith = [
    'collisions_pd',
    'collisions_fatalities',
    'collisions_ftr',
    'collisions_injury',
    'collisions_day',
    'collisions_month',
    'collisions_year',
    'collisions_hour',
    'collisions_day',
    'collisions_month'
]
drop = [c for c in x.columns if any(c.startswith(s) for s in startswith) and c != 'collisions_count']
x = x.drop(columns=drop)

ask = models.AutoMLModel('AutoSklearnRegressor')
ask.fit(x=x, y='collisions_count')

tpt = models.AutoMLModel('TPOTRegressor')
tpt.fit(x=x, y='collisions_count')

lconstr = {''}

aopt = optimization.Optimizer(ask, lconstr, 'BayesianOptimization')
#topt = optimization.Optimizer(tpt, 'BayesianOptimization')

aopt.optimize(n_iter=2)
#topt.optimize(n_iter=2)