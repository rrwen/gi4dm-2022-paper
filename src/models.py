import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from tpot import TPOTClassifier, TPOTRegressor

_autosklearn_default_kwargs = dict(
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 10}
)

class AutoMLModel:
    
    def __init__(self, base_model=AutoSklearnClassifier, *args, **kwargs):
        
        # Convert str to model class
        model = getattr(sys.modules[__name__], model) if isinstance(model, str) else model
        
        # Determine model type
        self.model_type = 'classifier' if model == AutoSklearnClassifier or model == TPOTClassifier else 'regressor'
        
        # Init model with params
        if model == AutoSklearnClassifier or model == AutoSklearnRegressor:
            kwargs = _autosklearn_default_kwargs | kwargs
        self.model = model(*args, **kwargs)
    
    def fit(self, x, y, *args, **kwargs):
        
        x = x[[c for c in x.columns if c != y]] if isinstance(y, str) else x
        y = x[y] if isinstance(y, str) else y
        
        self.last_y = y
        self.model.fit(*args, **kwargs)
        
    def predict(self, *args, **kwargs):
        out = self.model.predict(*args, **kwargs)
        self.last_predicted = out
        return out
    
    def score(self, metric=None, y=None, predicted=None,  *args, **kwargs):
        metric = 'f1_score' if self.model_type == 'classifier' else 'mean_squared_error'
        metric = getattr(sklearn.metrics, metric) if isinstance(metric, str) else metric
        out = metric(y, predicted, *args, **kwargs)
        self.last_score = out
        self.last_metric = metric.__name__
        return out
