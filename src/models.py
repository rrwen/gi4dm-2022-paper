import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from tpot import TPOTClassifier, TPOTRegressor

_autosklearn_default_kwargs = dict(
    resampling_strategy = 'cv',
    resampling_strategy_arguments = {'folds': 10}
)

_tpot_default_kwargs = dict(
    warm_start = True
)

class AutoMLModel:
    
    def __init__(self, model=AutoSklearnClassifier, model_type=None, *args, **kwargs):
        
        # Convert str to model class
        model = getattr(sys.modules[__name__], model) if isinstance(model, str) else model
        
        # Auto determine model type
        if not model_type:
            if 'class' in model.__name__.lower():
                self.model_type = 'classifier'
            elif 'regress' in model.__name__.lower():
                self.model_type = 'regressor'
            else:
                self.model_type = 'unknown'
            
        # Set default kwargs for model
        if model.__name__.lower() in ['autosklearnclassifier', 'autosklearnregressor']:
            kwargs = _autosklearn_default_kwargs | kwargs
        elif model.__name__.lower() in ['tpotclassifier', 'tpotregressor']:
            kwargs = _tpot_default_kwargs | kwargs
            
        # Initiate model with params
        self.model = model(*args, **kwargs)
    
    def fit(self, x, y, *args, **kwargs):
        
        # Get input x and output y data for training
        x = x[[c for c in x.columns if c != y]] if isinstance(y, str) else x
        y = x[y] if isinstance(y, str) else y
        
        # Train model and set attributes
        self.last_x = x
        self.last_y = y
        self.model.fit(x, y, *args, **kwargs)
        
        # Refit if autosklearn cv
        if hasattr(self.model, 'resampling_strategy'):
            if self.model.resampling_strategy.lower() == 'cv':
                self.model.refit(x, y)
        
    def predict(self, *args, **kwargs):
        out = self.model.predict(*args, **kwargs)
        self.last_predicted = out
        return out
    
    def score(self, metric=None, y=None, predicted=None,  *args, **kwargs):
        
        # Get metrics from sklearn
        metric = 'f1_score' if self.model_type == 'classifier' else 'mean_squared_error'
        metric = getattr(sklearn.metrics, metric) if isinstance(metric, str) else metric
        
        # Calculate score with metric
        out = metric(y, predicted, *args, **kwargs)
        
        # Set attributes and return
        self.last_score = out
        self.last_metric = metric
        self.last_metric_name = metric.__name__
        return out
