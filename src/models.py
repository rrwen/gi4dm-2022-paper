import pandas as pd
import sklearn.metrics
import sys

from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from tpot import TPOTClassifier, TPOTRegressor

_autosklearn_default_kwargs = dict(
    resampling_strategy = 'cv',
    resampling_strategy_arguments = {'folds': 10},
    memory_limit = None,
    time_left_for_this_task = 30
)

_tpot_default_kwargs = dict(
    warm_start = True,
    max_time_mins = 0.5
)

def _is_autosklearn(model):
    out = model.__name__.lower() in ['autosklearnclassifier', 'autosklearnregressor']
    return out

def _is_tpot(model):
    out = model.__name__.lower() in ['tpotclassifier', 'tpotregressor']
    return out

class AutoMLModel:
    
    def __init__(self, model=TPOTClassifier, *args, **kwargs):
        
        # Convert str to model class
        model = getattr(sys.modules[__name__], model) if isinstance(model, str) else model
        
        # Auto determine model type
        if 'class' in model.__name__.lower():
            self.model_type = 'classifier'
        elif 'regress' in model.__name__.lower():
            self.model_type = 'regressor'
        else:
            self.model_type = 'unknown'
            
        # Set default kwargs for model
        if _is_autosklearn(model):
            kwargs = _autosklearn_default_kwargs | kwargs
        elif _is_tpot(model):
            kwargs = _tpot_default_kwargs | kwargs
            
        # Initiate model with params
        self.model = model(*args, **kwargs)
    
    def fit(self, x, y, *args, **kwargs):
        
        # Get input x and output y data for training
        y = x[y] if isinstance(y, str) else y
        x = x[[c for c in x.columns if c != y.name]]
        
        # Train model
        self.model.fit(x, y, *args, **kwargs)
        
        # Refit if autosklearn cv
        if hasattr(self.model, 'resampling_strategy'):
            if self.model.resampling_strategy.lower() == 'cv':
                self.model.refit(x, y)
        
        # Get model details from fitting
        if _is_autosklearn(self.model):
            self.model_details = pd.DataFrame(self.model.cv_results_)
        if _is_tpot(self.model):
            self.model_details = pd.DataFrame.from_dict(self.model.evaluated_individuals_, orient='index').reset_index(drop=True)
        
        # Set last fitted input x and output y
        self.last_x = x
        self.last_y = y
        
    def predict(self, x=None, *args, **kwargs):
        x = x if x else self.last_x
        out = self.model.predict(x, *args, **kwargs)
        self.last_predicted = out
        return out
    
    def score(self, metric=None, y=None, predicted=None,  *args, **kwargs):
        
        # Get metrics from sklearn
        metric = 'f1_score' if self.model_type == 'classifier' else 'r2_score'
        metric = getattr(sklearn.metrics, metric) if isinstance(metric, str) else metric
        
        # Calculate score with metric
        y = y if y else self.last_y
        predicted = predicted if predicted else self.last_predicted
        out = metric(y, predicted, *args, **kwargs)
        
        # Set attributes and return
        self.last_score = out
        self.last_metric = metric
        self.last_metric_name = metric.__name__
        return out
