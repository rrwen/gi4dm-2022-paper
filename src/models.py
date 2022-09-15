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

class AutoMLModel:
    
    def __init__(self, model=TPOTClassifier, *args, **kwargs):
        
        # Convert str to model class
        model = getattr(sys.modules[__name__], model) if isinstance(model, str) else model
        
        # Get class name of model
        name = model.__name__ if hasattr(model, '__name__') else type(model).__name__
        
        # Auto determine model type
        if 'class' in name.lower():
            mtype = 'classifier'
        elif 'regress' in name.lower():
            mtype = 'regressor'
        else:
            mtype = 'unknown'
            
        # Set default kwargs for model
        if name.lower() in ['autosklearnclassifier', 'autosklearnregressor']:
            kwargs = _autosklearn_default_kwargs | kwargs
            group = 'autosklearn'
        elif name.lower() in ['tpotclassifier', 'tpotregressor']:
            kwargs = _tpot_default_kwargs | kwargs
            group = 'tpot'
        else:
            group = 'unknown'
            
        # Set attrs
        self.model = model(*args, **kwargs)
        self.model_name = name
        self.model_group = group
        self.model_call = call
        self.model_type = mtype
    
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
        if self.model_group == 'autosklearn':
            self.model_details = pd.DataFrame(self.model.cv_results_)
        if self.model_group == 'tpot':
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
