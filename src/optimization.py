import data
import pandas as pd

from bayes_opt import BayesianOptimization

# Create optimization parameters from dataframe
def _create_params(x, gconstr={}, lconstr={}):
    
    # Apply global constraints as query
    x = x.query(gconstr['query']) if gconstr else x
    
    # Construct param df structure
    out = {
        'column': [],
        'row': [],
        'param': [],
        'value': []
    }
    
    # Create params from constraints and df
    for c in x.columns:
        
        # Apply local constraints if avail
        if c in lconstr:
            constr = lconstr[c]
            values = x.query(constr['query'])[c]
        else:
            constr = None
            values = x[c]
        
        # Get rows and convert values to list
        rows = values.index.tolist()
        values = values.tolist()
        
        # Append parameters and their indices for opt
        out['column'] += [c] * len(values)
        out['row'] += rows
        out['param'] += [f'{c}_{i}' for i, v in zip(rows, values)]
        out['value'] += values
        
    # Create params df and return
    out = pd.DataFrame(out)
    out.index = out.param
    return out

# Create optimization function
def _create_func(model, x, y, params, metric, mult=1):
    def f(**kwargs):
        
        # Use param values for simulated x
        params.loc[kwargs.keys(), 'value'] = list(kwargs.values())
        xs = x.copy()
        
        # Assign simulated values to real x values
        for c in params.column.unique():
            p = params[params.column == c]
            xs.loc[p.row, c] = p.value
        
        # Run prediction on simulated x input
        model.predict(xs)
        out = model.score(y=y) * mult
        return out
    
    return f

class Optimizer:
    
    def __init__(self, model, optimizer=BayesianOptimization, gconstr=None, lconstr=None, x=None, y=None, infer_bounds=True *args, **kwargs):
        
        # Get class name of opt
        name = optimizer.__name__ if hasattr(optimizer, '__name__') else type(optimizer).__name__
        
        # Auto determine x and y model input and output
        x = model.last_x if hasattr(model, 'last_x') and not x else x
        y = model.last_y if hasattr(model, 'last_y') and not y else y
        
        # Create opt params
        params = _create_params(x=x, gconstr=gconstr, lconstr=lconstr)
        params['value_orig'] = params['value']
        
        # Calculate opt bounds
        for k, v in lconstr.items():
            if 'bounds' not in v and infer_bounds:
                lconstr[k]['bounds'] = (x[k].min, x[k].max)
        bounds = {k: v['bounds'] for k, v in lconstr.items()}
        
        # Add opt group and call
        if name.lower() in 'bayesianoptimization':
            group = 'bayesian'
            call = 'maximize'
            kwargs['f'] = func
            kwargs['pbounds'] = bounds
        else:
            group = 'unknown'
            call = 'unknown'
            
        # Auto determine opt type and multiplier
        if 'max' in call.lower():
            otype = 'maximizer'
            mult = 1 if model.model_metric_positive else -1
        elif 'min' in call.lower():
            otype = 'minimizer'
            mult = -1 if model.model_metric_positive else 1
        else:
            otype = 'unknown'
        
        # Create opt func
        func = _create_func(model=model, x=x, y=y, params=params, mult=mult)
        
        # Set opt attrs
        self.optimizer = optimizer(*args, **kwargs)
        self.optimizer_name = name
        self.optimizer_group = group
        self.optimizer_call = call
        self.optimizer_type = otype
        
        # Set model attrs
        self.model = model
        self.model_x = x
        self.model_y = y
        self.model_gconstr = gconstr
        self.model_lconstr = lconstr
        self.model_params = params
        self.model_params_bounds = bounds
        
    def optimize(self, *args, **kwargs):
        # TODO
        # df.loc[idx, 'col'] = ['val1', 'val2']
        getattr(self.optimizer, self.optimizer_call)(*args, **kwargs)
        