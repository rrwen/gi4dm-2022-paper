import data
import pandas as pd

from bayes_opt import BayesianOptimization

# Create optimization parameters from dataframe
def _create_params(df, gconstr={}, lconstr={}):
    
    # Apply global constraints as query
    df = df.query(gconstr['query']) if gconstr else df
    
    # Construct param df structure
    out = {
        'column': [],
        'row': [],
        'param': [],
        'value': []
    }
    
    # Create params from constraints and df
    for c in df.columns:
        
        # Apply local constraints if avail
        if c in lconstr:
            constr = lconstr[c]
            values = df.query(constr['query'])[c]
        else:
            constr = None
            values = df[c]
        
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
    return out

# Create optimization function
def _create_func(model, params):
    pass

class Optimizer:
    
    def __init__(self, model, data=None, gconstr=None, lconstr=None, optimizer=BayesianOptimization, *args, **kwargs):
        
        # Get class name of opt
        name = model.__name__ if hasattr(model, '__name__') else type(model).__name__
        
        # Add opt group and call
        if name.lower() in 'bayesianoptimization':
            group = 'bayesian'
            call = 'maximize'
        else:
            group = 'unknown'
            call = 'unknown'
            
        # Auto determine opt type
        call = self.optimizer.ca
        if 'max' in call.lower():
            otype = 'maximizer'
        elif 'min' in call.lower():
            otype = 'minimizer'
        else:
            otype = 'unknown'
            
        # Auto determine data struct
        data = model.last_x if hasattr(model, 'last_x') and not data else data
        
        # Create params and func
        params = _create_params(df=data, gconstr=gconstr, lconstr=lconstr)
        func = _create_func(model=model, params=params)
        
        # Set opt attrs
        self.optimizer = optimizer(*args, **kwargs)
        self.optimizer_name = name
        self.optimizer_group = group
        self.optimizer_call = call
        self.optimizer_type = otype
        
        # Set model attrs
        self.model = model
        self.model_data = data
        self.model_gconstr = gconstr
        self.model_lconstr = lconstr
        self.model_params = params
        
    def optimize(self, *args, **kwargs):
        # TODO
        # df.loc[idx, 'col'] = ['val1', 'val2']
        getattr(self.optimizer, self.optimizer_call)(*args, **kwargs)
        