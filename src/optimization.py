import data

from bayes_opt import BayesianOptimization


def create_params(df, gconstr={}, lconstr={}, format_colnames=data.format_colnames):
    
    # Apply global constraints as query
    df = df.query(gconstr['query']) if gconstr else df
    df.columns = format_colnames(df.columns)
    
    # Construct param df structure
    out = {
        'column': [],
        'param': [],
        'index': [],
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
        
        # Get idx and convert values to list
        index = values.index.tolist()
        values = values.tolist()
        
        # Append parameters and their indices for opt
        out['column'] += [c] * len(values)
        out['param'] += [f'{c}_{i}' for i, v in zip(index, values)]
        out['index'] += index
        out['value'] += values
        
    # Create params df and return
    out = pd.DataFrame(out)
    return out

class Optimizer:
    
    def __init__(self, optimizer=BayesianOptimization, *args, **kwargs):
        # TODO
        # df.loc[idx, 'col'] = ['val1', 'val2']
        pass