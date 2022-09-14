import data

from bayes_opt import BayesianOptimization


def create_params(df, gconstr=None, lconstr=None, format_colnames=data.format_colnames):
    
    # TODO: build dataframe for params
    
    # Apply global constraints as query
    df = df.query(gconstr['query']) if gconstr else df
    df.columns = format_colnames(df.columns)
    
    # 
    out = []
    i = 0
    for c in df.columns:
        
        if c in lconstr:
            constr = lconstr[c]
            values = df.query(constr['query'])[c]
        else:
            constr = None
            values = df[c]
        index = values.index.tolist()
        
        # Track start and end of flattened df
        start = i
        i += len(values)
        end = i
        
        # Append 
        out.append({
            'name': c,
            'start': start,
            'end': end,
            'values': values,
            'index': index,
            'constraints': constr
        })
    return out

class Optimizer:
    
    def __init__(self, optimizer=BayesianOptimization, *args, **kwargs):
        # TODO
        # df.loc[idx, 'col'] = ['val1', 'val2']
        pass