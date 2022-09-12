from bayes_opt import BayesianOptimization

def create_params(df, constraints=None):
    # TODO
    # x1 = df.query('col == "value"').col
    # idx = x1.index.tolist()
    # df.loc[idx, 'col'] = ['val1', 'val2']
    # out = dict(col1 = {start: 0, end: 5, values=[1,2,3,4,5], index=[], constraints={}}, col2 = {})
    pass

class Optimizer:
    
    def __init__(self, optimizer=BayesianOptimization, *args, **kwargs):
        pass