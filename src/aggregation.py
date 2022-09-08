import pandas as pd

# Calculates the mode for a series
def mode(x):
    out = x.map(str).value_counts().index[0]
    return out

# 25 per quantile
def quantile25(df):
    out = df.quantile(0.25)
    return out

# 50 per quantile
def quantile50(df):
    out = df.quantile(0.5)
    return out

# 75 per quantile
def quantile75(df):
    out = df.quantile(0.75)
    return out

# Calculates the counts of unique values given all possible values
def ucount(df, c, possible, round=False):
    
    # Get counts in df
    counts = df[c].value_counts().to_dict()
    
    # Round numeric values
    if round:
        possible = [int(k) if isinstance(k, Number) else k for k in possible]
        counts = {int(k) if isinstance(k, Number) else k: v for k,v in counts.items()}
    
    # Apply counts as dict with possible values
    possible = [str(k) for k in possible]
    out = {k:[0] for k in possible}
    for k, v in counts.items():
        out[str(k)] = [v]
    out = pd.DataFrame(out)
    return out