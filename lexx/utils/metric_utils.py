def ranking_array(vals, reverse=False):
    import numpy as np

    """Returns an array of rankings from an array of values"""
    if reverse:
        vals = -vals
    return np.argsort(np.argsort(vals))
