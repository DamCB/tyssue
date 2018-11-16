import numpy as np
import pandas as pd


def make_df(index, spec):
    """

    """
    dtypes = np.dtype([(name, type(val)) for name, val in spec.items()])
    N = len(index)
    arr = np.empty(N, dtype=dtypes)
    df = pd.DataFrame.from_records(arr, index=index)
    for name, val in spec.items():
        df[name] = val
    return df
