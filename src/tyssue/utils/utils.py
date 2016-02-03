import numpy as np
import logging

logger = logging.getLogger(name=__name__)

def _to_2d(df):
    # TODO test at_least_2d
    df_2d = np.asarray(df).repeat(2).reshape((df.size, 2))
    return df_2d


def _to_3d(df):

    df_3d = np.asarray(df).repeat(3).reshape((df.size, 3))
    return df_3d

def set_data_columns(datasets, specs, reset=False):

    if reset:
        logger.warn('Reseting datasets values to default specs')


    for name, data_dict in specs.items():
        if 'setting' in name:
            continue
        df = datasets[name]
        for col, default in data_dict.items():
            if not col in df.columns or reset:
                df[col] = default
