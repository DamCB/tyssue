import numpy as np
import logging

logger = logging.getLogger(name=__name__)

def _to_2d(df):
    df_2d = np.asarray(df).repeat(2).reshape((df.size, 2))
    return df_2d


def _to_3d(df):

    df_3d = np.asarray(df).repeat(3).reshape((df.size, 3))
    return df_3d


def spec_updater(specs, new):

    for key, spec in specs.items():
        if new.get(key) is not None:
            spec.update(new[key])



def set_data_columns(datasets, specs, reset=False):

    if reset:
        logger.warn('Reseting datasets values with new specs')

    for name, spec in specs.items():
        if 'setting' in name:
            continue
        df = datasets[name]
        for col, default in spec.items():
            if not col in df.columns or reset:
                df[col] = default
