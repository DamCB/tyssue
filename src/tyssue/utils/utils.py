import numpy as np
import logging
import pandas as pd

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


def data_at_opposite(sheet, edge_data, free_value=None):
    """
    Returns a pd.DataFrame with the values of the input edge_data
    at the opposite edges. For free edges, optionaly replaces Nan values
    with free_value
    """
    if isinstance(edge_data, pd.Series):
        opposite = pd.Series(
            edge_data.loc[sheet.edge_df['opposite']].values,
            index=edge_data.index)
    else:
        opposite = pd.DataFrame(
            edge_data.loc[sheet.edge_df['opposite']].values,
            index=edge_data.index)
    if free_value is not None:
        opposite = opposite.replace(np.nan, free_value)

    return opposite
