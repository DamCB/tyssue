import numpy as np

def _to_2d(df):

    df_2d = np.asarray(df).repeat(2).reshape((df.size, 2))
    return df_2d


def _to_3d(df):

    df_3d = np.asarray(df).repeat(3).reshape((df.size, 3))
    return df_3d

def set_data_columns(eptm, data_specs, reset=False):

    for name, data_dict in data_specs.items():
        if 'setting' in name:
            continue
        df = getattr(eptm, '{}_df'.format(name))
        for col, (default, dtype) in data_dict.items():
            if not col in df.columns or reset:
                df[col] = default
