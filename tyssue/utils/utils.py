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


def combine_specs(*specs):

    combined = {}
    for spec in specs:
        for key in spec:
            if key in combined:
                combined[key].update(spec[key])
            else:
                combined[key] = spec[key]
    return combined


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
            if col not in df.columns or reset:
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


def get_sub_eptm(eptm, edges):
    from ..objects.core import Epithelium

    edge_df = eptm.edge_df.loc[edges]
    vert_df = eptm.vert_df.loc[set(edge_df['srce'])].copy()
    face_df = eptm.face_df.loc[set(edge_df['face'])].copy()
    cell_df = eptm.cell_df.loc[set(edge_df['cell'])].copy()

    datasets = {'edge': edge_df,
                'face': face_df,
                'vert': vert_df,
                'cell': cell_df}

    sub_eptm = Epithelium('sub', datasets, eptm.specs)
    sub_eptm.reset_index()
    sub_eptm.reset_topo()
    return sub_eptm


def single_cell(eptm, cell):
    edges = eptm.edge_df[eptm.edge_df['cell'] == cell].index
    return get_sub_eptm(eptm, edges)


def scaled_unscaled(func, scale, eptm, geom,
                    args=(), kwargs={}, coords=None):
    """Scales the epithelium by an homotetic factor `scale`, applies
    the function `func`, and scales back to original size.

    Parameters
    ----------
    func: the function to apply to the scaled epithelium
    scale: float, the scale to apply
    eptm: a :class:`Epithelium` instance
    geom: a :class:`Geometry` class
    args: sequence, the arguments to pass to func
    kwargs: dictionary, the keywords arguments
      to pass to func
    coords: the coordinates on which the scaling applies

    Returns
    -------
    res: the result of the function func
    """
    if coords is None:
        coords = eptm.coords
    geom.scale(eptm, scale, coords)
    geom.update_all(eptm)
    res = func(*args, **kwargs)
    geom.scale(eptm, 1/scale, coords)
    geom.update_all(eptm)
    return res
