import numpy as np
import logging
import pandas as pd

logger = logging.getLogger(name=__name__)


def _to_2d(df):
    df_2d = to_nd(df, 2)
    return df_2d


def _to_3d(df):
    df_3d = to_nd(df, 3)
    return df_3d


def to_nd(df, ndim):
    """
    Give a new shape to an input data by duplicating its column.
    Parameters
    ----------
    df: input data that will be reshape
    ndim: dimension of the new reshape data.

    Returns
    -------
    df_nd: return array reshaped in ndim.
    """
    df_nd = np.asarray(df).repeat(ndim).reshape((df.size, ndim))
    return df_nd


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
    """
    Add element to the new dictionary to the specs dictionary.
    Update value if the key already exist.

    Parameters
    ----------
    specs: specification that will be modified
    new: dictionary of new specification
    """
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

    Parameters
    ----------
    sheet: a :class:`Sheet` instance
    edge_data:  dataframe contain value of edge

    Returns
    -------
    opposite: pandas series contain value of opposite edge
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
    """
    Define sub-epithelium corresponding to the edges.

    Parameters
    ----------
    eptm: a :class:`Epithelium` instance
    edges: list of edges includes in the sub-epithelium

    Returns
    -------
    sub_eptm: a :class:`Epithelium` instance
    """
    from ..core.objects import Epithelium

    edge_df = eptm.edge_df.loc[edges]
    vert_df = eptm.vert_df.loc[set(edge_df['srce'])]  # .copy()
    face_df = eptm.face_df.loc[set(edge_df['face'])]  # .copy()
    cell_df = eptm.cell_df.loc[set(edge_df['cell'])]  # .copy()

    datasets = {'edge': edge_df,
                'face': face_df,
                'vert': vert_df,
                'cell': cell_df}

    sub_eptm = Epithelium('sub', datasets, eptm.specs)
    sub_eptm.datasets['edge']['edge_o'] = edges
    sub_eptm.datasets['edge']['srce_o'] = edge_df['srce']
    sub_eptm.datasets['edge']['trgt_o'] = edge_df['trgt']
    sub_eptm.datasets['edge']['face_o'] = edge_df['face']
    sub_eptm.datasets['edge']['cell_o'] = edge_df['cell']

    sub_eptm.datasets['vert']['srce_o'] = set(edge_df['srce'])
    sub_eptm.datasets['face']['face_o'] = set(edge_df['face'])
    sub_eptm.datasets['cell']['cell_o'] = set(edge_df['cell'])

    sub_eptm.reset_index()
    sub_eptm.reset_topo()
    return sub_eptm


def single_cell(eptm, cell):
    """
    Define epithelium instance for all element to a define cell.

    Parameters
    ----------
    eptm: a :class:`Epithelium` instance
    cell: identifier of a cell

    Returns
    -------
    sub_etpm: class:'Epithelium' instance corresponding to the cell
    """
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
    geom.scale(eptm, 1 / scale, coords)
    geom.update_all(eptm)
    return res


def modify_segments(eptm, modifiers):
    """Modifies the datasets of a segmented epithelium
    according to the passed modifiers.

    Parameters
    ----------
    eptm : :class:`tyssue.Epithelium`
    modifiers : nested dictionnary

    Note
    ----
    This functions assumes that the epithelium has a `segment_index`
    method as implemented in the :class:`tyssue.Monolayer`.

    Example
    -------
    >>> modifiers = {
    >>>     'apical' : {
    >>>         'edge': {'line_tension': 1.},
    >>>         'face': {'prefered_area': 0.2},
    >>>     },
    >>>     'basal' : {
    >>>         'edge': {'line_tension': 3.},
    >>>         'face': {'prefered_area': 0.1},
    >>>     }
    >>> modify_segments(monolayer, modifiers)
    >>> monolayer.ver_df.loc[monolayer.apical_edges,
    >>>                      'line_tension'].unique()[0] == 1.
    True
    """

    for segment, spec in modifiers.items():
        for element, parameters in spec.items():
            idx = eptm.segment_index(segment, element)
            for param_name, param_value in parameters.items():
                eptm.datasets[element].loc[idx, param_name] = param_value


def ar_calculation(sheet):
    """ Calculate the aspect ratio of each fac of the sheet

    Parameters
    ----------
    eptm: a :class:Sheet object

    Returns
    -------
    AR: pandas series of aspect ratio for all faces.

    """
    face_vertices = sheet.face_polygons(sheet.coords)
    ar = []
    for face_number in range(len(sheet.face_df)):
        x_value = []
        z_value = []
        for i in range(len(face_vertices[face_number])):
            x_value.append(face_vertices[face_number][i][0])
            z_value.append(face_vertices[face_number][i][2])
        x_value = np.array(x_value)
        z_value = np.array(z_value)
        major = x_value.ptp()
        minor = z_value.ptp()

        if major < minor:
            tmp = major
            major = minor
            minor = tmp
        if minor == 0:
            ar.append(0)
        else:
            ar.append(major / minor)
    return pd.Series(ar)
