import warnings
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

    df : input data that will be reshape
    ndim : dimension of the new reshape data.

    Returns
    -------

    df_nd : return array reshaped in ndim.

    """
    df_nd = np.asarray(df).reshape((df.size, 1))
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
    for key in new.keys():
        if specs.get(key) is not None:
            specs[key].update(new[key])
        else:
            specs[key] = new[key]



def set_data_columns(datasets, specs, reset=False):
    """Sets the columns of the dataframes in the datasets dictionnary to
    the uniform values in the specs sub-dictionnaries.

    Parameters
    ----------
    datasets : dict of dataframes
    specs : dict of dicts
    reset : bool, default False

    For each key in specs, the value is a dictionnary whose
    keys are column names for the corresponding dataframe in
    datasets. If there is no such column in the dataframe,
    it is created. If the columns allready exists and  reset is `True`,
    the new value is used.
    """

    for name, spec in specs.items():
        if not len(spec):
            continue
        if "setting" in name:
            continue
        df = datasets.get(name)
        if df is None:
            warnings.warn(
                f"There is no {name} dataset, so the {name}" " spec have no effect."
            )
            continue
        for col, default in spec.items():
            if col in df.columns and reset:
                logger.warning(
                    "Reseting column %s of the %s" " dataset with new specs", col, name
                )
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
            edge_data.reindex(sheet.edge_df["opposite"]).to_numpy(),
            index=edge_data.index,
        )
    elif isinstance(edge_data, pd.DataFrame):
        opposite = pd.DataFrame(
            edge_data.reindex(sheet.edge_df["opposite"]).to_numpy(),
            index=edge_data.index,
            columns=edge_data.columns,
        )
    else:
        opposite = pd.DataFrame(
            np.asarray(edge_data).take(sheet.edge_df["opposite"].to_numpy(), axis=0),
            index=sheet.edge_df.index,
        )
    if free_value is not None:
        opposite = opposite.replace(np.nan, free_value)

    return opposite


def get_sub_eptm(eptm, edges, copy=False):
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

    datasets = {}
    edge_df = eptm.edge_df.loc[edges]
    if edge_df.empty:
        warnings.warn("Sub epithelium appears to be empty")
        return None
    datasets["edge"] = edge_df
    datasets["vert"] = eptm.vert_df.loc[set(edge_df["srce"])]
    datasets["face"] = eptm.face_df.loc[set(edge_df["face"])]
    if "cell" in eptm.datasets:
        datasets["cell"] = eptm.cell_df.loc[set(edge_df["cell"])]

    if copy:
        for elem, df in datasets.items():
            datasets[elem] = df.copy()

    sub_eptm = Epithelium("sub", datasets, eptm.specs)
    sub_eptm.datasets["edge"]["edge_o"] = edges
    sub_eptm.datasets["edge"]["srce_o"] = edge_df["srce"]
    sub_eptm.datasets["edge"]["trgt_o"] = edge_df["trgt"]
    sub_eptm.datasets["edge"]["face_o"] = edge_df["face"]
    if "cell" in eptm.datasets:
        sub_eptm.datasets["edge"]["cell_o"] = edge_df["cell"]

    sub_eptm.datasets["vert"]["srce_o"] = set(edge_df["srce"])
    sub_eptm.datasets["face"]["face_o"] = set(edge_df["face"])
    if "cell" in eptm.datasets:
        sub_eptm.datasets["cell"]["cell_o"] = set(edge_df["cell"])

    sub_eptm.reset_index()
    sub_eptm.reset_topo()
    return sub_eptm


def single_cell(eptm, cell, copy=False):
    """
    Define epithelium instance for all element to a define cell.

    Parameters
    ----------
    eptm : a :class:`Epithelium` instance
    cell : identifier of a cell
    copy : bool, default `False`

    Returns
    -------
    sub_etpm: class:'Epithelium' instance corresponding to the cell
    """
    edges = eptm.edge_df[eptm.edge_df["cell"] == cell].index
    return get_sub_eptm(eptm, edges, copy)


def scaled_unscaled(func, scale, eptm, geom, args=(), kwargs={}, coords=None):
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

    If the execution of function fails, the scaling is still reverted

    Returns
    -------
    res: the result of the function func
    """
    if coords is None:
        coords = eptm.coords
    geom.scale(eptm, scale, coords)
    geom.update_all(eptm)
    try:
        res = func(*args, **kwargs)
    except:
        raise
    finally:
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


def _compute_ar(df, coords):
    u, v = coords
    major = np.ptp(df[u].values)
    minor = np.ptp(df[v].values)
    if major < minor:
        minor, major = major, minor
    return 0 if minor == 0 else major / minor


def ar_calculation(sheet, coords=["x", "y"]):
    """Calculates the aspect ratio of each face of the sheet

    Parameters
    ----------
    eptm : a :class:`Sheet` object
    coords : list of str, optional, default ['x', 'y']
      the coordinates on which to compute the aspect ratio

    Returns
    -------
    AR: pandas series of aspect ratio for all faces.

    Note
    ----
    As is the case in ImageJ, the returned aspect ratio is always higher than 1

    """
    srce_pos = sheet.upcast_srce(sheet.vert_df[sheet.coords])
    srce_pos["face"] = sheet.edge_df["face"]
    return srce_pos.groupby("face").apply(_compute_ar, coords)


def get_next(eptm):
    """
    Returns the indices of the next edge for each edge
    """
    fs_indexed = (
        eptm.edge_df[["face", "srce"]]
        .reset_index()
        .set_index(["face", "srce"], drop=False)
    )
    ft_index = pd.MultiIndex.from_frame(
        eptm.edge_df[["face", "trgt"]], names=["face", "srce"]
    )
    next_ = fs_indexed.loc[ft_index, "edge"].values
    return next_


## small utlity to swap apical and basal segments
def swap_apico_basal(organo):
    """Swap apical and basal segments of an organoid."""
    for elem in ["vert", "face", "edge"]:
        swaped = organo.datasets[elem]["segment"].copy()
        swaped.loc[organo.segment_index("apical", elem)] = "basal"
        swaped.loc[organo.segment_index("basal", elem)] = "apical"
        organo.datasets[elem]["segment"] = swaped


def elem_centered_patch(eptm, elem_idx, neighbour_order, elem):
    """
    Return subeptm centered on the element (cell or face) with index elem_idx
    with neighbour_order neighbours around it.

    Parameters
    ----------
    eptm : a :class:`Epithelim` instance
    index : int, id of the center element
    neighbour_order: int,
        neighbourhood 'degree' around the center element

    Returns
    -------
    patch: an object with the same class as eptm

    """
    if elem not in ("face", "cell"):
        raise ValueError

    elems = pd.Series(elem_idx).append(
        eptm.get_neighborhood(elem_idx, neighbour_order, elem)[elem]
    )
    print(elems, elem)
    edges = eptm.edge_df[eptm.edge_df[elem].isin(elems)].copy()

    vertices = eptm.vert_df.loc[set(edges["srce"])].copy()

    if elem == "cell":
        faces = eptm.face_df.loc[set(edges["face"])].copy()
        cells = eptm.cell_df.loc[elems].copy()

    elif "cell" in edges.columns:

        faces = eptm.face_df.loc[elems].copy()
        cells = eptm.cell_df.loc[set(edges["cell"])].copy()
    else:
        faces = eptm.face_df.loc[elems].copy()
        cells = None

    pos = (
        vertices[eptm.coords].values
        - vertices[eptm.coords].mean(axis=0).values[None, :]
    )
    u, v, rotation = np.linalg.svd(pos, full_matrices=False)

    vertices[eptm.coords] = np.dot(pos, rotation.T)
    patch_dset = {"vert": vertices, "face": faces, "edge": edges}

    if cells is not None:
        patch_dset["cell"] = cells

    patch = eptm.__class__("patch", patch_dset, eptm.specs)
    patch.reset_index()
    return patch


def face_centered_patch(sheet, face, neighbour_order):
    """
    Return subsheet centered on face with a distance of
    neighbour order around the face

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : int, id of the center face
    neighbour_order: int, number of neighbour around the center face

    Returns
    -------
    patch: an object of the same class as the input object
    """
    return elem_centered_patch(sheet, face, neighbour_order, "face")


def cell_centered_patch(eptm, cell, neighbour_order):
    """
    Return subsheet centered on cell with a distance of
    neighbour order around the cell

    Parameters
    ----------
    eptm : a :class:`Epithelium` instance
    face : int, id of the center face
    neighbour_order: int, number of neighbour around the center face

    Returns
    -------
    patch: an object of the same class as the input object
    """
    return elem_centered_patch(eptm, cell, neighbour_order, "cell")
