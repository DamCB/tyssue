"""
Basic event module
=======================


"""

import logging
import numpy as np
from ...topology.base_topology import collapse_edge
from ...topology.sheet_topology import remove_face, type1_transition
from ...topology.sheet_topology import split_vert as sheet_split
from ...topology.bulk_topology import split_vert as bulk_split

from ...geometry.sheet_geometry import SheetGeometry
from ...core.sheet import Sheet
from ...utils import connectivity

import warnings

logger = logging.getLogger(__name__)


def merge_vertices(sheet):
    """Merges all the vertices that are closer than the threshold length

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    """
    d_min = sheet.settings.get("threshold_length", 1e-3)
    short = sheet.edge_df[sheet.edge_df["length"] < d_min].index
    if not short.shape[0]:
        return -1
    logger.info(f"Collapsing {short.shape[0]} edges")
    while short.shape[0]:
        collapse_edge(sheet, short[0], allow_two_sided=True)
        short = sheet.edge_df[sheet.edge_df["length"] < d_min].index
    return 0


def detach_vertices(sheet):
    """Stochastically detaches vertices from rosettes.


    Uses two probabilities `p_4` and `p_5p` stored in
    sheet.settings.

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    """
    # sheet.update_rank()
    st_connect = connectivity.srce_trgt_connectivity(sheet)
    rank = ((st_connect + st_connect.T) > 0).sum(axis=0)
    if isinstance(sheet, Sheet):
        min_rank = 3
        split_vert = sheet_split
    else:
        min_rank = 4
        split_vert = bulk_split

    if rank.max() == min_rank:
        return 0

    dt = sheet.settings.get("dt", 1.0)
    p_4 = sheet.settings.get("p_4", 0.1) * dt
    p_5p = sheet.settings.get("p_5p", 1e-2) * dt

    rank4 = sheet.vert_df[rank == min_rank + 1].index
    dice4 = np.random.random(rank4.size)

    rank5p = sheet.vert_df[rank > min_rank + 1].index
    dice5p = np.random.random(rank5p.size)

    to_detach = np.concatenate([rank4[dice4 < p_4], rank5p[dice5p < p_5p]])
    logger.info(f"Detaching {to_detach.size} vertices")

    for vert in to_detach:
        split_vert(sheet, vert)


def set_value(sheet, element, index, set_value, col):
    """Set the value in the dataset at position index/col.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    element : str: 'cell' or 'face' or 'edge' or 'vert'
    index : index in the datasets[element]
    set_value : value to set.
    col : column from dataset which apply increase_rate.
    """
    sheet.datasets[element].loc[index, col] = set_value


def increase(sheet, element, index, increase_rate, col, multiply=True, bound=None):
    """Increase the value in the dataset at position index/col.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    element : str: 'cell' or 'face' or 'edge' or 'vert'
    index : index in the datasets[element]
    increase_rate : rate use to multiply value in the column col.
    col : column from dataset which apply increase_rate.
    multiply : bool: if true, the current col value is multiplied by increase_rate. if false it is added. Default multiply.
    bound: Higher limit of the modify value. Default None
    """
    if multiply:
        new_value = sheet.datasets[element].loc[index, col] * increase_rate
    else:
        new_value = sheet.datasets[element].loc[index, col] + increase_rate

    if bound is not None:
        if new_value <= bound:
            sheet.datasets[element].loc[index, col] = new_value
    else:
        sheet.datasets[element].loc[index, col] = new_value


def decrease(sheet, element, index, decrease_rate, col, divide=True, bound=None):
    """Decrease the value in the dataset at position index/col.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    element : str: 'cell' or 'face' or 'edge' or 'vert'
    index : index in the datasets[element]
    decrease_rate : rate use to divide value in the column col.
    col : column from element which apply decrease_rate.
    divide : bool: if true the current col value is divide by decrease_rate. If false it is substracted. Default divide.
    bound: lower limit of the modify value. Default None.
    """
    if divide:
        new_value = sheet.datasets[element].loc[index, col] / decrease_rate
    else:
        new_value = sheet.datasets[element].loc[index, col] - decrease_rate

    if bound is not None:
        if new_value >= bound:
            sheet.datasets[element].loc[index, col] = new_value
    else:
        sheet.datasets[element].loc[index, col] = new_value


def exchange(sheet, face, geom, remove_tri_faces=True):
    """
    Execute a type1 transition on the shorter edge of a face.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class
    remove_tri_faces : remove automaticaly triangular faces if existed. Default True.
    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    shorter = edges.length.idxmin()
    # type1_transition(sheet, shorter, 2 * min(edges.length), remove_tri_faces)
    type1_transition(sheet, shorter, epsilon=0.1, remove_tri_faces=remove_tri_faces)
    geom.update_all(sheet)


def remove(sheet, face, geom):
    """
    Removes the face and updates the geometry

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class

    """
    remove_face(sheet, face)
    geom.update_all(sheet)


def ab_pull(sheet, face, radial_tension, distributed=False):
    """ Adds radial_tension to the face's vertices radial_tension

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    radial_tension :
    distributed : bool: If true devide radial_tension by number of vertices, and apply this new radial tension to each vertices. Default not distributed.

    """
    verts = sheet.edge_df[sheet.edge_df["face"] == face]["srce"].unique()
    if distributed:
        radial_tension = radial_tension / len(verts)

    sheet.vert_df.loc[verts, "radial_tension"] += radial_tension


def increase_linear_tension(
    sheet,
    face,
    line_tension_increase,
    multiply=True,
    isotropic=True,
    angle=np.pi / 4,
    limit=100,
):
    """
    Increase edges line tension from face isotropic or according to an angle.

    Parameters
    ----------
    face : index of face
    line_tension_increase : factor for increase line tension value
    multiply : line_tension_increase is multiply or add to the current
                line_tension value. Default True.
    isotropic : all edges are increase, or only a subset of edges. Default True.
    angle : angle below edges are increase by line_tension_increase if
                isotropic is False. Default pi/4
    limit : line_tension stay below this limit value

    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face]

    if isotropic:
        for index, edge in edges.iterrows():
            increase(
                sheet,
                "edge",
                edge.name,
                line_tension_increase,
                "line_tension",
                multiply,
                limit,
            )

    else:
        for index, edge in edges.iterrows():
            angle_ = np.arctan2(
                sheet.edge_df.loc[edge.name, "dx"], sheet.edge_df.loc[edge.name, "dy"]
            )

            if np.abs(angle_) < np.pi / 4:
                increase(
                    sheet,
                    "edge",
                    edge.name,
                    line_tension_increase,
                    "line_tension",
                    multiply,
                    limit,
                )


def grow(sheet, face, growth_rate, growth_col="prefered_vol"):
    """Multiplies the grow columns of face by a factor.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    growth_rate : rate use to multiply value of growth_col of face.
    growth_col : column from face dataframe which apply growth_rate.
                growth_col need to exist in face_df. Default 'prefered_vol'


    :Example:

    >>> print(sheet.face_df[face, 'prefered_vol'])
    10
    >>> grow(sheet, face, 1.7, 'prefered_vol')
    >>> print(sheet.face_df[face, 'prefered_vol'])
    17.0

    """
    warnings.warn("deprecated, use increase function")
    increase(sheet, "face", face, growth_rate, growth_col, True)


def shrink(sheet, face, shrink_rate, shrink_col="prefered_vol"):
    """Devides the shrink_col of face by a factor.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    shrink_rate : rate use to multiply value of shrink_col of face.
    shrink_col : column from face dataframe which apply shrink_rate.
                shrink_col need to exist in face_df. Default 'prefered_vol'
    """
    warnings.warn("deprecated, use decrease function")
    decrease(sheet, "face", face, shrink_rate, shrink_col, True)


def contract(
    sheet, face, contractile_increase, multiply=False, contract_col="contractility"
):
    """
    Contract the face by increasing the 'contractility' parameter
    by contractile_increase

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    contractile_increase : rate use to multiply/add value of contraction_col of face.
    multiply : contractile_increase is multiply/add to the current contract_col value.
                Default False.
    contract_col : column from face dataframe which apply contractile_increase.
                contract_col need to exist in face_df. Default 'contractility'

    """
    warnings.warn("deprecated, use increase function")
    increase(sheet, "face", face, contractile_increase, contract_col, multiply)


def relax(sheet, face, relax_decrease, relax_col="contractility"):
    """
    Relax the face by decreasing the relax_col parameter
    by relax_decrease

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    relax_decrease : rate use to divide value of relax_col of face.
    relax_col : column from face dataframe which apply relax_decrease.
                relax_col need to exist in face_df. Default 'contractility'

    """

    warnings.warn("deprecated, use decrease function")
    initial_contractility = 1.12
    decrease(
        sheet,
        "face",
        face,
        relax_decrease,
        col=relax_col,
        divide=True,
        bound=(initial_contractility / 2),
    )
    increase(sheet, "face", face, relax_decrease, "prefered_area", True)
