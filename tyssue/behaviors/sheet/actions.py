"""
Basic event module
=======================


"""

import logging
import numpy as np
from ...topology.base_topology import collapse_edge
from ...topology.sheet_topology import remove_face, type1_transition, split_vert
from ...geometry.sheet_geometry import SheetGeometry
from ...core.sheet import Sheet

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
        return
    logger.info(f"Collapsing {short.shape[0]} edges")
    while short.shape[0]:
        collapse_edge(sheet, short[0], allow_two_sided=True)
        short = sheet.edge_df[sheet.edge_df["length"] < d_min].index


def detach_vertices(sheet):
    """Stochastically detaches vertices from rosettes.


    Uses two probabilities `p_4` and `p_5p` stored in
    sheet.settings.

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    """
    sheet.update_rank()
    min_rank = 3 if isinstance(sheet, Sheet) else 4

    if sheet.vert_df["rank"].max() == min_rank:
        return 0

    dt = sheet.settings.get("dt", 1.0)
    p_4 = sheet.settings.get("p_4", 0.1) * dt
    p_5p = sheet.settings.get("p_5p", 1e-2) * dt

    rank4 = sheet.vert_df[sheet.vert_df["rank"] == min_rank + 1].index
    dice4 = np.random.random(rank4.size)

    rank5p = sheet.vert_df[sheet.vert_df["rank"] > min_rank + 1].index
    dice5p = np.random.random(rank5p.size)

    to_detach = np.concatenate([rank4[dice4 < p_4], rank5p[dice5p < p_5p]])
    logger.info(f"Detaching {to_detach.size} vertices")
    for vert in to_detach:
        split_vert(sheet, vert)


def grow(sheet, face, growth_rate, growth_col="prefered_vol"):
    """Multiplies the grow columns of face by a factor.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    growth_rate : rate use to multiply value of growth_col of face.
    growth_col : column from face dataframe which apply growth_rate. growth_col need to exist in face_df. Default 'prefered_vol'


    :Example:

    >>> print(sheet.face_df[face, 'prefered_vol'])
    10
    >>> grow(sheet, face, 1.7, 'prefered_vol')
    >>> print(sheet.face_df[face, 'prefered_vol'])
    17.0

    """
    sheet.face_df.loc[face, growth_col] *= growth_rate


def shrink(sheet, face, shrink_rate, shrink_col="prefered_vol"):
    """Devides the shrink_col of face by a factor.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    shrink_rate : rate use to multiply value of shrink_col of face.
    shrink_col : column from face dataframe which apply shrink_rate. shrink_col need to exist in face_df. Default 'prefered_vol'
    """
    sheet.face_df.loc[face, shrink_col] /= shrink_rate


def exchange(sheet, face, geom, remove_tri_faces=True):
    """
    Execute a type1 transition on the shorter edge of a face.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class
    remove_tri_faces : remove automaticaly tri faces if existed. Default True.
    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    shorter = edges.length.idxmin()
    # type1_transition(sheet, shorter, 2 * min(edges.length), remove_tri_faces)
    type1_transition(sheet, shorter, epsilon=0.1,
                     remove_tri_faces=remove_tri_faces)
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


def contract(
    sheet,
    face,
    contractile_increase,
    multiple=False,
    contract_col="contractility",
):
    """
    Contract the face by increasing the 'contractility' parameter
    by contractile_increase

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    contractile_increase : rate use to multiply/add value of contraction_col of face.
    multiple : line_tension_increase is multiply or add to the current line_tension value. Default False.
    contract_col : column from face dataframe which apply contractile_increase. contract_col need to exist in face_df. Default 'contractility'

    """
    if multiple:
        sheet.face_df.loc[face, contract_col] *= contractile_increase
    else:
        new_contractility = contractile_increase
        sheet.face_df.loc[face, contract_col] += new_contractility


def ab_pull(sheet, face, radial_tension, distributed=False):
    """ Adds radial_tension to the face's vertices radial_tension

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    radial_tension :
    distributed : Devide radial_tension by number of vertices, and apply this new radial tension to each vertices. Default False.

    """
    verts = sheet.edge_df[sheet.edge_df["face"] == face]["srce"].unique()
    if distributed:
        radial_tension = radial_tension / len(verts)

    sheet.vert_df.loc[verts, "radial_tension"] += radial_tension


def relax(sheet, face, relax_decrease, relax_col="contractility"):
    """
    Relax the face by decreasing the relax_col parameter
    by relax_decrease

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    relax_decrease : rate use to divide value of relax_col of face.
    relax_col : column from face dataframe which apply relax_decrease. relax_col need to exist in face_df. Default 'contractility'

    """
    initial_contractility = 1.12
    new_contractility = (
        sheet.face_df.loc[face, relax_col] / relax_decrease
    )

    if new_contractility >= (initial_contractility / 2):
        sheet.face_df.loc[face, relax_col] = new_contractility
        sheet.face_df.loc[face, "prefered_area"] *= relax_decrease


def increase_linear_tension(sheet, face, line_tension_increase, multiple=True, isotropic=True, angle=np.pi / 4, limit=100, geom=SheetGeometry):
    """
    Increase edges line tension from face isotropic or according to an angle.

    Parameters
    ----------
    face : index of face
    line_tension_increase : factor for increase line tension value
    multiple : line_tension_increase is multiply or add to the current line_tension value. Default True.
    isotropic : all edges are increase, or only a subset of edges. Default True.
    angle : angle below edges are increase by line_tension_increase if isotropic is False. Default pi/4
    limit : line_tension stay below this limit value
    geom : a geometry class

    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face]

    if isotropic:
        for index, edge in edges.iterrows():
            if multiple:
                new_line_tension = sheet.edge_df.loc[
                    edge.name, "line_tension"] * line_tension_increase
            else:
                new_line_tension = sheet.edge_df.loc[
                    edge.name, "line_tension"] + line_tension_increase

            if new_line_tension <= limit:
                sheet.edge_df.loc[edge.name, "line_tension"] = new_line_tension

    else:
        for index, edge in edges.iterrows():
            angle_ = np.arctan2(
                sheet.edge_df.loc[edge.name, "dx"], sheet.edge_df.loc[
                    edge.name, "dy"]
            )

            if np.abs(angle_) < np.pi / 4:
                if multiple:
                    new_line_tension = sheet.edge_df.loc[
                        edge.name, "line_tension"] * line_tension_increase

                else:
                    new_line_tension = sheet.edge_df.loc[
                        edge.name, "line_tension"] + line_tension_increase
                if new_line_tension <= limit:
                    sheet.edge_df.loc[edge.name,
                                      "line_tension"] = new_line_tension
