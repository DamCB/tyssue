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


def grow(sheet, face, growth_rate, shrink_col="prefered_vol"):
    """Multiplies the equilibrium volume of face by a
    a factor (1+growth_rate)
    """
    sheet.face_df.loc[face, shrink_col] *= 1 + growth_rate


def shrink(sheet, face, shrink_rate, shrink_col="prefered_vol"):
    """Devides the equilibrium volume of face face by a
    a factor 1+shrink_rate
    """
    sheet.face_df.loc[face, shrink_col] /= 1 + shrink_rate


def exchange(sheet, face, geom, remove_tri_faces=True):
    """
    Execute a type1 transition on the shorter edge of a face.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class
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


def contract(
    sheet,
    face,
    contractile_increase,
    multiple=False,
    contraction_column="contractility",
):
    """
    Contract the face by increasing the 'contractility' parameter
    by contractile_increase

    Parameters
    ----------
    face : id face

    """
    if multiple:
        sheet.face_df.loc[face, contraction_column] *= contractile_increase
    else:
        new_contractility = contractile_increase
        sheet.face_df.loc[face, contraction_column] += new_contractility


def ab_pull(sheet, face, radial_tension, distributed=False):
    """ Adds radial_tension to the face's vertices radial_tension
    """
    verts = sheet.edge_df[sheet.edge_df["face"] == face]["srce"].unique()
    if distributed:
        radial_tension = radial_tension / len(verts)

    sheet.vert_df.loc[verts, "radial_tension"] += radial_tension


def relax(sheet, face, contractility_decrease, contraction_column="contractility"):

    initial_contractility = 1.12
    new_contractility = (
        sheet.face_df.loc[face, contraction_column] / contractility_decrease
    )

    if new_contractility >= (initial_contractility / 2):
        sheet.face_df.loc[face, contraction_column] = new_contractility
        sheet.face_df.loc[face, "prefered_area"] *= contractility_decrease


def increase_linear_tension(sheet, face, line_tension, geom=SheetGeometry):
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(
            sheet.edge_df.loc[edge.name, "dx"], sheet.edge_df.loc[edge.name, "dy"]
        )

        if np.abs(angle_) < np.pi / 4:
            sheet.edge_df.loc[edge.name, "line_tension"] *= line_tension
