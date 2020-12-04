import logging
import warnings
import itertools

import numpy as np
import pandas as pd

from .sheet_topology import face_division
from .base_topology import (
    add_vert,
    close_face,
    collapse_edge,
    remove_face,
)
from .base_topology import split_vert as base_split_vert
from ..geometry.utils import rotation_matrix
from ..core.objects import euler_characteristic, _is_closed_cell
from ..core.monolayer import Monolayer
from ..core.sheet import get_opposite

logger = logging.getLogger(name=__name__)
MAX_ITER = 10


def remove_cell(eptm, cell):
    """Removes a tetrahedral cell from the epithelium."""
    eptm.get_opposite_faces()
    edges = eptm.edge_df.query(f"cell == {cell}")
    if not edges.shape[0] == 12:
        warnings.warn(f"{cell} is not a tetrahedral cell, aborting.")
        return -1
    faces = eptm.face_df.loc[edges["face"].unique()]
    oppo = faces["opposite"][faces["opposite"] != -1]
    verts = eptm.vert_df.loc[edges["srce"].unique()].copy()

    eptm.vert_df = eptm.vert_df.append(verts.mean(), ignore_index=True)
    new_vert = eptm.vert_df.index[-1]

    eptm.vert_df.loc[new_vert, "segment"] = "basal"
    eptm.edge_df.replace(
        {"srce": verts.index, "trgt": verts.index}, new_vert, inplace=True
    )

    collapsed = eptm.edge_df.query("srce == trgt")

    eptm.face_df.drop(faces.index, axis=0, inplace=True)
    eptm.face_df.drop(oppo, axis=0, inplace=True)

    eptm.edge_df.drop(collapsed.index, axis=0, inplace=True)

    eptm.cell_df.drop(cell, axis=0, inplace=True)
    eptm.vert_df.drop(verts.index, axis=0, inplace=True)
    eptm.reset_index()
    eptm.reset_topo()
    return 0


def close_cell(eptm, cell):
    """Closes the cell by adding a face. Assumes a single face is missing"""
    face_edges = eptm.edge_df[eptm.edge_df["cell"] == cell]
    euler_c = euler_characteristic(face_edges)

    if euler_c == 2:
        logger.warning("cell %s is already closed", cell)
        return 0

    if euler_c != 1:
        raise ValueError("Cell has more than one hole")

    eptm.face_df = eptm.face_df.append(eptm.face_df.loc[0:0], ignore_index=True)
    new_face = eptm.face_df.index[-1]

    oppo = get_opposite(face_edges, raise_if_invalid=True)
    new_edges = face_edges[oppo == -1].copy()
    logger.info("closing cell %d", cell)
    new_edges[["srce", "trgt"]] = new_edges[["trgt", "srce"]]
    new_edges["face"] = new_face
    new_edges.index = new_edges.index + eptm.edge_df.index.max()
    eptm.edge_df = eptm.edge_df.append(new_edges, ignore_index=False)

    eptm.reset_index()
    eptm.reset_topo()
    return 0


def split_vert(eptm, vert, face=None, multiplier=1.5):
    """Splits a vertex towards a face.

    Parameters
    ----------
    eptm : a :class:`tyssue.Epithelium` instance
    vert : int the vertex to split
    face : int, optional, the face to split
        if face is None, one face will be chosen at random
    multiplier: float, default 1.5
        length of the new edge(s) in units of eptm.settings["threshold_length"]

    Note on the algorithm
    ---------------------

    For a given face, we look for the adjacent cell with the lowest number
    of faces converging on the vertex. If this number is higher than 4
    we raise a ValueError

    If it's 3, we do a OI transition, resulting in a new edge but no new faces
    If it's 4, we do a IH transition, resulting in a new face and 2 ne edges.

    see ../doc/illus/IH_transition.png
    """
    all_edges = eptm.edge_df[
        (eptm.edge_df["trgt"] == vert) | (eptm.edge_df["srce"] == vert)
    ]

    faces = all_edges.groupby("face").apply(
        lambda df: pd.Series(
            {
                "verts": frozenset(df[["srce", "trgt"]].values.ravel()),
                "cell": df["cell"].iloc[0],
            }
        )
    )

    cells = all_edges.groupby("cell").apply(
        lambda df: pd.Series(
            {
                "verts": frozenset(df[["srce", "trgt"]].values.ravel()),
                "faces": frozenset(df["face"]),
                "size": df.shape[0] // 2,
            }
        )
    )

    # choose a face
    if face is None:
        face = np.random.choice(faces.index)

    pair = faces[faces["verts"] == faces.loc[face, "verts"]].index
    # Take the cell adjacent to the face with the smallest size
    cell = cells.loc[faces.loc[pair, "cell"], "size"].idxmin()
    face = pair[0] if pair[0] in cells.loc[cell, "faces"] else pair[1]
    elements = vert, face, cell

    if cells.loc[cell, "size"] == 3:
        logger.info(f"OI for face {face} of cell {cell}")
        _OI_transition(eptm, all_edges, elements, multiplier)
    elif cells.loc[cell, "size"] == 4:
        logger.info(f"OH for face {face} of cell {cell}")
        _OH_transition(eptm, all_edges, elements, multiplier)
    else:
        return 1
    # Tidy up
    for face in all_edges["face"].unique():
        close_face(eptm, face)
    eptm.reset_index()
    eptm.reset_topo()

    for cell in all_edges["cell"].unique():
        try:
            close_cell(eptm, cell)
        except ValueError as e:
            logger.error(f"Close failed for cell {cell}")
            raise e

    eptm.reset_index()
    eptm.reset_topo()

    if isinstance(eptm, Monolayer):
        for vert_ in eptm.vert_df.index[-2:]:
            eptm.guess_vert_segment(vert_)
        for face_ in eptm.face_df.index[-2:]:
            eptm.guess_face_segment(face_)
    return 0


def _OI_transition(eptm, all_edges, elements, multiplier=1.5, recenter=False):

    epsilon = eptm.settings.get("threshold_length", 0.1) * multiplier
    vert, face, cell = elements

    # Get all the edges bordering this terahedron
    cell_eges = eptm.edge_df.query(f"cell == {cell}")
    prev_vs = cell_eges[cell_eges["trgt"] == vert]["srce"]
    next_vs = cell_eges[cell_eges["srce"] == vert]["trgt"]

    connected = all_edges[
        all_edges["trgt"].isin(next_vs)
        | all_edges["srce"].isin(prev_vs)
        | all_edges["srce"].isin(next_vs)
        | all_edges["trgt"].isin(prev_vs)
    ]
    base_split_vert(eptm, vert, face, connected, epsilon, recenter)


def _OH_transition(eptm, all_edges, elements, multiplier=1.5, recenter=False):

    epsilon = eptm.settings.get("threshold_length", 0.1) * multiplier
    vert, face, cell = elements

    # all_cell_edges = eptm.edge_df.query(f'cell == {cell}').copy()
    cell_edges = all_edges.query(f"cell == {cell}").copy()

    face_verts = cell_edges.groupby("face").apply(
        lambda df: set(df["srce"]).union(df["trgt"]) - {vert}
    )

    for face_, verts_ in face_verts.items():
        if not verts_.intersection(face_verts.loc[face]):
            opp_face = face_
            break
    else:
        raise ValueError

    for to_split in (face, opp_face):
        face_edges = all_edges.query(f"face == {to_split}").copy()

        (prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]
        (next_v,) = face_edges[face_edges["srce"] == vert]["trgt"]
        connected = all_edges[
            all_edges["trgt"].isin((next_v, prev_v))
            | all_edges["srce"].isin((next_v, prev_v))
        ]
        base_split_vert(eptm, vert, to_split, connected, epsilon, recenter)


def get_division_edges(
    eptm, mother, plane_normal, plane_center=None, return_verts=False
):
    """Returns an index of the mother cell edges crossed by the division plane, ordered
    clockwize around the division plane normal.



    """
    if plane_normal is None:
        plane_normal = np.random.normal(size=3)

    plane_normal = np.asarray(plane_normal)
    if plane_center is None:
        plane_center = eptm.cell_df.loc[mother, eptm.coords]

    n_xy = np.linalg.norm(plane_normal[:2])
    theta = -np.arctan2(n_xy, plane_normal[2])
    if np.linalg.norm(plane_normal[:2]) < 1e-10:
        rot = None
    else:
        direction = [plane_normal[1], -plane_normal[0], 0]
        rot = rotation_matrix(theta, direction)

    cell_verts = set(eptm.edge_df[eptm.edge_df["cell"] == mother]["srce"])
    vert_pos = eptm.vert_df.loc[cell_verts, eptm.coords]
    for coord in eptm.coords:
        vert_pos[coord] -= plane_center[coord]
    if rot is not None:
        vert_pos[:] = np.dot(vert_pos, rot)

    mother_edges = eptm.edge_df[eptm.edge_df["cell"] == mother]
    srce_z = vert_pos.loc[mother_edges["srce"], "z"]
    srce_z.index = mother_edges.index
    trgt_z = vert_pos.loc[mother_edges["trgt"], "z"]
    trgt_z.index = mother_edges.index
    division_edges = mother_edges[((srce_z < 0) & (trgt_z >= 0))]
    mother_verts = mother_edges[(srce_z < 0) & (trgt_z < 0)]["srce"].unique()
    daughter_verts = mother_edges[(srce_z >= 0) & (trgt_z >= 0)]["srce"].unique()

    # Order the returned edges so that their centers
    # are oriented counterclockwize in the division plane
    # in preparation for septum creation
    srce_pos = vert_pos.loc[division_edges["srce"], eptm.coords].values
    trgt_pos = vert_pos.loc[division_edges["trgt"], eptm.coords].values
    centers = (srce_pos + trgt_pos) / 2
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    if not return_verts:
        return division_edges.iloc[np.argsort(theta)].index
    return division_edges.iloc[np.argsort(theta)].index, mother_verts, daughter_verts


def get_division_vertices(
    eptm,
    division_edges=None,
    mother=None,
    plane_normal=None,
    plane_center=None,
    return_all=False,
):
    if division_edges is None:
        division_edges, mother_verts, daughter_verts = get_division_edges(
            eptm, mother, plane_normal, plane_center, return_verts=True
        )
    else:
        return_all = False

    septum_vertices = []
    for edge in division_edges:
        new_vert, *_ = add_vert(eptm, edge)
        septum_vertices.append(new_vert)
    if not return_all:
        return septum_vertices
    return septum_vertices, mother_verts, daughter_verts


# @check_condition4
def cell_division(
    eptm, mother, geom, vertices=None, mother_verts=None, daughter_verts=None
):
    if vertices is None:
        vertices, mother_verts, daughter_verts = get_division_vertices(
            eptm,
            mother=mother,
            return_all=True,
        )
    cell_cols = eptm.cell_df.loc[mother:mother]
    eptm.cell_df = eptm.cell_df.append(cell_cols, ignore_index=True)
    eptm.cell_df.index.name = "cell"
    daughter = eptm.cell_df.index[-1]
    if "id" not in eptm.cell_df.columns:
        warnings.warn(
            """Adding 'id' columns to cell_df, as dataframe index is not a reliable
identifier. Consider doing this at initialisation time
    """
        )
        eptm.cell_df["id"] = eptm.cell_df.index.copy()

    daughter_id = eptm.cell_df.id.max() + 1
    mother_id = eptm.cell_df.loc[mother, "id"]

    eptm.cell_df.loc[daughter, "id"] = daughter_id
    pairs = {
        frozenset([v1, v2])
        for v1, v2 in itertools.product(vertices, vertices)
        if v1 != v2
    }

    # divide existing faces-
    daughter_faces = []

    for v1, v2 in pairs:
        v1_faces = eptm.edge_df[eptm.edge_df["srce"] == v1]["face"]
        v2_faces = eptm.edge_df[eptm.edge_df["srce"] == v2]["face"]
        # we should devide a face if both v1 and v2
        # are part of it
        faces = set(v1_faces).intersection(v2_faces)
        for face in faces:
            daughter_faces.append(face_division(eptm, face, v1, v2))

    # septum
    face_cols = eptm.face_df.iloc[-2:]
    eptm.face_df = eptm.face_df.append(face_cols, ignore_index=True)
    eptm.face_df.index.name = "face"
    septum = eptm.face_df.index[-2:]

    num_v = len(vertices)
    num_new_edges = num_v * 2

    edge_cols = eptm.edge_df.iloc[-num_new_edges:]
    eptm.edge_df = eptm.edge_df.append(edge_cols, ignore_index=True)
    eptm.edge_df.index.name = "edge"
    new_edges = eptm.edge_df.index[-num_new_edges:]

    # To keep mother orientation, the first septum face
    # belongs to mother
    for v1, v2, edge, oppo in zip(
        vertices, np.roll(vertices, -1), new_edges[:num_v], new_edges[num_v:]
    ):
        # Mother septum
        eptm.edge_df.loc[edge, ["srce", "trgt", "face", "cell"]] = (
            v1,
            v2,
            septum[0],
            mother,
        )
        # Daughter septum
        eptm.edge_df.loc[oppo, ["srce", "trgt", "face", "cell"]] = (
            v2,
            v1,
            septum[1],
            daughter,
        )

    if (mother_verts is not None) and (daughter_verts is not None):
        # assign edges linked to daughter verts to daughter
        daughter_faces = eptm.edge_df.loc[
            eptm.edge_df["srce"].isin(daughter_verts) & (eptm.edge_df["cell"] == mother)
        ]["face"].unique()

        eptm.edge_df.loc[eptm.edge_df["face"].isin(daughter_faces), "cell"] = daughter
        eptm.edge_df.loc[eptm.edge_df["face"] == septum[1], "cell"] = daughter
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)

    else:
        warnings.warn(
            "This method in cell_division is deprecated and can produce inconsistencies"
        )
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)

        m_septum_edges = eptm.edge_df[eptm.edge_df["face"] == septum[0]]
        m_septum_norm = m_septum_edges[eptm.ncoords].mean()
        m_septum_pos = eptm.face_df.loc[septum[0], eptm.coords]
        if eptm.cell_df[eptm.cell_df["id"] == mother_id].index[0] != mother:
            raise RuntimeError

        # splitting the faces between mother and daughter
        # based on the orientation of the vector from septum
        # center to each face center w/r to the septum norm
        mother_faces = set(eptm.edge_df[eptm.edge_df["cell"] == mother]["face"])
        for face in mother_faces:
            if face == septum[0]:
                continue

            dr = eptm.face_df.loc[face, eptm.coords] - m_septum_pos
            proj = (dr.values * m_septum_norm).sum(axis=0)
            f_edges = eptm.edge_df[eptm.edge_df["face"] == face].index
            if proj < 0:
                eptm.edge_df.loc[f_edges, "cell"] = mother
            else:
                eptm.edge_df.loc[f_edges, "cell"] = daughter

        eptm.reset_index()
        eptm.reset_topo()
    return daughter


def find_rearangements(eptm):
    """Finds the candidates for IH and HI transitions
    Returns
    -------
    edges_HI: set of indexes of short edges
    faces_IH: set of indexes of small triangular faces
    """
    l_th = eptm.settings.get("threshold_length", 1e-6)
    shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return np.array([]), np.array([])
    edges_IH = find_IHs(eptm, shorts)
    faces_HI = find_HIs(eptm, shorts)
    return edges_IH, faces_HI


def find_IHs(eptm, shorts=None):

    l_th = eptm.settings.get("threshold_length", 1e-6)
    if shorts is None:
        shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return []

    edges_IH = shorts.groupby("srce").apply(
        lambda df: pd.Series(
            {
                "edge": df.index[0],
                "length": df["length"].iloc[0],
                "num_sides": min(eptm.face_df.loc[df["face"], "num_sides"]),
                "pair": frozenset(df.iloc[0][["srce", "trgt"]]),
            }
        )
    )
    # keep only one of the edges per vertex pair and sort by length
    edges_IH = (
        edges_IH[edges_IH["num_sides"] > 3]
        .drop_duplicates("pair")
        .sort_values("length")
    )
    return edges_IH["edge"].values


def find_HIs(eptm, shorts=None):
    l_th = eptm.settings.get("threshold_length", 1e-6)
    if shorts is None:
        shorts = eptm.edge_df[(eptm.edge_df["length"] < l_th)]
    if not shorts.shape[0]:
        return []

    max_f_length = shorts.groupby("face")["length"].apply(max)
    short_faces = eptm.face_df.loc[max_f_length[max_f_length < l_th].index]
    faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    return faces_HI


# @check_condition4
def IH_transition(eptm, edge):
    """
    I → H transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the algorithm
    """
    srce, trgt, face, cell = eptm.edge_df.loc[edge, ["srce", "trgt", "face", "cell"]]
    vert = min(srce, trgt)
    collapse_edge(eptm, edge)

    split_vert(eptm, vert, face)

    logger.info(f"IH transition on edge {edge}")
    return 0


# @check_condition4
def HI_transition(eptm, face):
    """
    H → I transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the algorithm
    """
    remove_face(eptm, face)
    vert = eptm.vert_df.index[-1]
    all_edges = eptm.edge_df[
        (eptm.edge_df["srce"] == vert) | (eptm.edge_df["trgt"] == vert)
    ]

    cells = all_edges.groupby("cell").size()
    cell = cells.idxmin()
    face = all_edges[all_edges["cell"] == cell]["face"].iloc[0]
    split_vert(eptm, vert, face)

    logger.info(f"HI transition on edge {face}")
    return 0


def fix_pinch(eptm):
    """Due to rearangements, some faces in an epithelium will have more than one opposite face.

    This method fixes the issue so we can have a valid epithelium back.
    """
    logger.debug("Fixing pinch")
    face_v = eptm.edge_df.groupby("face").apply(lambda df: frozenset(df["srce"]))
    face_v2 = pd.Series(data=face_v.index, index=face_v.values)
    grouped = face_v2.groupby(level=0)
    cardinal = grouped.apply(len)
    faces = face_v2[cardinal > 2].to_list()
    if not faces:
        logger.debug("no pinch found")
        return
    cells = eptm.edge_df.loc[eptm.edge_df["face"].isin(faces), "cell"].unique()
    bad_cells = []
    for cell in cells:
        if not _is_closed_cell(eptm.edge_df.query(f"cell == {cell}")):
            bad_cells.append(cell)

    logger.info("Fixing pinch for cells %s", bad_cells)
    to_remove = eptm.edge_df.loc[
        eptm.edge_df["face"].isin(faces) & (eptm.edge_df["cell"].isin(bad_cells))
    ]

    bad_faces = to_remove["face"].unique()
    bad_edges = to_remove.index.values

    eptm.edge_df = eptm.edge_df.drop(bad_edges)
    eptm.face_df = eptm.face_df.drop(bad_faces)
    eptm.reset_index()
    eptm.reset_topo()
