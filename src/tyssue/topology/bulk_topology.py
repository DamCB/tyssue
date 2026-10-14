import itertools
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from ..core.monolayer import Monolayer
from ..core.objects import _is_closed_cell, euler_characteristic
from ..core.sheet import get_opposite
from ..geometry.utils import rotation_matrix
from .base_topology import add_vert, close_face, collapse_edge, remove_face
from .base_topology import split_vert as base_split_vert
from .sheet_topology import face_division

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
    eptm.vert_df = pd.concat(
        [eptm.vert_df, pd.DataFrame(verts.mean(numeric_only=True))], ignore_index=True
    )
    new_vert = eptm.vert_df.index[-1]

    eptm.vert_df.loc[new_vert, "segment"] = "basal"
    for co in ("srce", "trgt"):
        eptm.edge_df.loc[eptm.edge_df[co].isin(verts.index), co] = new_vert

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
        logger.info("cell %s is already closed", cell)
        return 0

    if euler_c != 1:
        raise ValueError("Cell has more than one hole")

    eptm.face_df = pd.concat([eptm.face_df, eptm.face_df.loc[0:0]], ignore_index=True)
    new_face = eptm.face_df.index[-1]

    oppo = get_opposite(face_edges, raise_if_invalid=True)
    new_edges = face_edges[oppo == -1].copy()
    logger.info("closing cell %d", cell)
    new_edges[["srce", "trgt"]] = new_edges[["trgt", "srce"]]
    new_edges["face"] = new_face
    new_edges.index = new_edges.index + eptm.edge_df.index.max()

    eptm.edge_df = pd.concat([eptm.edge_df, new_edges], ignore_index=False)

    eptm.reset_index()
    eptm.reset_topo()
    return 0


def split_vert(eptm, vert, face=None, multiplier=1.5, recenter=False):
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

    faces = all_edges.groupby("face")[["srce", "trgt", "cell"]].apply(
        lambda df: pd.Series(
            {
                "verts": frozenset(df[["srce", "trgt"]].values.ravel()),
                "cell": df["cell"].iloc[0],
            }
        )
    )

    cells = all_edges.groupby("cell")[["srce", "trgt", "face"]].apply(
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
        _OI_transition(eptm, all_edges, elements, multiplier, recenter=recenter)
    elif cells.loc[cell, "size"] == 4:
        logger.info(f"OH for face {face} of cell {cell}")
        _OH_transition(eptm, all_edges, elements, multiplier, recenter=recenter)
    else:
        logger.info("Nothing happened ")
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

    face_verts = cell_edges.groupby("face")[["srce", "trgt"]].apply(
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

    cell_verts = frozenset(eptm.edge_df[eptm.edge_df["cell"] == mother]["srce"])
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

    eptm.cell_df = pd.concat([eptm.cell_df, cell_cols], ignore_index=True)
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
    eptm.face_df = pd.concat([eptm.face_df, face_cols], ignore_index=True)
    eptm.face_df.index.name = "face"
    septum = eptm.face_df.index[-2:]

    num_v = len(vertices)
    num_new_edges = num_v * 2

    edge_cols = eptm.edge_df.iloc[-num_new_edges:]

    eptm.edge_df = pd.concat([eptm.edge_df, edge_cols], ignore_index=True)
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
    l_th = eptm.settings.get("threshold_length", 1e-2)
    shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return np.array([]), np.array([])
    edges_IH = find_IHs(eptm, shorts)
    faces_HI = find_HIs(eptm, shorts)
    return edges_IH, faces_HI


def find_IHs(eptm, shorts=None):

    l_th = eptm.settings.get("threshold_length", 1e-2)
    if shorts is None:
        shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return []

    def _short_edge(grp):
        # the groups exclude the grouping column
        df = grp.assign(srce=grp.name)
        return pd.Series(
            {
                "edge": df.index[0],
                "length": df["length"].iloc[0],
                "num_sides": min(eptm.face_df.loc[df["face"], "num_sides"]),
                "pair": frozenset(df.iloc[0][["srce", "trgt"]]),
            }
        )

    edges_IH = shorts.groupby("srce")[["trgt", "length", "face"]].apply(_short_edge)
    # keep only one of the edges per vertex pair and sort by length
    edges_IH = (
        edges_IH[edges_IH["num_sides"] > 3]
        .drop_duplicates("pair")
        .sort_values("length")
    )
    return edges_IH["edge"].values


def find_HIs(eptm, shorts=None):
    l_th = eptm.settings.get("threshold_length", 1e-2)
    if shorts is None:
        shorts = eptm.edge_df[(eptm.edge_df["length"] < l_th)]
    if not shorts.shape[0]:
        return []

    max_f_length = shorts.groupby("face")["length"].max()
    short_faces = eptm.face_df.loc[max_f_length[max_f_length < l_th].index]
    faces_HI = short_faces[short_faces["num_sides"] == 3].sort_values("area").index
    return faces_HI


# @check_condition4
def IH_transition(eptm, edge, recenter=False):
    """
    I → H transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the algorithm
    """
    srce, trgt, face, cell = eptm.edge_df.loc[edge, ["srce", "trgt", "face", "cell"]]
    vert = min(srce, trgt)
    collapse_edge(eptm, edge)

    split_vert(eptm, vert, face, recenter=recenter)

    logger.info("IH transition on edge %d", edge)
    return 0


# @check_condition4
def HI_transition(eptm, face, recenter=False):
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
    split_vert(eptm, vert, face, recenter=recenter)

    logger.info("HI transition on face %d", face)
    return 0


def fix_pinch(eptm):
    """Due to rearangements, some faces in an epithelium will have
    more than one opposite face.

    This method fixes the issue so we can have a valid epithelium back.
    """
    logger.debug("Fixing pinch")
    face_v = eptm.edge_df.groupby("face")["srce"].apply(frozenset)
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


def _face_centroids_normals(eptm, faces):
    """Returns the centroid and unit normal of each face in `faces`, computed
    from the vertex positions."""
    centroids, normals = {}, {}
    grouped = eptm.edge_df[eptm.edge_df["face"].isin(faces)].groupby("face")
    for face, edges in grouped:
        srce_pos = eptm.vert_df.loc[edges["srce"], eptm.coords].to_numpy()
        trgt_pos = eptm.vert_df.loc[edges["trgt"], eptm.coords].to_numpy()
        centroids[face] = srce_pos.mean(axis=0)
        normal = np.cross(srce_pos, trgt_pos).sum(axis=0)
        norm = np.linalg.norm(normal)
        normals[face] = normal / norm if norm > 0 else normal
    return centroids, normals


def _ordered_boundary(eptm, face):
    """Vertex indices around `face`, in half-edge order."""
    edges = eptm.edge_df[eptm.edge_df["face"] == face]
    nxt = dict(zip(edges["srce"].astype(int), edges["trgt"].astype(int)))
    start = int(edges["srce"].iloc[0])
    order, v = [start], nxt[start]
    while v != start and len(order) <= len(nxt):
        order.append(v)
        v = nxt[v]
    return order


def _lateral_apico_basal(eptm, face):
    """Returns (apical_verts, basal_verts) of a lateral `face`, from the
    vertices' `segment` column."""
    verts = _ordered_boundary(eptm, face)
    seg = eptm.vert_df.loc[verts, "segment"]
    apical = [v for v in verts if seg[v] == "apical"]
    basal = [v for v in verts if seg[v] == "basal"]
    return apical, basal


def _cell_adjacency(eptm):
    """Set of frozenset({cell_a, cell_b}) for cells sharing a face."""
    eptm.get_opposite_faces()
    face_cell = eptm.edge_df.groupby("face")["cell"].first()
    adjacency = set()
    for face, opp in eptm.face_df["opposite"].items():
        if opp != -1 and face in face_cell.index and opp in face_cell.index:
            adjacency.add(frozenset((int(face_cell[face]), int(face_cell[opp]))))
    return adjacency


def _free_lateral_faces(eptm, segment="lateral"):
    """Indices of the free (border) faces, restricted to `segment` if set."""
    eptm.get_opposite_faces()
    border = eptm.face_df[eptm.face_df["opposite"] == -1]
    if segment is not None and "segment" in eptm.face_df.columns:
        border = border[border["segment"] == segment]
    return list(border.index)


def _split_lateral_face(eptm, face):
    """Cuts a free lateral quad in two across its apico-basal axis.

    New vertices are added on the apical and basal edges and joined by a new
    edge, giving two lateral quads of the same cell. Used when one wide face
    is apposed to two narrower faces of the other front.

    Returns ``(new_apical_vert, new_basal_vert, new_face)``.
    """
    def _segment_edge(face, kind):
        seg = eptm.vert_df["segment"]
        edges = eptm.edge_df[eptm.edge_df["face"] == face]
        match = (seg.loc[edges["srce"]].values == kind) & (
            seg.loc[edges["trgt"]].values == kind
        )
        return edges.index[match][0]

    na, _, _ = add_vert(eptm, _segment_edge(face, "apical"))
    nb, _, _ = add_vert(eptm, _segment_edge(face, "basal"))
    cell = int(eptm.edge_df.loc[eptm.edge_df["face"] == face, "cell"].iloc[0])

    # boundary is [a1, na, a2, b1, nb, b2]: (na, a2, b1, nb) is the new face
    order = _ordered_boundary(eptm, face)
    n = len(order)
    ina = order.index(na)
    half = [order[(ina + k) % n] for k in range(n // 2 + 1)]
    assert half[-1] == nb, "unexpected lateral-face boundary in split"

    new_face_row = eptm.face_df.loc[[face]].copy()
    new_face_row.index = [eptm.face_df.index.max() + 1]
    eptm.face_df = pd.concat([eptm.face_df, new_face_row])
    new_face = eptm.face_df.index[-1]

    def _half_edge(srce, trgt):
        edges = eptm.edge_df
        return edges.index[
            (edges["face"] == face) & (edges["srce"] == srce) & (edges["trgt"] == trgt)
        ][0]

    for srce, trgt in zip(half[:-1], half[1:]):
        eptm.edge_df.loc[_half_edge(srce, trgt), "face"] = new_face

    template = eptm.edge_df[eptm.edge_df["face"] == face].iloc[[0]]
    for srce, trgt, fc in ((nb, na, new_face), (na, nb, face)):
        new_edge = template.copy()
        new_edge.index = [eptm.edge_df.index.max() + 1]
        eptm.edge_df = pd.concat([eptm.edge_df, new_edge])
        idx = eptm.edge_df.index[-1]
        eptm.edge_df.loc[idx, ["srce", "trgt", "face", "cell"]] = [srce, trgt, fc, cell]

    eptm.reset_index()
    eptm.reset_topo()
    logger.info("split lateral face %d into %d and %d", face, face, new_face)
    return na, nb, new_face


def fuse_lateral_faces(eptm, face_g, face_f, validate=True):
    """Welds two apposed free lateral faces into one internal interface.

    Unshared vertices are merged pairwise with their closest counterpart,
    apical to apical and basal to basal, so the two faces become opposites.
    Both faces need the same number of unshared apical and basal vertices.

    Returns 0 on success, -1 if the faces can not be fused (or, with
    `validate`, if the result is invalid, in which case it is rolled back).
    """
    if validate:
        vert_bak = eptm.vert_df.copy()
        edge_bak = eptm.edge_df.copy()
        face_bak = eptm.face_df.copy()

    g_ap, g_ba = _lateral_apico_basal(eptm, face_g)
    f_ap, f_ba = _lateral_apico_basal(eptm, face_f)

    shared = set(g_ap + g_ba) & set(f_ap + f_ba)
    g_ap = [v for v in g_ap if v not in shared]
    g_ba = [v for v in g_ba if v not in shared]
    f_ap = [v for v in f_ap if v not in shared]
    f_ba = [v for v in f_ba if v not in shared]

    if not (g_ap or g_ba):
        return 0  # already coincident
    if len(g_ap) != len(f_ap) or len(g_ba) != len(f_ba):
        return -1  # valence / registration mismatch -- deferred

    pairs = []
    for gv, fv in (g_ap, f_ap), (g_ba, f_ba):
        if not gv:
            continue
        cost = cdist(
            eptm.vert_df.loc[gv, eptm.coords].to_numpy(),
            eptm.vert_df.loc[fv, eptm.coords].to_numpy(),
        )
        rows, cols = linear_sum_assignment(cost)
        pairs += [(gv[r], fv[c]) for r, c in zip(rows, cols)]

    for va, vb in pairs:
        v_keep, v_drop = sorted((int(va), int(vb)))
        eptm.vert_df.loc[v_keep, eptm.coords] = (
            eptm.vert_df.loc[[v_keep, v_drop], eptm.coords].mean(axis=0).to_numpy()
        )
        for co in ("srce", "trgt"):
            eptm.edge_df.loc[eptm.edge_df[co] == v_drop, co] = v_keep
        eptm.vert_df.drop(v_drop, axis=0, inplace=True)

    degenerate = eptm.edge_df.query("srce == trgt")
    if degenerate.shape[0]:
        eptm.edge_df.drop(degenerate.index, axis=0, inplace=True)

    eptm.reset_index()
    eptm.reset_topo()
    fix_pinch(eptm)
    eptm.get_opposite_faces()
    eptm.reset_index()
    eptm.reset_topo()

    if validate and not eptm.validate():
        eptm.vert_df = vert_bak
        eptm.edge_df = edge_bak
        eptm.face_df = face_bak
        eptm.reset_index()
        eptm.reset_topo()
        eptm.get_opposite_faces()
        logger.info("rolled back invalid lateral fusion (%d, %d)", face_g, face_f)
        return -1

    logger.info("fused lateral faces %d and %d", face_g, face_f)
    return 0


def find_fusion_nucleations(eptm, d_max=None, theta_face=45.0, margin=0.5,
                            segment="lateral"):
    """Finds apposed free lateral faces that are approaching but not yet joined.

    Returns pairs ``(g, f)`` of free `segment` faces of two non-adjacent cells
    that share no edge, whose normals are anti-parallel within `theta_face`,
    and where a vertex of g lies within `d_max` of f's plane and over f (within
    `margin` of its radius).
    """
    faces = _free_lateral_faces(eptm, segment)
    if len(faces) < 2:
        return []

    if d_max is None:
        d_max = eptm.settings.get("fusion_distance", None)
    if d_max is None:
        bedges = eptm.edge_df[eptm.edge_df["face"].isin(faces)]
        if "length" in bedges.columns:
            d_max = float(bedges["length"].mean())
        else:
            d_max = 0.5

    centroids, normals = _face_centroids_normals(eptm, faces)
    bedges = eptm.edge_df[eptm.edge_df["face"].isin(faces)]
    face_cell = bedges.groupby("face")["cell"].first().to_dict()
    face_verts = bedges.groupby("face")["srce"].apply(lambda s: set(s)).to_dict()
    radius = {
        face: np.linalg.norm(
            eptm.vert_df.loc[list(face_verts[face]), eptm.coords].to_numpy()
            - centroids[face],
            axis=1,
        ).max()
        for face in faces
    }
    adjacency = _cell_adjacency(eptm)

    cent_arr = np.array([centroids[f] for f in faces])
    tree = cKDTree(cent_arr)
    cos_face = np.cos(np.radians(theta_face))
    max_radius = max(radius.values())

    candidates = []
    for i, j in tree.query_pairs(d_max + 2 * max_radius):
        g, f = faces[i], faces[j]
        cg, cf = face_cell[g], face_cell[f]
        if cg == cf or frozenset((cg, cf)) in adjacency:
            continue
        if len(face_verts[g] & face_verts[f]) >= 2:
            continue  # share an edge -> handled by propagation
        if np.dot(normals[g], normals[f]) > -cos_face:
            continue  # not anti-parallel enough
        gpos = eptm.vert_df.loc[list(face_verts[g]), eptm.coords].to_numpy()
        rel = gpos - centroids[f]
        perp = rel @ normals[f]
        lateral = np.linalg.norm(rel - np.outer(perp, normals[f]), axis=1)
        hit = (np.abs(perp) < d_max) & (lateral < radius[f] * (1 + margin))
        if hit.any():
            candidates.append((g, f, np.abs(perp[hit]).min()))

    candidates.sort(key=lambda c: c[2])
    out, used = [], set()
    for g, f, _ in candidates:
        if g in used or f in used:
            continue
        out.append((int(g), int(f)))
        used.update((g, f))
    return out


def find_fusion_propagations(eptm, theta_fold=20.0, segment="lateral"):
    """Finds free lateral faces that have folded together along a shared edge.

    Returns pairs ``(g, f)`` of free `segment` faces of two non-adjacent cells
    that share an edge, with a fold angle about that edge below `theta_fold`.
    """
    faces = _free_lateral_faces(eptm, segment)
    if len(faces) < 2:
        return []

    centroids, _ = _face_centroids_normals(eptm, faces)
    bedges = eptm.edge_df[eptm.edge_df["face"].isin(faces)]
    face_cell = bedges.groupby("face")["cell"].first().to_dict()
    adjacency = _cell_adjacency(eptm)

    edge_faces = {}
    for face, srce, trgt in bedges[["face", "srce", "trgt"]].to_numpy():
        key = (int(min(srce, trgt)), int(max(srce, trgt)))
        edge_faces.setdefault(key, set()).add(int(face))

    cos_fold = np.cos(np.radians(theta_fold))
    pos = eptm.vert_df[eptm.coords]
    candidates = []
    for (a, b), efaces in edge_faces.items():
        if len(efaces) < 2:
            continue
        e_mid = (pos.loc[a].to_numpy() + pos.loc[b].to_numpy()) / 2
        for g, f in itertools.combinations(sorted(efaces), 2):
            cg, cf = face_cell[g], face_cell[f]
            if cg == cf or frozenset((cg, cf)) in adjacency:
                continue
            r_g = centroids[g] - e_mid
            r_f = centroids[f] - e_mid
            ng, nf = np.linalg.norm(r_g), np.linalg.norm(r_f)
            if ng == 0 or nf == 0:
                continue
            cos_phi = np.dot(r_g, r_f) / (ng * nf)
            if cos_phi > cos_fold:  # angle below theta_fold -> folded together
                candidates.append((g, f, np.arccos(np.clip(cos_phi, -1, 1))))

    candidates.sort(key=lambda c: c[2])
    out, used = [], set()
    for g, f, _ in candidates:
        if g in used or f in used:
            continue
        out.append((int(g), int(f)))
        used.update((g, f))
    return out


def find_fusion_splits(eptm, theta_fold=20.0, segment="lateral"):
    """Finds wide walls straddling a valence mismatch in a closing seam.

    Returns the free `segment` faces folded (within `theta_fold`) against
    faces of two or more other non-adjacent cells, which must be split before
    they can fuse. Faces against the most cells come first.
    """
    faces = _free_lateral_faces(eptm, segment)
    if len(faces) < 3:
        return []

    centroids, _ = _face_centroids_normals(eptm, faces)
    bedges = eptm.edge_df[eptm.edge_df["face"].isin(faces)]
    face_cell = bedges.groupby("face")["cell"].first().to_dict()
    adjacency = _cell_adjacency(eptm)

    edge_faces = {}
    for face, srce, trgt in bedges[["face", "srce", "trgt"]].to_numpy():
        key = (int(min(srce, trgt)), int(max(srce, trgt)))
        edge_faces.setdefault(key, set()).add(int(face))

    cos_fold = np.cos(np.radians(theta_fold))
    pos = eptm.vert_df[eptm.coords]
    # for each free face, collect the distinct opposing cells it is folded against
    opposing = {face: set() for face in faces}
    for (a, b), efaces in edge_faces.items():
        if len(efaces) < 2:
            continue
        e_mid = (pos.loc[a].to_numpy() + pos.loc[b].to_numpy()) / 2
        for g, f in itertools.combinations(sorted(efaces), 2):
            cg, cf = face_cell[g], face_cell[f]
            if cg == cf or frozenset((cg, cf)) in adjacency:
                continue
            r_g = centroids[g] - e_mid
            r_f = centroids[f] - e_mid
            ng, nf = np.linalg.norm(r_g), np.linalg.norm(r_f)
            if ng == 0 or nf == 0:
                continue
            if np.dot(r_g, r_f) / (ng * nf) > cos_fold:  # folded together
                opposing[g].add(cf)
                opposing[f].add(cg)

    splits = [(face, len(cells)) for face, cells in opposing.items() if len(cells) >= 2]
    splits.sort(key=lambda c: c[1], reverse=True)
    return [int(face) for face, _ in splits]


def all_lateral_fusions(eptm, d_max=None, theta_face=45.0, theta_fold=20.0,
                        margin=0.5, segment="lateral", validate=True):
    """Performs every available lateral-face fusion on `eptm`.

    Each round fuses one pair from :func:`find_fusion_propagations` or
    :func:`find_fusion_nucleations`, or else splits a face returned by
    :func:`find_fusion_splits`, until nothing changes.

    Returns the number of fusions performed (splits are not counted).
    """
    count = 0
    max_rounds = 4 * eptm.face_df.shape[0] + 10
    for _ in range(max_rounds):
        splits = find_fusion_splits(eptm, theta_fold=theta_fold, segment=segment)
        flagged = set(splits)
        candidates = find_fusion_propagations(
            eptm, theta_fold=theta_fold, segment=segment
        ) + find_fusion_nucleations(
            eptm, d_max=d_max, theta_face=theta_face, margin=margin, segment=segment
        )
        # faces that need splitting are not fused yet
        candidates = [
            (g, f) for g, f in candidates if g not in flagged and f not in flagged
        ]
        progressed = False
        for g, f in candidates:
            if fuse_lateral_faces(eptm, g, f, validate=validate) == 0:
                count += 1
                progressed = True
                break  # indices changed, re-detect
        if not progressed and splits:
            _split_lateral_face(eptm, splits[0])
            progressed = True  # a piece can now fuse on the next round
        if not progressed:
            break
    return count
