import logging
import warnings
import itertools
from functools import wraps

import numpy as np
import pandas as pd

from .sheet_topology import face_division
from .base_topology import (
    add_vert,
    close_face,
    condition_4i,
    condition_4ii,
    collapse_edge,
    remove_face,
)
from .base_topology import split_vert as base_split_vert
from ..geometry.utils import rotation_matrix
from ..core.monolayer import Monolayer
from ..core.sheet import get_opposite

logger = logging.getLogger(name=__name__)
MAX_ITER = 10


def check_condition4(func):
    @wraps(func)
    def decorated(eptm, *args, **kwargs):
        eptm.backup()
        res = func(eptm, *args, **kwargs)
        if len(condition_4i(eptm)) or len(condition_4ii(eptm)):
            print("Invalid epithelium produced, restoring")
            # print("4i on", condition_4i(eptm))
            # print("4ii on", condition_4ii(eptm))
            eptm.restore()
            eptm.topo_changed = True
        return res

    return decorated


def remove_cell(eptm, cell):
    """Removes a tetrahedral cell from the epithelium
    """
    eptm.get_opposite_faces()
    edges = eptm.edge_df.query(f"cell == {cell}")
    if not edges.shape[0] == 12:
        warnings.warn(f"{cell} is not a tetrahedral cell, aborting")
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
    assert eptm.validate()
    return 0


def close_cell(eptm, cell):
    """Closes the cell by adding a face. Assumes a single face is missing
    """
    eptm.face_df = eptm.face_df.append(eptm.face_df.loc[0:0], ignore_index=True)

    new_face = eptm.face_df.index[-1]

    face_edges = eptm.edge_df[eptm.edge_df["cell"] == cell]
    oppo = get_opposite(face_edges)
    new_edges = face_edges[oppo == -1].copy()
    if not new_edges.shape[0]:
        return 0
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
        raise ValueError(
            "Cell has too many edges connected to the vertex, try with another"
        )

    # Tidy up
    for face in all_edges["face"].unique():
        close_face(eptm, face)
    eptm.reset_index()
    eptm.reset_topo()

    for cell in all_edges["cell"]:
        close_cell(eptm, cell)

    eptm.reset_index()
    eptm.reset_topo()

    if isinstance(eptm, Monolayer):
        for vert_ in eptm.vert_df.index[-2:]:
            eptm.guess_vert_segment(vert_)
        for face_ in eptm.face_df.index[-2:]:
            eptm.guess_face_segment(face_)


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

        prev_v, = face_edges[face_edges["trgt"] == vert]["srce"]
        next_v, = face_edges[face_edges["srce"] == vert]["trgt"]
        connected = all_edges[
            all_edges["trgt"].isin((next_v, prev_v))
            | all_edges["srce"].isin((next_v, prev_v))
        ]
        base_split_vert(eptm, vert, to_split, connected, epsilon, recenter)


def get_division_edges(eptm, mother, plane_normal, plane_center=None):
    """Returns an index of the mother cell edges crossed by the division plane, ordered
    clockwize around the division plane normal.


    """
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

    # Order the returned edges so that their centers
    # are oriented counterclockwize in the division plane
    # in preparation for septum creation
    srce_pos = vert_pos.loc[division_edges["srce"], eptm.coords].values
    trgt_pos = vert_pos.loc[division_edges["trgt"], eptm.coords].values
    centers = (srce_pos + trgt_pos) / 2
    theta = np.arctan2(centers[:, 1], centers[:, 0])
    return division_edges.iloc[np.argsort(theta)].index


def get_division_vertices(
    eptm, division_edges=None, mother=None, plane_normal=None, plane_center=None
):

    if division_edges is None:
        division_edges = get_division_edges(eptm, mother, plane_normal, plane_center)
    vertices = []
    for edge in division_edges:
        new_vert, *_ = add_vert(eptm, edge)
        vertices.append(new_vert)
    return vertices


@check_condition4
def cell_division(eptm, mother, geom, vertices=None):

    if vertices is None:
        vertices = get_division_vertices(
            eptm,
            division_edges=None,
            mother=mother,
            plane_normal=None,
            plane_center=None,
        )

    cell_cols = eptm.cell_df.loc[mother:mother]
    eptm.cell_df = eptm.cell_df.append(cell_cols, ignore_index=True)
    eptm.cell_df.index.name = "cell"
    daughter = eptm.cell_df.index[-1]

    pairs = set(
        [
            frozenset([v1, v2])
            for v1, v2 in itertools.product(vertices, vertices)
            if v1 != v2
        ]
    )
    daughter_faces = []

    # devide existing faces
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
    daughter_faces.extend(list(septum))

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

    eptm.reset_index()
    eptm.reset_topo()
    geom.update_all(eptm)

    m_septum_edges = eptm.edge_df[eptm.edge_df["face"] == septum[0]]
    m_septum_norm = m_septum_edges[eptm.ncoords].mean()
    m_septum_pos = eptm.face_df.loc[septum[0], eptm.coords]

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


@check_condition4
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


@check_condition4
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


def _add_edge_to_existing(eptm, cell, vi, vj, new_srce, new_trgt):
    """
    Add edges between vertices v7, v8 and v9 to the existing faces
    """
    cell_edges = eptm.edge_df[eptm.edge_df["cell"] == cell]
    for f, data in cell_edges.groupby("face"):
        if {vi, vj, new_srce, new_trgt}.issubset(set(data["srce"]).union(data["trgt"])):
            good_f = f
            break
    else:
        raise ValueError(
            "no face with vertices {}, {}, {} and {}"
            " was found for cell {}".format(vi, vj, new_srce, new_trgt, cell)
        )
    eptm.edge_df = eptm.edge_df.append(cell_edges.iloc[-1], ignore_index=True)
    new_e = eptm.edge_df.index[-1]
    eptm.edge_df.loc[new_e, ["srce", "trgt", "face", "cell"]] = (
        new_srce,
        new_trgt,
        good_f,
        cell,
    )


def _set_new_pos_IH(eptm, e_1011, vertices):
    """Okuda 2013 equations 46 to 56
    """
    Dl_th = eptm.settings["threshold_length"]

    (v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11) = vertices

    # eq. 49
    r_1011 = -eptm.edge_df.loc[e_1011, eptm.dcoords].values
    u_T = r_1011 / np.linalg.norm(r_1011)
    # eq. 50
    r0 = eptm.vert_df.loc[[v10, v11], eptm.coords].mean(axis=0).values

    v_0ns = []
    for vi, vj, vk in zip((v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        # eq. 54 - 56
        r0i, r0j = eptm.vert_df.loc[[vi, vj], eptm.coords].values - r0[np.newaxis, :]
        w_0k = (r0i / np.linalg.norm(r0i) + r0j / np.linalg.norm(r0j)) / 2
        # eq. 51 - 53
        v_0k = w_0k - (np.dot(w_0k, u_T)) * u_T
        v_0ns.append(v_0k)

    # see definition of l_max bellow eq. 56
    l_max = np.max(
        [np.linalg.norm(v_n - v_m) for (v_n, v_m) in itertools.combinations(v_0ns, 2)]
    )
    # eq. 46 - 49
    for vk, v_0k in zip((v7, v8, v9), v_0ns):
        eptm.vert_df.loc[vk, eptm.coords] = r0 + (Dl_th / l_max) * v_0k


def _get_vertex_pairs_IH(eptm, e_1011):

    srce_face_orbits = eptm.get_orbits("srce", "face")
    v10, v11 = eptm.edge_df.loc[e_1011, ["srce", "trgt"]]
    common_faces = set(srce_face_orbits.loc[v10]).intersection(
        srce_face_orbits.loc[v11]
    )
    if eptm.face_df.loc[common_faces, "num_sides"].min() < 4:
        logger.warning(
            "Edge %i has adjacent triangular faces"
            " can't perform IH transition, aborting",
            e_1011,
        )
        return None

    v10_out = set(eptm.edge_df[eptm.edge_df["srce"] == v10]["trgt"]) - {v11}
    faces_123 = {
        v: set(srce_face_orbits.loc[v])  # .intersection(srce_face_orbits.loc[v10])
        for v in v10_out
    }

    v11_out = set(eptm.edge_df[eptm.edge_df["srce"] == v11]["trgt"]) - {v10}
    faces_456 = {
        v: set(srce_face_orbits.loc[v])  # .intersection(srce_face_orbits.loc[v11])
        for v in v11_out
    }
    v_pairs = []
    for vi in v10_out:
        for vj in v11_out:
            common_face = faces_123[vi].intersection(faces_456[vj])
            if common_face:
                v_pairs.append((vi, vj))
                break
        else:
            return None
    return v_pairs


def _set_new_pos_HI(eptm, fa, v10, v11):

    r0 = eptm.face_df.loc[fa, eptm.coords].values

    norm_a = eptm.edge_df[eptm.edge_df["face"] == fa][eptm.ncoords].mean(axis=0).values
    norm_a = norm_a / np.linalg.norm(norm_a)
    norm_b = -norm_a
    Dl_th = eptm.settings["threshold_length"] * 1.01
    eptm.vert_df.loc[v10, eptm.coords] = r0 + Dl_th / 2 * norm_b
    eptm.vert_df.loc[v11, eptm.coords] = r0 + Dl_th / 2 * norm_a
