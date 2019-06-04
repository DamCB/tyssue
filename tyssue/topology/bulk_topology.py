import logging
import warnings
import itertools


import numpy as np
import pandas as pd

from .sheet_topology import face_division
from .base_topology import add_vert, close_face
from ..geometry.utils import rotation_matrix
from ..core.monolayer import Monolayer

logger = logging.getLogger(name=__name__)
MAX_ITER = 10


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


def cell_division(eptm, mother, geom, vertices=None):

    if vertices is None:
        vertices = get_division_vertices(
            eptm,
            division_edges=None,
            mother=mother,
            plane_normal=None,
            plane_center=None,
        )

    cell_cols = eptm.cell_df.loc[mother]
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
        return [], []
    edges_IH = find_IHs(eptm, shorts)
    faces_HI = find_HIs(eptm, shorts)
    return edges_IH, faces_HI


def find_IHs(eptm, shorts=None):

    l_th = eptm.settings.get("threshold_length", 1e-6)
    if shorts is None:
        shorts = eptm.edge_df[eptm.edge_df["length"] < l_th]
    if not shorts.shape[0]:
        return []

    edges_IH = shorts.groupby(["srce", "trgt"]).apply(
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


def IH_transition(eptm, e_1011):
    """
    I → H transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the definition of the
    edges, which follow the one in the above article
    """

    v10, v11 = eptm.edge_df.loc[e_1011, ["srce", "trgt"]]
    v_pairs = _get_vertex_pairs_IH(eptm, e_1011)
    if v_pairs is None:
        logger.warning(
            "Edge %i is not a valid junction to perform IH transition, aborting", e_1011
        )
        return -1

    try:
        (v1, v4), (v2, v5), (v3, v6) = v_pairs
    except ValueError:
        logger.warning(
            "Edge %i is not a valid junction to perform IH transition, aborting", e_1011
        )
        return -1

    if len({v1, v4, v2, v5, v3, v6}) != 6:
        logger.warning(
            "Edge %i has adjacent triangular faces"
            " can't perform IH transition, aborting",
            e_1011,
        )
        return -1

    new_vs = eptm.vert_df.loc[[v1, v2, v3]].copy()
    eptm.vert_df = eptm.vert_df.append(new_vs, ignore_index=True)
    v7, v8, v9 = eptm.vert_df.index[-3:]

    cells = []
    srce_cell_orbits = eptm.get_orbits("srce", "cell")
    for vi, vj, vk in [
        (v1, v2, v3),
        (v4, v5, v6),
        (v1, v2, v11),
        (v2, v3, v11),
        (v3, v1, v11),
    ]:
        cell = list(
            set(srce_cell_orbits.loc[vi])
            .intersection(srce_cell_orbits.loc[vj])
            .intersection(srce_cell_orbits.loc[vk])
        )
        cells.append(cell[0] if cell else None)

    cA, cB, cC, cD, cE = cells
    if cA is not None:
        # orient vertices 1,2,3 positively
        r_12 = (
            eptm.vert_df.loc[v2, eptm.coords].values
            - eptm.vert_df.loc[v1, eptm.coords].values
        ).astype(np.float)
        r_23 = (
            eptm.vert_df.loc[v3, eptm.coords].values
            - eptm.vert_df.loc[v2, eptm.coords].values
        ).astype(np.float)
        r_123 = eptm.vert_df.loc[[v1, v2, v3], eptm.coords].mean(axis=0).values
        r_A = eptm.cell_df.loc[cA, eptm.coords].values
        orient = np.dot(np.cross(r_12, r_23), (r_123 - r_A))
    elif cB is not None:
        # orient vertices 4,5,6 negatively
        r_45 = (
            eptm.vert_df.loc[v5, eptm.coords].values
            - eptm.vert_df.loc[v4, eptm.coords].values
        ).astype(np.float)
        r_56 = (
            eptm.vert_df.loc[v6, eptm.coords].values
            - eptm.vert_df.loc[v5, eptm.coords].values
        ).astype(np.float)
        r_456 = eptm.vert_df.loc[[v4, v5, v6], eptm.coords].mean(axis=0).values
        r_B = eptm.cell_df.loc[cB, eptm.coords].values
        orient = -np.dot(np.cross(r_45, r_56), (r_456 - r_B))
    else:
        logger.warning(
            "I - H transition is not possible without cells on either ends"
            " of the edge - would result in a hole"
        )
        return -1

    if orient < 0:
        v1, v2, v3 = v1, v3, v2
        v4, v5, v6 = v4, v6, v5
        cC, cE = cE, cC
    vertices = [v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11]

    for i, va, vb, new in zip(range(3), (v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        # assign v1 -> v10 edges to  v1 -> v7
        e_a10s = eptm.edge_df[
            (eptm.edge_df["srce"] == va) & (eptm.edge_df["trgt"] == v10)
        ].index
        eptm.edge_df.loc[e_a10s, "trgt"] = new
        # assign v10 -> v1 edges to  v7 -> v1
        e_10as = eptm.edge_df[
            (eptm.edge_df["srce"] == v10) & (eptm.edge_df["trgt"] == va)
        ].index
        eptm.edge_df.loc[e_10as, "srce"] = new
        # assign v4 -> v11 edges to  v4 -> v7
        e_b11s = eptm.edge_df[
            (eptm.edge_df["srce"] == vb) & (eptm.edge_df["trgt"] == v11)
        ].index
        eptm.edge_df.loc[e_b11s, "trgt"] = new
        # assign v11 -> v4 edges to  v7 -> v4
        e_11bs = eptm.edge_df[
            (eptm.edge_df["srce"] == v11) & (eptm.edge_df["trgt"] == vb)
        ].index
        eptm.edge_df.loc[e_11bs, "srce"] = new

    _set_new_pos_IH(eptm, e_1011, vertices)

    face = eptm.edge_df.loc[e_1011, "face"]
    new_fs = eptm.face_df.loc[[face, face]].copy()
    eptm.face_df = eptm.face_df.append(new_fs, ignore_index=True)
    fa, fb = eptm.face_df.index[-2:]
    edges_fa_fb = eptm.edge_df.loc[[e_1011] * 6].copy()
    eptm.edge_df = eptm.edge_df.append(edges_fa_fb, ignore_index=True)
    new_es = eptm.edge_df.index[-6:]
    for eA, eB, (vi, vj) in zip(
        new_es[::2], new_es[1::2], [(v7, v8), (v8, v9), (v9, v7)]
    ):
        eptm.edge_df.loc[eA, ["srce", "trgt", "face", "cell"]] = vi, vj, fa, cA
        eptm.edge_df.loc[eB, ["srce", "trgt", "face", "cell"]] = vj, vi, fb, cB

    for cell in cells:
        for face in eptm.edge_df[eptm.edge_df["cell"] == cell]["face"]:
            close_face(eptm, face)

    # Removing the remaining edges and vertices
    todel_edges = eptm.edge_df[
        (eptm.edge_df["srce"] == v10)
        | (eptm.edge_df["trgt"] == v10)
        | (eptm.edge_df["srce"] == v11)
        | (eptm.edge_df["trgt"] == v11)
        | pd.isna(eptm.edge_df["cell"])
    ].index

    eptm.edge_df = eptm.edge_df.loc[eptm.edge_df.index.delete(todel_edges)]
    eptm.vert_df = eptm.vert_df.loc[set(eptm.edge_df.sort_values("srce")["srce"])]
    eptm.face_df = eptm.face_df.loc[set(eptm.edge_df.sort_values("face")["face"])]
    eptm.cell_df = eptm.cell_df.loc[set(eptm.edge_df.sort_values("cell")["cell"])]

    eptm.edge_df.index.name = "edge"
    if isinstance(eptm, Monolayer):
        for vert in (v7, v8, v9):
            eptm.guess_vert_segment(vert)
        for face in fa, fb:
            eptm.guess_face_segment(face)

    eptm.reset_index()
    eptm.reset_topo()
    return 0


def HI_transition(eptm, face):
    """
    H → I transition as defined in Okuda et al. 2013
    (DOI 10.1007/s10237-012-0430-7).
    See tyssue/doc/illus/IH_transition.png for the definition of the
    edges, which follow the one in the above article
    """
    if eptm.face_df.loc[face, "num_sides"] != 3:
        raise ValueError("Only three sided faces can undergo a H-I transition")

    fa = face
    f_edges = eptm.edge_df[eptm.edge_df["face"] == face]
    v7 = f_edges.iloc[0]["srce"]
    v8 = f_edges.iloc[0]["trgt"]
    v9 = f_edges[f_edges["srce"] == v8]["trgt"].iloc[0]

    cA = f_edges["cell"].iloc[0]

    eptm.get_opposite_faces()
    fb = eptm.face_df["opposite"].loc[face]
    if fb > 0:
        cB = eptm.edge_df[eptm.edge_df["face"] == fb]["cell"].iloc[0]
    else:
        cB = None

    cA_edges = eptm.edge_df[eptm.edge_df["cell"] == cA]

    v_pairs = []
    for vk in (v7, v8, v9):
        vis = set(cA_edges[cA_edges["srce"] == vk]["trgt"])
        try:
            vi, = vis.difference({v7, v8, v9})
        except ValueError:
            warnings.warn("Invalid topology for a HI transition, aborting")
            return -1
        vjs = set(eptm.edge_df[eptm.edge_df["srce"] == vk]["trgt"])
        try:
            vj, = vjs.difference({v7, v8, v9, vi})
        except ValueError:
            warnings.warn("Invalid topology for a HI transition, aborting")
            return -1
        v_pairs.append((vi, vj))

    (v1, v4), (v2, v5), (v3, v6) = v_pairs

    srce_cell_orbit = eptm.get_orbits("srce", "cell")
    cells = [cA, cB]
    for (vi, vj, vk) in [(v1, v2, v4), (v2, v3, v5), (v1, v3, v4)]:
        cell = list(
            set(srce_cell_orbit.loc[vi])
            .intersection(srce_cell_orbit.loc[vj])
            .intersection(srce_cell_orbit.loc[vk])
        )

        cells.append(cell[0] if cell else None)

    cA, cB, cC, cD, cE = cells

    eptm.vert_df = eptm.vert_df.append(eptm.vert_df.loc[[v8, v9]], ignore_index=True)
    eptm.vert_df.index.name = "vert"
    v10, v11 = eptm.vert_df.index[-2:]
    _set_new_pos_HI(eptm, fa, v10, v11)

    for vi, vj, vk in zip((v1, v2, v3), (v4, v5, v6), (v7, v8, v9)):
        e_iks = eptm.edge_df[
            (eptm.edge_df["srce"] == vi) & (eptm.edge_df["trgt"] == vk)
        ].index
        eptm.edge_df.loc[e_iks, "trgt"] = v10

        e_kis = eptm.edge_df[
            (eptm.edge_df["srce"] == vk) & (eptm.edge_df["trgt"] == vi)
        ].index
        eptm.edge_df.loc[e_kis, "srce"] = v10

        e_jks = eptm.edge_df[
            (eptm.edge_df["srce"] == vj) & (eptm.edge_df["trgt"] == vk)
        ].index
        eptm.edge_df.loc[e_jks, "trgt"] = v11

        e_kjs = eptm.edge_df[
            (eptm.edge_df["srce"] == vk) & (eptm.edge_df["trgt"] == vj)
        ].index
        eptm.edge_df.loc[e_kjs, "srce"] = v11

    # Closing the faces with v10 → v11 edges
    for cell in cells:
        for face in eptm.edge_df[eptm.edge_df["cell"] == cell]["face"]:
            close_face(eptm, face)

    # Removing the remaining edges and vertices
    todel_edges = eptm.edge_df[
        (eptm.edge_df["srce"] == v7)
        | (eptm.edge_df["trgt"] == v7)
        | (eptm.edge_df["srce"] == v8)
        | (eptm.edge_df["trgt"] == v8)
        | (eptm.edge_df["srce"] == v9)
        | (eptm.edge_df["trgt"] == v9)
    ].index

    eptm.edge_df = eptm.edge_df.loc[eptm.edge_df.index.delete(todel_edges)]
    eptm.vert_df = eptm.vert_df.loc[eptm.vert_df.index.delete([v7, v8, v9])]
    orphan_faces = set(eptm.face_df.index).difference(eptm.edge_df.face)
    eptm.face_df = eptm.face_df.loc[
        eptm.face_df.index.delete(list(orphan_faces))
    ].copy()
    eptm.edge_df.index.name = "edge"
    eptm.reset_index()
    eptm.reset_topo()


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
    v10_out = set(eptm.edge_df[eptm.edge_df["srce"] == v10]["trgt"]) - {v11}
    faces_123 = {v: set(srce_face_orbits.loc[v]) for v in v10_out}

    v11_out = set(eptm.edge_df[eptm.edge_df["srce"] == v11]["trgt"]) - {v10}
    faces_456 = {v: set(srce_face_orbits.loc[v]) for v in v11_out}
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
