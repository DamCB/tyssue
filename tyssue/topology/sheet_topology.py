import logging
import numpy as np
from functools import wraps

from .base_topology import add_vert
from tyssue.utils.decorators import do_undo, validate


logger = logging.getLogger(name=__name__)


def type1_transition(sheet, edge01, epsilon=0.1, remove_tri_faces=True):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings

    Parameters
    ----------
    sheet : a `Sheet` instance
    edge_01 : int
       index of the edge around which the transition takes place
    epsilon : float, optional
       default 0.1, the initial length of the new edge.
    remove_tri_faces : bool, optional
       if True (the default), will remove triangular cells
       after the T1 transition is performed
    """
    # Grab the neighbours
    vert0, vert1, face_b = sheet.edge_df.loc[edge01, ["srce", "trgt", "face"]].astype(
        int
    )
    if sheet.face_df.loc[face_b, "num_sides"] < 4:
        logger.warning(
            """Face %s has 3 sides,
type 1 transition is not allowed"""
            % face_b
        )
        return face_b

    edge10_ = sheet.edge_df[
        (sheet.edge_df["srce"] == vert1) & (sheet.edge_df["trgt"] == vert0)
    ]
    if not len(edge10_.index):
        # edge01 is at a border, we need to add the opposite
        # half edge
        edge10_ = sheet.edge_df.loc[edge01].copy()
        sheet.edge_df = sheet.edge_df.append(edge10_, ignore_index=True)
        edge10 = sheet.edge_df.index[-1]
        sheet.edge_df.loc[edge10, ["srce", "trgt", "face"]] = vert1, vert0, -1
        face_d = -1
    else:
        edge10 = edge10_.index
        face_d = _cast_to_int(edge10_["face"])

    if face_d != -1 and sheet.face_df.loc[face_d, "num_sides"] < 4:
        logger.warning(
            """Face %s has 3 sides,
        type 1 transition is not allowed"""
            % face_d
        )
        return face_d
    edge20_ = sheet.edge_df[
        (sheet.edge_df["trgt"] == vert0) & (sheet.edge_df["face"] == face_b)
    ]
    edge20 = edge20_.index
    vert2 = _cast_to_int(edge20_["srce"])

    edge02_ = sheet.edge_df[
        (sheet.edge_df["srce"] == vert0) & (sheet.edge_df["trgt"] == vert2)
    ]
    edge02 = edge02_.index
    face_a = _cast_to_int(edge02_["face"])

    edge13_ = sheet.edge_df[
        (sheet.edge_df["srce"] == vert1) & (sheet.edge_df["face"] == face_b)
    ]
    edge13 = edge13_.index
    vert3 = _cast_to_int(edge13_["trgt"])

    edge31_ = sheet.edge_df[
        (sheet.edge_df["srce"] == vert3) & (sheet.edge_df["trgt"] == vert1)
    ]
    edge31 = edge31_.index
    face_c = _cast_to_int(edge31_["face"])

    edge13_ = sheet.edge_df[
        (sheet.edge_df["srce"] == vert1) & (sheet.edge_df["face"] == face_b)
    ]
    edge13 = edge13_.index
    vert3 = _cast_to_int(edge13_["trgt"])
    if face_a != -1:
        edge50_ = sheet.edge_df[
            (sheet.edge_df["trgt"] == vert0) & (sheet.edge_df["face"] == face_a)
        ]
        edge50 = edge50_.index
        vert5 = _cast_to_int(edge50_["srce"])

        edge05_ = sheet.edge_df[
            (sheet.edge_df["srce"] == vert0) & (sheet.edge_df["trgt"] == vert5)
        ]
        edge05 = edge05_.index

    elif face_d != -1:
        edge05_ = sheet.edge_df[
            (sheet.edge_df["srce"] == vert0) & (sheet.edge_df["face"] == face_d)
        ]
        edge05 = edge05_.index
        vert5 = _cast_to_int(edge05_["trgt"])

        edge50_ = sheet.edge_df[
            (sheet.edge_df["srce"] == vert5) & (sheet.edge_df["trgt"] == vert0)
        ]
        edge50 = edge50_.index

    else:
        raise ValueError("Edge has no neighbour around vertex %d", vert0)

    logger.debug("faces a, b, c, d")
    logger.debug(face_a, face_b, face_c, face_d)
    logger.debug("vertices 0, 1, 2, 3, 5")
    logger.debug(vert0, vert1, vert2, vert3, vert5)

    # Perform the rearangements
    sheet.edge_df.loc[edge01, "face"] = face_c
    sheet.edge_df.loc[edge10, "face"] = face_a
    sheet.edge_df.loc[edge13, ["srce", "trgt", "face"]] = vert0, vert3, face_b
    sheet.edge_df.loc[edge31, ["srce", "trgt", "face"]] = vert3, vert0, face_c

    sheet.edge_df.loc[edge50, ["srce", "trgt", "face"]] = vert5, vert1, face_a
    sheet.edge_df.loc[edge05, ["srce", "trgt", "face"]] = vert1, vert5, face_d

    # Displace the vertices
    mean_pos = (
        sheet.vert_df.loc[vert0, sheet.coords] + sheet.vert_df.loc[vert1, sheet.coords]
    ) / 2
    face_b_pos = sheet.face_df.loc[face_b, sheet.coords]
    sheet.vert_df.loc[vert0, sheet.coords] = (
        mean_pos - (mean_pos - face_b_pos) * epsilon
    )
    if face_d != -1:
        face_d_pos = sheet.face_df.loc[face_d, sheet.coords]
    else:
        face_d_pos = mean_pos

    sheet.vert_df.loc[vert1, sheet.coords] = (
        mean_pos - (mean_pos - face_d_pos) * epsilon
    )

    sheet.edge_df = sheet.edge_df[sheet.edge_df["face"] != -1].copy()
    sheet.reset_topo()
    if not remove_tri_faces:
        return 0
    # Type 1 transitions might create 3 or 2 sided cells, we remove those
    tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
    while len(tri_faces):
        remove_face(sheet, tri_faces[0])
        tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index

    return 0


def cell_division(sheet, mother, geom, angle=None):

    if not sheet.face_df.loc[mother, "is_alive"]:
        logger.warning("Cell %s is not alive and cannot devide", mother)
        return
    edge_a, edge_b = get_division_edges(sheet, mother, geom, angle=angle, axis="x")
    if edge_a is None:
        return

    vert_a, new_edge_a, new_opp_edge_a = add_vert(sheet, edge_a)
    vert_b, new_edge_b, new_opp_edge_b = add_vert(sheet, edge_b)
    sheet.vert_df.index.name = "vert"
    daughter = face_division(sheet, mother, vert_a, vert_b)
    return daughter


def get_division_edges(sheet, mother, geom, angle=None, axis="x"):

    if angle is None:
        angle = np.random.random() * np.pi

    m_data = sheet.edge_df[sheet.edge_df["face"] == mother]
    # if angle == 0:
    #     face_pos = sheet.face_df.loc[mother, sheet.coords]
    #     rot_pos = sheet.vert_df[sheet.coords].copy()
    #     for c in sheet.coords:
    #         rot_pos.loc[:, c] = rot_pos[c] - face_pos[c]
    # else:
    rot_pos = geom.face_projected_pos(sheet, mother, psi=angle)

    srce_pos = rot_pos.loc[m_data["srce"], axis]
    srce_pos.index = m_data.index
    trgt_pos = rot_pos.loc[m_data["trgt"], axis]
    trgt_pos.index = m_data.index
    try:
        edge_a = m_data[(srce_pos < 0) & (trgt_pos >= 0)].index[0]
        edge_b = m_data[(srce_pos >= 0) & (trgt_pos < 0)].index[0]
    except IndexError:
        print("Failed")
        logger.error("Division of Cell {} failed".format(mother))
        return None, None
    return edge_a, edge_b


def face_division(sheet, mother, vert_a, vert_b):
    """
    Divides the face associated with edges
    indexed by `edge_a` and `edge_b`, splitting it
    in the middle of those edes.
    """
    # mother = sheet.edge_df.loc[edge_a, 'face']

    face_cols = sheet.face_df.loc[mother]
    sheet.face_df = sheet.face_df.append(face_cols, ignore_index=True)
    sheet.face_df.index.name = "face"
    daughter = int(sheet.face_df.index[-1])

    edge_cols = sheet.edge_df[sheet.edge_df["face"] == mother].iloc[0]
    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge_m = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_m, "srce"] = vert_b
    sheet.edge_df.loc[new_edge_m, "trgt"] = vert_a

    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge_d = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_d, "srce"] = vert_a
    sheet.edge_df.loc[new_edge_d, "trgt"] = vert_b

    # ## Discover daughter edges
    m_data = sheet.edge_df[sheet.edge_df["face"] == mother]
    daughter_edges = [new_edge_d]
    srce, trgt = vert_a, vert_b
    srces, trgts = m_data[["srce", "trgt"]].values.T

    while trgt != vert_a:
        srce, trgt = trgt, trgts[srces == trgt][0]
        daughter_edges.append(
            m_data[(m_data["srce"] == srce) & (m_data["trgt"] == trgt)].index[0]
        )
    sheet.edge_df.loc[daughter_edges, "face"] = daughter
    sheet.edge_df.index.name = "edge"
    sheet.reset_topo()
    return daughter


def remove_face(sheet, face):

    if np.isnan(sheet.face_df.loc[face, "num_sides"]):
        logger.info("Face %i is not valid, aborting")
        return

    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    verts = edges["srce"].values

    out_orbits = sheet.get_orbits("srce", "trgt")
    in_orbits = sheet.get_orbits("trgt", "srce")

    new_vert_data = sheet.vert_df.loc[verts].mean()
    sheet.vert_df = sheet.vert_df.append(new_vert_data, ignore_index=True)
    new_vert = sheet.vert_df.index[-1]
    for v in verts:
        out_jes = out_orbits.loc[v].index
        sheet.edge_df.loc[out_jes, "srce"] = new_vert

        in_jes = in_orbits.loc[v].index
        sheet.edge_df.loc[in_jes, "trgt"] = new_vert

    sheet.edge_df = sheet.edge_df[sheet.edge_df["srce"] != sheet.edge_df["trgt"]]

    sheet.edge_df = sheet.edge_df[sheet.edge_df["face"] != face].copy()

    fidx = sheet.face_df.index.delete(face)
    sheet.face_df = sheet.face_df.loc[fidx].copy()

    logger.info("removed {} of {} vertices ".format(len(verts), sheet.vert_df.shape[0]))
    logger.info("face {} is now dead ".format(face))

    vidx = sheet.vert_df.index.delete(verts)
    sheet.vert_df = sheet.vert_df.loc[vidx].copy()
    sheet.reset_index()
    sheet.reset_topo()
    return new_vert


def split_vert(sheet, vert, epsilon=0.0):
    """
    Splits (or opens up) the sheet at vertex `vert`, creating
    new verts and separating the connected opposite edges, see
    ../../doc/illus/vertex_split.png

    Parameters
    ----------
    sheet: a :class:`Sheet` instance
    vert: int, index of the vertex to split
    epsilon: float, the relative amount of recoil
      of the new vertices towards the face centers
    """
    # Grab relevant edges
    vert_out_edges = sheet.edge_df[(sheet.edge_df["srce"] == vert)]
    vert_in_edges = sheet.edge_df[(sheet.edge_df["trgt"] == vert)]
    # Grab relevant faces
    neighbor_faces = set(vert_out_edges["face"])
    if len(neighbor_faces) == 1:
        logger.info(
            """
Chosen vertex %i is bound to a single cell, nothing to do"""
            % vert
        )
        return

    # Create the new vertices
    num_new_vert = len(neighbor_faces) - 1
    vert_data = sheet.vert_df.loc[[vert] * num_new_vert]
    sheet.vert_df = sheet.vert_df.append(vert_data, ignore_index=True)

    new_verts = list(sheet.vert_df.index[-num_new_vert:])
    split_verts = [vert] + new_verts

    # reasign sources and targets in edge_df
    for f, v in zip(neighbor_faces, split_verts):
        eo = vert_out_edges[vert_out_edges["face"] == f].index
        sheet.edge_df.loc[eo, "srce"] = v
        ei = vert_in_edges[vert_in_edges["face"] == f].index
        sheet.edge_df.loc[ei, "trgt"] = v
    sheet.reset_topo()
    sheet.reset_index()

    faces = sheet.face_df.loc[neighbor_faces]
    verts = sheet.vert_df.loc[split_verts]
    dr = -verts[sheet.coords] + faces[sheet.coords].values
    sheet.vert_df.loc[split_verts, sheet.coords] += dr * epsilon


def resolve_t1s(sheet, geom, model, solver, max_iter=60):

    l_th = sheet.settings["threshold_length"]
    i = 0
    while sheet.edge_df.length.min() < l_th:

        for edge in (
            sheet.edge_df[sheet.edge_df.length < l_th].sort_values("length").index
        ):
            try:
                type1_transition(sheet, edge)
            except KeyError:
                continue
            sheet.reset_index()
            sheet.reset_topo()
            geom.update_all(sheet)
        solver.find_energy_min(sheet, geom, model)
        i += 1
        if i > max_iter:
            break


def _cast_to_int(df_value):

    if len(df_value) == 1:
        return int(df_value)
    elif len(df_value) == 0:
        return -1
    else:
        raise ValueError(
            "Trying to retrieve an integer from " "a more than length 1 df "
        )
