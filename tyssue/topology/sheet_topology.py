import logging
import numpy as np
from functools import wraps

import warnings


from .base_topology import add_vert, collapse_edge, close_face, remove_face
from .base_topology import split_vert as base_split_vert
from tyssue.utils.decorators import do_undo, validate


logger = logging.getLogger(name=__name__)
MAX_ITER = 100


def split_vert(
    sheet, vert, face=None, multiplier=1.5, reindex=True, recenter=False, epsilon=None
):
    """Splits a vertex towards the center of the face.

    This operation removes the  face `face` from the neighborhood of the vertex.
    """
    # Get the value for the length of the new edge
    if epsilon is None:
        epsilon = sheet.settings.get("threshold_length", 0.1) * multiplier
    else:
        warnings.warn(
            "The epsilon argument is deprecated and will be removed in a future version. "
            "The length of the new edge should be set by "
            "`sheet.settings['threshold_length]*multiplier` "
        )
    if face is None:
        face = np.random.choice(sheet.edge_df[sheet.edge_df["srce"] == vert]["face"])

    face_edges = sheet.edge_df.query(f"face == {face}")
    (prev_v,) = face_edges[face_edges["trgt"] == vert]["srce"]
    (next_v,) = face_edges[face_edges["srce"] == vert]["trgt"]
    connected = sheet.edge_df[
        sheet.edge_df["trgt"].isin((next_v, prev_v))
        | sheet.edge_df["srce"].isin((next_v, prev_v))
    ]

    base_split_vert(sheet, vert, face, connected, epsilon, recenter)
    for face_ in connected["face"]:
        close_face(sheet, face_)

    if reindex:
        sheet.reset_index()
        sheet.reset_topo()

    return 0


def type1_transition(
    sheet, edge01, *, epsilon=None, remove_tri_faces=True, multiplier=1.5
):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    See Finegan et al. for a description of the algotithm https://doi.org/10.1101/704932


    Parameters
    ----------
    sheet : a `Sheet` instance
    edge_01 : int
       index of the edge around which the transition takes place
    epsilon : float, optional, deprecated
       default 0.1, the initial length of the new edge, in case "threshold_length"
       is not in the sheet.settings
    remove_tri_faces : bool, optional
       if True (the default), will remove triangular cells
       after the T1 transition is performed
    multiplier : float, optional
       default 1.5, the multiplier to the threshold length, so that the
       length of the new edge is set to multiplier * threshold_length


    """

    srce, trgt, face = sheet.edge_df.loc[edge01, ["srce", "trgt", "face"]].astype(int)

    vert = min(srce, trgt)  # find the vertex that wont be reindexed
    ret_code = collapse_edge(sheet, edge01, reindex=True)
    if ret_code != 0:
        warnings.warn(f"Collapse of edge {edge01} failed")
        return ret_code

    split_vert(
        sheet,
        vert,
        face,
        multiplier=multiplier,
        reindex=True,
        recenter=True,
        epsilon=epsilon,
    )

    if not remove_tri_faces:
        return 0
    # Type 1 transitions might create 3 or 2 sided cells, we remove those
    tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
    i = 0
    while len(tri_faces):
        remove_face(sheet, tri_faces[0])
        tri_faces = sheet.face_df[sheet.face_df["num_sides"] < 4].index
        i += 1
        if i > MAX_ITER:
            raise RecursionError
    return 0


def cell_division(sheet, mother, geom, angle=None):
    """ Causes a cell to divide

    Parameters
    ----------

    sheet : a 'Sheet' instance
    mother : face index of target dividing cell
    geom : a 2D geometry
    angle : division angle for newly formed edge

    Returns
    -------
    daughter: face index of new cell

    Notes
    -----
    - Function checks for perodic boundaries if there are, it checks if dividing cell
      rests on an edge of the periodic boundaries if so, it displaces the boundaries
      by a half a period and moves the target cell in the bulk of the tissue. It then
      performs cell division normally and reverts the periodic boundaries to the original
      configuration
    """

    if sheet.settings.get("boundaries") is not None:
        mother_on_periodic_boundary = False
        if (
            sheet.face_df.loc[mother]["at_x_boundary"]
            or sheet.face_df.loc[mother]["at_y_boundary"]
        ):
            mother_on_periodic_boundary = True
            saved_boundary = sheet.specs["settings"]["boundaries"].copy()
            for u, boundary in sheet.settings["boundaries"].items():
                if sheet.face_df.loc[mother][f"at_{u}_boundary"]:
                    period = boundary[1] - boundary[0]
                    sheet.specs["settings"]["boundaries"][u] = [
                        boundary[0] + period / 2.0,
                        boundary[1] + period / 2.0,
                    ]
            geom.update_all(sheet)

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

    if sheet.settings.get("boundaries") is not None and mother_on_periodic_boundary:
        sheet.specs["settings"]["boundaries"] = saved_boundary
        geom.update_all(sheet)
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

    face_cols = sheet.face_df.loc[mother:mother]
    sheet.face_df = sheet.face_df.append(face_cols, ignore_index=True)
    sheet.face_df.index.name = "face"
    daughter = int(sheet.face_df.index[-1])

    edge_cols = sheet.edge_df[sheet.edge_df["face"] == mother].iloc[0:1]
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
        raise ValueError("Trying to retrieve an integer from a more than length 1 df ")
