import logging
import warnings

import numpy as np

import pandas as pd


from itertools import combinations
from ..utils.connectivity import face_face_connectivity

logger = logging.getLogger(name=__name__)


def split_vert(sheet, vert, face, to_rewire, epsilon, recenter=False):
    """ Creates a new vertex and moves it towards the center of face.

    The edges in to_rewire will be connected to the new vertex.

    Parameters
    ----------

    sheet : a :class:`tyssue.Sheet` instance
    vert : int, the index of the vertex to split
    face : int, the index of the face where to move the vertex
    to_rewire : :class:`pd.DataFrame` a subset of `sheet.edge_df`
        where all the edges pointing to (or from) the old vertex will point
        to (or from) the new.

    Note
    ----

    This will leave opened faces and cells

    """
    # Add a vertex
    this_vert = sheet.vert_df.loc[vert:vert]  # avoid type munching
    sheet.vert_df = sheet.vert_df.append(this_vert, ignore_index=True)
    # reset datatypes
    new_vert = sheet.vert_df.index[-1]
    # Move it towards the face center
    r_ia = sheet.face_df.loc[face, sheet.coords] - sheet.vert_df.loc[vert, sheet.coords]
    shift = r_ia * epsilon / np.linalg.norm(r_ia)
    if recenter:
        sheet.vert_df.loc[new_vert, sheet.coords] += shift / 2.0
        sheet.vert_df.loc[vert, sheet.coords] -= shift / 2.0

    else:
        sheet.vert_df.loc[new_vert, sheet.coords] += shift

    # rewire
    sheet.edge_df.loc[to_rewire.index] = to_rewire.replace(
        {"srce": vert, "trgt": vert}, new_vert
    )


def add_vert(eptm, edge):
    """Adds a vertex in the middle of the edge,

    which is split as is its opposite(s)

    Parameters
    ----------
    eptm : a :class:`Epithelium` instance
    edge : int
    the index of one of the half-edges to split

    Returns
    -------
    new_vert : int
    the index to the new vertex
    new_edges : int or list of ints
    index to the new edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges
    new_opp_edges : int or list of ints
    index to the new opposite edge(s). For a sheet, returns
    a single index, for a 3D epithelium, returns
    the list of all the new parallel edges


    In the simple case whith two half-edge, returns
    indices to the new edges, with the following convention:

    s    e    t
      ------>
    * <------ *
    oe

    s    e       ne   t
      ------   ----->
    * <----- * ------ *
        oe   nv   noe

    where "e" is the passed edge as argument, "s" its source "t" its
    target and "oe" its opposite. The returned edges are the ones
    between the new vertex and the input edge's original target.
    """

    srce, trgt = eptm.edge_df.loc[edge, ["srce", "trgt"]]
    opposites = eptm.edge_df[
        (eptm.edge_df["srce"] == trgt) & (eptm.edge_df["trgt"] == srce)
    ]
    parallels = eptm.edge_df[
        (eptm.edge_df["srce"] == srce) & (eptm.edge_df["trgt"] == trgt)
    ]

    new_vert = eptm.vert_df.loc[srce:srce]
    eptm.vert_df = eptm.vert_df.append(new_vert, ignore_index=True)
    new_vert = eptm.vert_df.index[-1]
    eptm.vert_df.loc[new_vert, eptm.coords] = eptm.vert_df.loc[
        [srce, trgt], eptm.coords
    ].mean()

    new_edges = []

    for p, p_data in parallels.iterrows():
        eptm.edge_df.loc[p, "trgt"] = new_vert
        eptm.edge_df = eptm.edge_df.append(p_data, ignore_index=True)
        new_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_edge, "srce"] = new_vert
        eptm.edge_df.loc[new_edge, "trgt"] = trgt
        new_edges.append(new_edge)

    new_opp_edges = []
    for o, o_data in opposites.iterrows():
        eptm.edge_df.loc[o, "srce"] = new_vert
        eptm.edge_df = eptm.edge_df.append(o_data, ignore_index=True)
        new_opp_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_opp_edge, "trgt"] = new_vert
        eptm.edge_df.loc[new_opp_edge, "srce"] = trgt
        new_opp_edges.append(new_opp_edge)

    # ## Sheet special case
    if len(new_edges) == 1:
        new_edges = new_edges[0]
    if len(new_opp_edges) == 1:
        new_opp_edges = new_opp_edges[0]
    elif len(new_opp_edges) == 0:
        new_opp_edges = None
    return new_vert, new_edges, new_opp_edges


def close_face(eptm, face):
    """Closes the face if a single edge is missing.

    This function **does not** close the adjacent and opposite
    faces.
    """
    face_edges = eptm.edge_df[eptm.edge_df["face"] == face]
    srces = set(face_edges["srce"])
    trgts = set(face_edges["trgt"])

    if srces == trgts:
        logger.debug("Face {} already closed".format(face))
        return
    try:
        single_srce, = srces.difference(trgts)
        single_trgt, = trgts.difference(srces)
    except ValueError as err:
        print("Closing only possible with exactly two dangling vertices")
        raise err

    eptm.edge_df = eptm.edge_df.append(face_edges.iloc[0:1], ignore_index=True)
    eptm.edge_df.index.name = "edge"
    new_edge = eptm.edge_df.index[-1]
    eptm.edge_df.loc[new_edge, ["srce", "trgt"]] = single_trgt, single_srce


def drop_two_sided_faces(eptm):
    """Removes all the two (or one?) sided faces from the epithelium

    Note that they are not collapsed, but simply eliminated
    Does not reindex
    """

    num_sides = eptm.edge_df.groupby("face").size()
    if num_sides.min() > 2:
        return

    two_sided = eptm.face_df[num_sides < 3].index
    edges = eptm.edge_df[eptm.edge_df["face"].isin(two_sided)].index
    eptm.edge_df.drop(edges, axis=0, inplace=True)
    eptm.face_df.drop(two_sided, axis=0, inplace=True)


def remove_face(sheet, face):
    """Removes a face from the mesh
    """

    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    verts = edges["srce"].unique()
    new_vert_data = sheet.vert_df.loc[verts].mean()
    sheet.vert_df = sheet.vert_df.append(new_vert_data, ignore_index=True)
    new_vert = sheet.vert_df.index[-1]

    # collapse all edges connected to the face vertices
    sheet.edge_df.replace({"srce": verts, "trgt": verts}, new_vert, inplace=True)

    collapsed = sheet.edge_df.query("srce == trgt")

    sheet.edge_df.drop(collapsed.index, axis=0, inplace=True)
    remanent = sheet.edge_df.query(f"face == {face}").index
    if remanent.shape[0]:
        warnings.warn(f"something fishy with face {face}")
        sheet.edge_df.drop(remanent, axis=0, inplace=True)

    sheet.face_df.drop(face, axis=0, inplace=True)
    sheet.vert_df.drop(verts, axis=0, inplace=True)

    logger.info("removed %d of %d vertices", len(verts), sheet.vert_df.shape[0])
    logger.info("face %d is now dead ", face)
    drop_two_sided_faces(sheet)

    sheet.reset_index()
    sheet.reset_topo()

    return 0


def collapse_edge(sheet, edge, reindex=True, allow_two_sided=False):
    """Collapses edge and merges it's vertices, creating (or increasing the rank of)
    a rosette structure.

    If `reindex` is `True` (the default), resets indexes and topology data.
    The edge is collapsed on the smaller of the srce, trgt indexes (to minimize reindexing impact)

    """
    srce, trgt = np.sort(sheet.edge_df.loc[edge, ["srce", "trgt"]]).astype(int)

    edges = sheet.edge_df[
        ((sheet.edge_df["srce"] == srce) & (sheet.edge_df["trgt"] == trgt))
        | ((sheet.edge_df["srce"] == trgt) & (sheet.edge_df["trgt"] == srce))
    ]

    has_3_sides = np.any(sheet.face_df.loc[edges["face"].astype(int), "num_sides"] < 4)
    if has_3_sides and not allow_two_sided:
        warnings.warn("""This operation would result in a two sided face, aborting""")
        return -1

    sheet.vert_df.loc[srce, sheet.coords] = sheet.vert_df.loc[
        [srce, trgt], sheet.coords
    ].mean(axis=0)
    sheet.vert_df.drop(trgt, axis=0, inplace=True)
    # rewire
    sheet.edge_df.replace({"srce": trgt, "trgt": trgt}, srce, inplace=True)
    # all the edges parallel to the original
    collapsed = sheet.edge_df.query("srce == trgt")
    sheet.edge_df.drop(collapsed.index, axis=0, inplace=True)
    if allow_two_sided:
        drop_two_sided_faces(sheet)
    if reindex:
        sheet.reset_index()
        sheet.reset_topo()
    return 0


def merge_vertices(sheet, vert0, vert1, reindex=True):
    """Merge the two vertices vert0 and vert1 iff they are linked by an edge

    If `reindex` is `True` (the default), resets indexes and topology data

    Note:
    -----
    It is more efficient to call directly `collapse_edge`

    """
    edges = sheet.edge_df[
        ((sheet.edge_df["srce"] == vert0) & (sheet.edge_df["trgt"] == vert1))
        | ((sheet.edge_df["srce"] == vert1) & (sheet.edge_df["trgt"] == vert0))
    ].index
    if not len(edges):
        raise ValueError(
            f"""No edge found between vertices {vert0} and {vert1}, cannot merge"""
        )
    return collapse_edge(sheet, edges[0], reindex)


def condition_4i(eptm):
    """
    Return an index over the faces violating condition 4 i in Okuda et al 2013,
    that is edges (from the same face) sharing two vertices simultaneously.
    """
    num_srces = eptm.edge_df.groupby("face")["srce"].apply(lambda s: len(set(s)))
    num_sides = eptm.face_df["num_sides"]
    return eptm.face_df[(num_srces != num_sides) | (num_sides < 3)].index


def get_neighbour_face_pairs(eptm):
    """
    Returns a pandas Series of neighboring face pairs (as forzen sets of 2 indexes)
    """
    pairs = []
    eptm.edge_df["v_pair"] = eptm.edge_df[["srce", "trgt"]].apply(frozenset, axis=1)

    _ = eptm.edge_df.groupby("v_pair")["face"].apply(
        lambda s: pairs.extend(
            [frozenset((a, b)) for a, b in combinations(s.values, 2)]
        )
    )
    return pd.Series(pairs).drop_duplicates()


def get_num_common_edges(eptm):
    """
    Returns the number of common edges between two neighboring faces
    this number is set to -1 if those faces are opposite and share the
    same edges.
    """
    pairs = get_neighbour_face_pairs(eptm)
    face_v_pair_orbit = eptm.edge_df.groupby("face").apply(
        lambda df: frozenset(df["v_pair"])
    )
    n_common = [
        len(face_v_pair_orbit.loc[fa].intersection(face_v_pair_orbit.loc[fb]))
        if face_v_pair_orbit.loc[fb] != face_v_pair_orbit.loc[fa]
        else -1
        for fa, fb in pairs
    ]
    n_common = pd.Series(n_common, index=pd.Index(pairs, name="face_pairs"))
    return n_common


def condition_4ii(eptm):
    """
    Return an array of face pairs sharing more than two half-edges, as defined
    in Okuda et al. 2013 condition 4 ii
    """
    conmat = face_face_connectivity(eptm, exclude_opposites=True)
    return np.vstack(np.where(conmat > 2)).T


def merge_border_edges(sheet, drop_two_sided=True):
    """Merge edges at the border of a sheet such that no vertex has only
    one incoming and one outgoing edge.

    """

    single_trgt = sheet.edge_df[
        sheet.upcast_trgt(sheet.edge_df.groupby("trgt").apply(len) == 1)
    ]
    faces = set(single_trgt["face"])
    single_srce = sheet.edge_df[
        sheet.upcast_srce(sheet.edge_df.groupby("srce").apply(len) == 1)
    ]
    sheet.edge_df.drop(single_srce.index, inplace=True)
    sheet.edge_df.drop(
        set(single_trgt.index).difference(single_srce.index), inplace=True
    )
    for face in faces:
        close_face(sheet, face)

    if drop_two_sided:
        drop_two_sided_faces(sheet)

    sheet.reset_index(order=False)
    sheet.reset_topo()
