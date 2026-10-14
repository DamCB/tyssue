"""
Basic event module
=======================


"""

import logging
import warnings

import numpy as np
from scipy.spatial import cKDTree

from ...core.sheet import Sheet
from ...geometry.planar_geometry import update_boundary_index
from ...topology.base_topology import collapse_edge
from ...topology.bulk_topology import split_vert as bulk_split
from ...topology.sheet_topology import remove_face
from ...topology.sheet_topology import split_vert as sheet_split
from ...topology.sheet_topology import t3_transition, type1_transition
from ...utils import connectivity

logger = logging.getLogger(__name__)


def merge_vertices(sheet):
    """Merges all the vertices that are closer than the threshold length

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    """
    d_min = sheet.settings.get("threshold_length", 1e-3)
    short = _short_edges(sheet, d_min)
    np.random.shuffle(short)
    if not short.shape[0]:
        return -1
    logger.info(f"Collapsing {short.shape[0]} edges")
    while short.shape[0]:
        collapse_edge(sheet, short[0], allow_two_sided=False)
        short = _short_edges(sheet, d_min)
        np.random.shuffle(short)

    sheet.network_changed = True

    return 0


def _short_edges(sheet, d_min):
    """Labels of the edges shorter than `d_min`, as a writeable array."""
    mask = sheet.edge_df["length"].to_numpy() < d_min
    return sheet.edge_df.index[mask].to_numpy(copy=True)


def detach_vertices(sheet):
    """Stochastically detaches vertices from rosettes.


    Uses two probabilities `p_4` and `p_5p` stored in
    sheet.settings.

    If `sheet.settings["detach_boundary"]` is False (default True), vertices on
    the free border of a :class:`Sheet` are never detached, which keeps the
    rank 4 border vertices left by type 3 transitions.

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    """
    rank = connectivity.vertex_rank(sheet)
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
    # filtered after the draws so the random stream is the same either way
    if (
        to_detach.size
        and isinstance(sheet, Sheet)
        and not sheet.settings.get("detach_boundary", True)
    ):
        update_boundary_index(sheet, with_faces=False)
        on_border = sheet.vert_df.loc[to_detach, "boundary"].to_numpy() == 1
        to_detach = to_detach[~on_border]
    if to_detach.size:
        logger.info(f"Detaching {to_detach.size} vertices")
        nv = sheet.Nv
        for vert in to_detach:
            split_vert(sheet, vert)

        if sheet.Nv != nv:
            sheet.network_changed = True


def t3_transitions(sheet):
    """Merges every border vertex closer than `t3_threshold` to a free edge of
    another cell into that edge (type 3 transitions).

    Uses `t3_threshold` (default 1e-2) and `t3_skin` (default
    `10 * t3_threshold`) from `sheet.settings`.

    Candidate pairs closer than `t3_threshold + t3_skin` are kept in a Verlet
    list, rebuilt when a border vertex has moved by more than half the skin.

    Parameters
    ----------
    sheet : a :class:`Sheet` object

    Returns
    -------
    the number of transitions performed, or -1 if the sheet has no free border
    """
    if not isinstance(sheet, Sheet):
        raise TypeError("T3 transitions are only implemented for Sheet objects")

    d_min = sheet.settings.get("t3_threshold", 1e-2)
    skin = sheet.settings.get("t3_skin", 10 * d_min)
    verts, edges, _ = find_t3_candidates(sheet, d_min, skin)
    if verts is None:
        return -1
    if not len(verts):
        return 0

    used_verts, used_edges = set(), set()
    done = 0
    edge_index = sheet.edge_df.index
    vert_index = sheet.vert_df.index
    for v_pos, e_pos in zip(verts, edges):
        vert, edge = vert_index[v_pos], edge_index[e_pos]
        srce, trgt = sheet.edge_df.loc[edge, ["srce", "trgt"]]
        # at most one transition per vertex and edge per call
        if {vert, srce, trgt} & used_verts or edge in used_edges:
            continue
        if t3_transition(sheet, vert, edge, reindex=False) == 0:
            used_verts.update((vert, srce, trgt))
            used_edges.add(edge)
            done += 1
    if done:
        sheet.reset_topo()
        logger.info("Performed %d T3 transitions", done)
    return done


def find_t3_candidates(sheet, d_min, skin=None):
    """Border vertex / free edge pairs closer than `d_min`, closest first.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    d_min : float, the merge distance
    skin : float, default `10 * d_min`, the Verlet list margin

    Returns
    -------
    verts, edges : positional indices in vert_df and edge_df
    dists : the corresponding vertex to edge distances

    All three are None if the sheet has no free edge.
    """
    if skin is None:
        skin = 10 * d_min
    update_boundary_index(sheet, with_faces=False)
    bnd = sheet._boundary
    if not len(bnd.free_edges):
        return None, None, None

    # column by column, faster than vert_df[coords]
    pos = np.column_stack(
        [sheet.vert_df[c].to_numpy(dtype=float) for c in sheet.coords]
    )
    cache = getattr(sheet, "_t3_cache", None)
    if cache is None or cache["signature"] != bnd.signature:
        cache = {"signature": bnd.signature, **_t3_topology(sheet, bnd)}
        sheet._t3_cache = cache

    # rebuild the list once a border vertex moved by more than skin / 2
    if (
        cache.get("d_min") != d_min
        or cache.get("skin") != skin
        or np.any(
            np.linalg.norm(pos[cache["border"]] - cache["ref_pos"], axis=1) > skin / 2
        )
    ):
        cand_v, cand_k = _t3_neighbor_list(bnd, cache, pos, d_min + skin)
        cache.update(
            d_min=d_min,
            skin=skin,
            ref_pos=pos[cache["border"]],
            verts=cand_v,
            free_k=cand_k,
        )

    cand_v, cand_k = cache["verts"], cache["free_k"]
    if not len(cand_v):
        empty = np.array([], dtype=np.int64)
        return empty, empty, np.array([])

    pa = pos[bnd.free_srce[cand_k]]
    pb = pos[bnd.free_trgt[cand_k]]
    pv = pos[cand_v]
    ab = pb - pa
    t = np.einsum("ij,ij->i", pv - pa, ab) / np.maximum(
        np.einsum("ij,ij->i", ab, ab), 1e-30
    )
    dist = np.linalg.norm(pa + t[:, None] * ab - pv, axis=1)
    # a projection close to an end is left to that end's own edges
    close = (dist < d_min) & (t > 0.05) & (t < 0.95)
    order = np.argsort(dist[close], kind="stable")
    return (
        cand_v[close][order],
        bnd.free_edges[cand_k[close]][order],
        dist[close][order],
    )


def _t3_topology(sheet, bnd):
    """The parts of the T3 search that only depend on topology.

    nv, nf     element counts the keys are built with
    border     positional indices of every vertex on a free edge
    face_keys  sorted `vert * Nf + face` keys, a border vertex in a face
    link_keys  sorted `srce * Nv + trgt` keys, a half-edge from or to a border
               vertex
    """
    edge_df = sheet.edge_df
    nv, nf = len(sheet.vert_df), len(sheet.face_df)
    border = np.union1d(bnd.free_srce, bnd.free_trgt)
    on_border = np.zeros(nv + 1, dtype=bool)  # extra slot for get_indexer's -1
    on_border[border] = True

    e_srce = sheet.vert_df.index.get_indexer(edge_df["srce"].to_numpy())
    e_trgt = sheet.vert_df.index.get_indexer(edge_df["trgt"].to_numpy())
    e_face = sheet.face_df.index.get_indexer(edge_df["face"].to_numpy())

    in_face = on_border[e_srce]
    face_keys = np.unique(e_srce[in_face] * nf + e_face[in_face])
    linked = on_border[e_srce] | on_border[e_trgt]
    link_keys = np.unique(e_srce[linked] * nv + e_trgt[linked])
    return {
        "nv": nv,
        "nf": nf,
        "border": border,
        "face_keys": face_keys,
        "link_keys": link_keys,
    }


def _in_sorted(keys, sorted_keys):
    """Membership of `keys` in the sorted array `sorted_keys`."""
    if not len(sorted_keys):
        return np.zeros(len(keys), dtype=bool)
    idx = np.minimum(np.searchsorted(sorted_keys, keys), len(sorted_keys) - 1)
    return sorted_keys[idx] == keys


def _t3_neighbor_list(bnd, cache, pos, radius):
    """(border vertex, free edge) pairs possibly closer than `radius`.

    Returns positional vertex indices and indices into the boundary index
    free-edge arrays, without the pairs :func:`t3_transition` would refuse.
    """
    empty = np.array([], dtype=np.int64)
    pa = pos[bnd.free_srce]
    pb = pos[bnd.free_trgt]
    mid = (pa + pb) / 2
    half = np.linalg.norm(pb - pa, axis=1) / 2

    # skip non finite positions
    border = cache["border"]
    border = border[np.isfinite(pos[border]).all(axis=1)]
    edges_k = np.flatnonzero(np.isfinite(mid).all(axis=1) & np.isfinite(half))
    if not (len(border) and len(edges_k)):
        return empty, empty

    tree = cKDTree(pos[border])
    hits = tree.query_ball_point(mid[edges_k], half[edges_k] + radius)
    counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
    if not counts.sum():
        return empty, empty
    free_k = np.repeat(edges_k, counts)
    verts = border[np.concatenate(hits).astype(np.int64)]

    nv, nf = cache["nv"], cache["nf"]
    in_face = _in_sorted(verts * nf + bnd.free_face[free_k], cache["face_keys"])
    srce, trgt = bnd.free_srce[free_k], bnd.free_trgt[free_k]
    duplicated = (
        (verts == srce)
        | (verts == trgt)
        | _in_sorted(srce * nv + verts, cache["link_keys"])  # a -> v
        | _in_sorted(verts * nv + trgt, cache["link_keys"])  # v -> b
    )
    keep = ~(in_face | duplicated)
    return verts[keep], free_k[keep]


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
    multiply : bool:
        if True, the current col value is multiplied by increase_rate.
        if False it is added. Default True.
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
    divide : bool: if true the current col value is divide by decrease_rate.
        If false it is substracted. Default divide.
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
    type1_transition(sheet, shorter, remove_tri_faces=remove_tri_faces)
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
    """Adds radial_tension to the face's vertices radial_tension

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of face
    radial_tension :
    distributed : bool: If true devide radial_tension by number of vertices,
        and apply this new radial tension to each vertices. Default not distributed.

    """
    verts = sheet.edge_df[sheet.edge_df["face"] == face]["srce"].unique()
    if distributed:
        radial_tension = radial_tension / len(verts)

    # an integer column can not hold a float increment
    if sheet.vert_df["radial_tension"].dtype.kind in "iub":
        sheet.vert_df["radial_tension"] = sheet.vert_df["radial_tension"].astype(float)

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
    warnings.warn("contract is deprecated, use increase function")
    increase(sheet, "face", face, contractile_increase, contract_col, multiply)
