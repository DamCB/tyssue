import logging
import warnings
from functools import wraps
from itertools import count

import numpy as np

from ..core.sheet import Sheet
from .base_topology import (  # noqa: F401
    add_vert,
    close_face,
    collapse_edge,
    condition_4i,
    condition_4ii,
    drop_two_sided_faces,
    get_neighbour_face_pairs,
    get_num_common_edges,
    merge_border_edges,
    merge_vertices,
    split_vert,
)
from .bulk_topology import (  # noqa: F401
    HI_transition,
    IH_transition,
    all_lateral_fusions,
    find_HIs,
    find_IHs,
    find_fusion_nucleations,
    find_fusion_propagations,
    find_fusion_splits,
    find_rearangements,
    fuse_lateral_faces,
)
from .sheet_topology import remove_face, t3_transition, type1_transition

MAX_ITER = 10
logger = logging.getLogger(name=__name__)


class TopologyChangeError(ValueError):
    """Raised when trying to assign values without
    the correct length to an epithelium dataset
    """

    pass


def _t2_flag(with_t2, with_t3, stacklevel=3):
    """Resolves the deprecated `with_t3` alias of `with_t2`."""
    if with_t3 is None:
        return with_t2
    warnings.warn(
        "`with_t3` is deprecated, use `with_t2`: the removal of small triangular "
        "faces is a type 2 transition. In tyssue, T3 now refers to a border vertex "
        "merging into a nearby free edge (see tyssue.behaviors.sheet.t3).",
        FutureWarning,
        stacklevel=stacklevel,
    )
    return with_t3


def all_rearangements(eptm, with_t2=True, with_t3=None):
    """Performs rearangements (T2/HI first) until
    there are none left or MAX_ITER is reached.

    `with_t3` is a deprecated alias of `with_t2`.
    """
    with_t2 = _t2_flag(with_t2, with_t3)
    for i in count():
        if i == MAX_ITER:
            return 2
        retcode = single_rearangement(eptm, with_t2=with_t2)
        if retcode:  # No transition found
            return retcode


def single_rearangement(eptm, with_t2=True, with_t3=None):
    """Performs a single rearangement (if any) on epithelium `eptm`.

    If `with_t2` is True and there are removeable faces, will perform
    a type 2 or HI transition on one of those faces. If no such
    transition should occur, will perform a type 1 or IH transition on one
    of the edges.

    `with_t3` is a deprecated alias of `with_t2`.
    """
    with_t2 = _t2_flag(with_t2, with_t3)
    edges, faces = find_rearangements(eptm)

    if len(edges):
        for i in count():
            if i == MAX_ITER:
                return 3
            if isinstance(eptm, Sheet):
                retcode = type1_transition(eptm, np.random.choice(edges))
            else:
                retcode = IH_transition(eptm, np.random.choice(edges))
            if not retcode:
                return 0

    elif len(faces) and with_t2:
        for i in count():
            if i == MAX_ITER:
                return 3
            if isinstance(eptm, Sheet):
                retcode = remove_face(eptm, np.random.choice(faces))
            else:
                retcode = HI_transition(eptm, np.random.choice(faces))
            if not retcode:
                return 0
    return 1


def auto_t1(fun):
    @wraps(fun)
    def with_rearange(*args, **kwargs):
        eptm, geom = args[:2]
        logger.debug("checking for t1s")
        res = fun(*args, **kwargs)
        shorts = find_IHs(eptm)
        if not len(shorts):
            return res
        i = 0
        while len(shorts):
            # shorter_edge = shorts[0]
            shorter_edge = np.random.choice(shorts)
            logger.info("transition on  edge %i", shorter_edge)
            if isinstance(eptm, Sheet):
                type1_transition(eptm, shorter_edge)
            else:
                IH_transition(eptm, shorter_edge)
            eptm.reset_index()
            eptm.reset_topo()
            geom.update_all(eptm)
            shorts = find_IHs(eptm)
            if len(shorts) and shorts[0] == shorter_edge:
                # IH transition did not work, skipping
                logger.info("removing shorter edge %i from edge list", shorter_edge)
                shorts = shorts[1:]
            i += 1
            if i > MAX_ITER:
                break
        if eptm.position_buffer is not None:
            logger.info("out T1 changed buffer")
            eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        logger.info("performed %i T1", i)
        return res

    return with_rearange


def auto_t2(fun):
    @wraps(fun)
    def with_rearange(*args, **kwargs):
        eptm, geom = args[:2]
        logger.debug("checking for t2s")
        res = fun(*args, **kwargs)
        tri_faces = find_HIs(eptm)
        if not len(tri_faces):
            return res
        i = 0
        while len(tri_faces):
            smaller_face = tri_faces[0]
            logger.debug("Performing t2 on face %d", smaller_face)
            if isinstance(eptm, Sheet):
                remove_face(eptm, smaller_face)
            else:
                HI_transition(eptm, smaller_face)
            eptm.reset_index()
            eptm.reset_topo()
            geom.update_all(eptm)
            tri_faces = find_HIs(eptm)
            i += 1
            if i > MAX_ITER:
                break
        if eptm.position_buffer is not None:
            logger.info("out T2 changed buffer")
            eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        logger.info("performed %i T2", i)
        return res

    return with_rearange


def auto_t3(fun):
    """Deprecated alias of :func:`auto_t2` (removal of small triangular faces)."""
    warnings.warn(
        "`auto_t3` is deprecated, use `auto_t2`: the removal of small triangular "
        "faces is a type 2 transition.",
        FutureWarning,
        stacklevel=2,
    )
    return auto_t2(fun)
