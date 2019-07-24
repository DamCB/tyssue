import numpy as np
from functools import wraps
from itertools import count

from ..core.sheet import Sheet

from .base_topology import *
from .sheet_topology import type1_transition, remove_face
from .bulk_topology import (
    HI_transition,
    IH_transition,
    find_HIs,
    find_IHs,
    find_rearangements,
)


MAX_ITER = 10


class TopologyChangeError(ValueError):
    """ Raised when trying to assign values without
    the correct length to an epithelium dataset
    """

    pass


def all_rearangements(eptm, with_t3=True):
    """Performs rearangements (T3/HI first) until
    there are none left or MAX_ITER is reached.
    """
    for i in count():
        if i == MAX_ITER:
            return 2
        retcode = single_rearangement(eptm, with_t3=with_t3)
        if retcode:  # No transition found
            return retcode


def single_rearangement(eptm, with_t3=True):
    """Performs a single rearangement (if any) on epithelium `eptm`.

    If `with_t3` is True and there are removeable faces, will perform
    a type 3 or HI transition on one of those faces. If no such
    transition should occur, will perform a type 1 or IH transition on one
    of the edges.
    """
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

    elif len(faces) and with_t3:
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


def auto_t3(fun):
    @wraps(fun)
    def with_rearange(*args, **kwargs):
        eptm, geom = args[:2]
        logger.debug("checking for t3s")
        res = fun(*args, **kwargs)
        tri_faces = find_HIs(eptm)
        if not len(tri_faces):
            return res
        i = 0
        while len(tri_faces):
            smaller_face = tri_faces[0]
            logger.debug("Performing t3 on face %d", smaller_face)
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
            logger.info("out T3 changed buffer")
            eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        logger.info("performed %i T3", i)
        return res

    return with_rearange
