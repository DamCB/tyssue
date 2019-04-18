import numpy as np
from functools import wraps

from ..core.monolayer import Monolayer
from ..core.sheet import Sheet

from .base_topology import *
from .sheet_topology import type1_transition, remove_face
from .bulk_topology import HI_transition, IH_transition, find_HIs, find_IHs


MAX_ITER = 10


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
            print("out T1 changed buffer")
            eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        logger.info("performed %i T1", i)
        print("T1")
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
            print("out T3 changed buffer")
            eptm.position_buffer = eptm.vert_df[eptm.coords].copy()
        print("T3")
        logger.info("performed %i T3", i)
        return res

    return with_rearange


def _get_shorter_edges(eptm, discarded, l_th):
    shorts = eptm.edge_df[eptm.edge_df.length < 5 * l_th].sort_values("length")
    shorts = (
        shorts[["srce", "trgt"]].apply(frozenset, axis=1).drop_duplicates().index.values
    )
    return shorts
