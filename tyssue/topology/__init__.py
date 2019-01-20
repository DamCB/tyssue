import numpy as np
from functools import wraps

from ..core.monolayer import Monolayer
from ..core.sheet import Sheet

from .base_topology import *
from .sheet_topology import type1_transition, remove_face
from .bulk_topology import HI_transition, IH_transition


def auto_t1(fun):
    @wraps(fun)
    def with_rearange(*args, **kwargs):
        eptm, geom = args[:2]
        logger.debug("checking for t1s")
        l_th = eptm.settings.get("threshold_length", 1e-6)
        res = fun(*args, **kwargs)
        shorts = eptm.edge_df[eptm.edge_df.length < l_th].sort_values("length").index
        np.random.shuffle(shorts.values)
        if not len(shorts):
            return res
        if isinstance(eptm, Sheet):
            for edge in shorts:
                type1_transition(eptm, edge)
        elif isinstance(eptm, Monolayer):
            for edge in shorts:
                IH_transition(eptm, edge)
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)
        # re-execute with updated topology
        res = fun(*args, **kwargs)
        return res

    return with_rearange


def auto_t3(fun):
    @wraps(fun)
    def with_rearange(*args, **kwargs):
        eptm, geom = args[:2]
        logger.debug("checking for t1s")
        a_th = eptm.settings.get("threshold_area", 1e-8)
        res = fun(*args, **kwargs)
        tri_faces = eptm.face_df[
            (eptm.face_df.area < a_th) & (eptm.face_df.num_sides > 3)
        ]

        np.random.shuffle(tri_faces.values)
        if not len(tri_faces):
            return res
        logger.info("performing %i T3", len(tri_faces))
        if isinstance(eptm, Sheet):
            for face in tri_faces:
                remove_face(eptm, face)
        elif isinstance(eptm, Monolayer):
            for face in tri_faces:
                HI_transition(eptm, face)
        eptm.reset_index()
        eptm.reset_topo()
        geom.update_all(eptm)
        # re-execute with updated topology
        res = fun(*args, **kwargs)
        return res

    return with_rearange
