"""
Tissue closure / fusion events
==============================

Behaviors that close an open monolayer (e.g. two fronts coming into contact)
into a continuous epithelium with a lumen, by combining:

- rearrangements (T1 / IH and T2 / HI transitions) at the boundary, and
- fusion of apposed free lateral faces into one internal interface
  (:func:`tyssue.topology.bulk_topology.fuse_lateral_faces`), apical vertices
  to apical and basal to basal.

Add them to an :class:`~tyssue.behaviors.event_manager.EventManager`, e.g.
``manager.append(lumen_closure, **spec)``.
"""

import logging

from ...topology import _t2_flag, all_lateral_fusions, single_rearangement

logger = logging.getLogger(__name__)


default_fuse_spec = {
    "d_max": None,
    "theta_face": 45.0,
    "theta_fold": 20.0,
    "margin": 0.5,
    "segment": "lateral",
}


def fuse(eptm, manager, **kwargs):
    """Fuses every apposed pair of free lateral faces, without
    rearrangements. Re-appends itself every step.

    Keyword Arguments
    -----------------
    d_max : float or None
        contact distance from a vertex to the opposing face plane for
        nucleation. Defaults to ``eptm.settings["fusion_distance"]`` (or the
        mean free lateral-edge length).
    theta_face : float, default 45.0
        maximum deviation, in degrees, from anti-parallel alignment of the two
        walls' normals for nucleation.
    theta_fold : float, default 20.0
        fold-angle threshold, in degrees, below which two edge-sharing flaps are
        considered closed onto each other (propagation).
    margin : float, default 0.5
        lateral-containment tolerance for nucleation (fraction of face radius).
    segment : str, default "lateral"
        the segment of free faces eligible to fuse.

    See Also
    --------
    tyssue.topology.bulk_topology.find_fusion_nucleations
    tyssue.topology.bulk_topology.find_fusion_propagations
    tyssue.topology.bulk_topology.fuse_lateral_faces
    """
    spec = default_fuse_spec.copy()
    spec.update(kwargs)

    n_fused = all_lateral_fusions(
        eptm,
        d_max=spec["d_max"],
        theta_face=spec["theta_face"],
        theta_fold=spec["theta_fold"],
        margin=spec["margin"],
        segment=spec["segment"],
    )
    if n_fused:
        logger.info("fused %d lateral face pair(s)", n_fused)

    manager.append(fuse, **kwargs)


default_lumen_closure_spec = {
    "d_max": None,
    "theta_face": 45.0,
    "theta_fold": 20.0,
    "margin": 0.5,
    "segment": "lateral",
    "with_rearrangements": True,
    "with_t2": True,
}


def lumen_closure(eptm, manager, **kwargs):
    """Drives a monolayer towards closure: rearrangements then lateral fusion.

    Each step it performs a single rearrangement (IH/T1, or HI/T2 if
    ``with_t2``), then fuses every apposed pair of lateral faces. Re-appends
    itself every step.

    Keyword Arguments
    -----------------
    d_max, theta_face, theta_fold, margin, segment :
        passed through to the lateral-face fusion topology functions.
    with_rearrangements : bool, default True
        whether to perform the IH/HI rearrangement step.
    with_t2 : bool, default True
        whether HI / T2 transitions are allowed; ``with_t3`` is a deprecated
        alias.
    """
    spec = default_lumen_closure_spec.copy()
    spec.update(kwargs)
    spec["with_t2"] = _t2_flag(spec["with_t2"], spec.pop("with_t3", None))

    if spec["with_rearrangements"]:
        single_rearangement(eptm, with_t2=spec["with_t2"])

    n_fused = all_lateral_fusions(
        eptm,
        d_max=spec["d_max"],
        theta_face=spec["theta_face"],
        theta_fold=spec["theta_fold"],
        margin=spec["margin"],
        segment=spec["segment"],
    )
    if n_fused:
        logger.info("lumen_closure fused %d lateral face pair(s)", n_fused)

    manager.append(lumen_closure, **kwargs)
