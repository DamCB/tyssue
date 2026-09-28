"""
Tissue closure / fusion events
==============================

Behaviors that close an open monolayer (e.g. the rim of a lifting tissue or
two fronts coming into contact) into a continuous epithelium with a lumen.

These events bundle the topology operations needed for closure:

- the local rearrangements (T1 / IH and T3 / HI transitions) that progressively
  squeeze cells away from the boundary -- the "zipper" mechanism, and
- the lateral-face fusion operation
  (:func:`tyssue.topology.bulk_topology.fuse_lateral_faces`) that welds two
  apposed free lateral walls into one internal interface. It is collision-
  triggered (nucleation on distance + facing angle, propagation on the fold
  angle of an existing seam) and polarity-typed (apical-to-apical, basal-to-
  basal), so no coincidence between whole faces is required and a basal face can
  never fuse to a lateral one. When two fusions trap an unequal number of cells
  between them (a wide wall apposed to two narrower ones -- a "2-vs-1 pocket"),
  the wide wall is split (:func:`tyssue.topology.bulk_topology.find_fusion_splits`)
  so the pocket can finish zipping shut.

The events are deliberately kept out of the solver's ``auto_reconnect`` path:
add them to an :class:`~tyssue.behaviors.event_manager.EventManager` manually,
e.g. ``manager.append(lumen_closure, **spec)``, while experimenting.
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
    """Fuses every apposed pair of lateral faces, then re-schedules itself.

    This is the pure fusion step: it performs no rearrangement, only the
    lateral-face fusion of fronts that have come into contact (nucleation where
    two fronts first meet, propagation along an existing seam). It re-appends
    itself to the manager so it keeps watching for new contacts every step.

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

    Each step this event

    1. performs a single boundary rearrangement (an IH/T1 or, if
       ``with_t2``, an HI/T2 transition) -- the zipper that brings border
       cells together, then
    2. fuses every apposed lateral-face pair between the approaching fronts.

    It re-appends itself, so adding it once to the manager keeps the closure
    process running for the whole simulation.

    Keyword Arguments
    -----------------
    d_max, theta_face, theta_fold, margin, segment :
        passed through to the lateral-face fusion topology functions.
    with_rearrangements : bool, default True
        whether to perform the IH/HI rearrangement step.
    with_t2 : bool, default True
        whether HI / T2 transitions are allowed in the rearrangement step.
        ``with_t3`` is accepted as a deprecated alias.
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
