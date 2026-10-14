"""
Small event module
=======================


"""
import logging

from ...collisions.intersections import find_intersections
from ...geometry.sheet_geometry import SheetGeometry
from ...topology.sheet_topology import cell_division
from ...utils.decorators import face_lookup
from .actions import (
    decrease,
    detach_vertices,
    exchange,
    increase,
    increase_linear_tension,
    merge_vertices,
    remove,
    t3_transitions,
)

logger = logging.getLogger(__name__)


def reconnect(sheet, manager, **kwargs):
    """Performs reconnections (vertex merging / splitting) following Finegan et al. 2019

    kwargs overwrite their corresponding `sheet.settings` entries

    Keyword Arguments
    -----------------
    threshold_length : the threshold length at which vertex merging is performed
    p_4 : the probability per unit time to perform a detachement from a rank 4 vertex
    p_5p : the probability per unit time to perform a detachement from a rank 5
        or more vertex


    See Also
    --------

    **The tricellular vertex-specific adhesion molecule Sidekick
    facilitates polarised cell intercalation during Drosophila axis
    extension** _Tara M Finegan, Nathan Hervieux, Alexander
    Nestor-Bergmann, Alexander G. Fletcher, Guy B Blanchard, Benedicte
    Sanson_ bioRxiv 704932; doi: https://doi.org/10.1101/704932

    """
    sheet.settings.update(kwargs)
    nv = sheet.Nv
    merge_vertices(sheet)
    if nv != sheet.Nv:
        logger.info(f"Merged {nv - sheet.Nv+1} vertices")
    nv = sheet.Nv
    retval = detach_vertices(sheet)
    if retval:
        logger.info("Failed to detach, skipping")

    if nv != sheet.Nv:
        logger.info(f"Detached {sheet.Nv - nv} vertices")

    manager.append(reconnect, **kwargs)


def t3(sheet, manager, **kwargs):
    """Performs type 3 transitions: border vertices that come closer than
    `t3_threshold` to a free edge of another cell are merged into that edge
    (Fletcher et al. 2013).

    kwargs overwrite their corresponding `sheet.settings` entries. Enable with
    `EulerSolver(..., with_t3=True)` or append it once to the manager; it
    re-appends itself every step.

    Keyword Arguments
    -----------------
    t3_threshold : float, default 1e-2
        the vertex to edge distance at which the merge happens
    t3_skin : float, default 10 * t3_threshold
        margin of the candidate list

    Note
    ----
    :func:`reconnect` can detach the rank 4 border vertices T3 creates. Set
    `sheet.settings["detach_boundary"] = False` to keep them.

    See Also
    --------
    :func:`tyssue.behaviors.sheet.actions.t3_transitions`
    :func:`tyssue.topology.sheet_topology.t3_transition`

    **Implementing vertex dynamics models of cell populations in biology within
    a consistent computational framework** _Alexander G Fletcher, James M
    Osborne, Philip K Maini, David J Gavaghan_ Progress in Biophysics and
    Molecular Biology 113 (2013)
    """
    sheet.settings.update(kwargs)
    n_t3 = t3_transitions(sheet)
    if n_t3 > 0:
        logger.info(f"Performed {n_t3} T3 transitions")

    manager.append(t3, **kwargs)


default_intersection_spec = {
    "every": 1,
    "strict": False,
    "trigger": None,
    "on_detect": None,
    "raise_on_detect": False,
    "max_span_rad": None,
    "history": None,
}


def check_intersections(sheet, manager, **kwargs):
    """Checks for folded or overlapping faces.

    Append it once to the manager; it re-appends itself every step::

        manager.append(check_intersections, raise_on_detect=True)

    It only reports and never modifies the mesh. The latest report is stored in
    ``sheet.settings["intersections"]``. Other kwargs overwrite their
    corresponding ``sheet.settings`` entries.

    Keyword Arguments
    -----------------
    every : int, default 1
        run the check every ``every``-th step
    strict : bool, default False
        also run the slower pairwise face-overlap test
    trigger : {None, "folded"}, default None
        ``"folded"`` only looks for contained vertices around self-crossing
        faces; faster, but misses convex cells sliding through each other
    raise_on_detect : bool, default False
        raise ``IntersectionError`` on the first defect
    on_detect : callable, optional
        called as ``on_detect(sheet, report)`` whenever a defect is found
    max_span_rad : float, optional
        faces more curved than this (default 30 degrees) are reported as
        *undecidable* instead of folded
    history : list, optional
        ``(sheet.settings.get("time"), report)`` is appended to it each step

    See Also
    --------
    :mod:`tyssue.collisions.intersections` : the tests and why they are cheap
    """
    spec = default_intersection_spec.copy()
    spec.update(kwargs)
    sheet.settings.update(
        {k: v for k, v in kwargs.items() if k not in default_intersection_spec}
    )

    step = sheet.settings.get("_intersection_step", 0)
    sheet.settings["_intersection_step"] = step + 1
    if spec["every"] > 1 and step % spec["every"]:
        manager.append(check_intersections, **kwargs)
        return

    detect_kwargs = {"strict": spec["strict"], "trigger": spec["trigger"]}
    if spec["max_span_rad"] is not None:
        detect_kwargs["max_span_rad"] = spec["max_span_rad"]

    # the ring matrix is reused until the topology changes
    report = find_intersections(
        sheet, cache=sheet.settings.get("_intersection_cache"), **detect_kwargs
    )
    sheet.settings["_intersection_cache"] = report.cache
    sheet.settings["intersections"] = report

    if spec["history"] is not None:
        spec["history"].append((sheet.settings.get("time"), report))

    if not report.clean:
        # only log when the counts change
        tally = (len(report.folded), len(report.contained), len(report.overlaps),
                 len(report.open_rings), len(report.undecidable))
        if tally != sheet.settings.get("_intersection_tally"):
            logger.warning(
                "intersections: %d folded, %d contained, %d overlapping, "
                "%d open rings, %d undecidable", *tally
            )
        sheet.settings["_intersection_tally"] = tally
        if spec["on_detect"] is not None:
            spec["on_detect"](sheet, report)
        if spec["raise_on_detect"]:
            raise IntersectionError(report)

    manager.append(check_intersections, **kwargs)


class IntersectionError(RuntimeError):
    """Raised by :func:`check_intersections` when ``raise_on_detect`` is set."""

    def __init__(self, report):
        self.report = report
        super().__init__(
            f"tissue is no longer an embedding: {len(report.folded)} self-intersecting "
            f"face(s) {list(report.folded)[:6]}, {len(report.contained)} contained "
            f"vertex/vertices, {len(report.overlaps)} overlapping pair(s), "
            f"{len(report.open_rings)} open ring(s)"
        )


default_division_spec = {
    "face_id": -1,
    "face": -1,
    "growth_rate": 0.1,
    "critical_vol": 2.0,
    "geom": SheetGeometry,
}


@face_lookup
def division(sheet, manager, **kwargs):
    """Cell division happens through cell growth up to a critical volume,
    followed by actual division of the face.

    Parameters
    ----------
    sheet : a `Sheet` object
    manager : an `EventManager` instance
    face_id : int,
      index of the mother face
    growth_rate : float, default 0.1
      rate of increase of the prefered volume
    critical_vol : float, default 2.
      volume at which the cells stops to grow and devides

    """
    division_spec = default_division_spec
    division_spec.update(**kwargs)

    face = division_spec["face"]

    division_spec["critical_vol"] *= sheet.specs["face"]["prefered_vol"]

    if sheet.face_df.loc[face, "vol"] < division_spec["critical_vol"]:
        increase(
            sheet, "face", face, division_spec["growth_rate"], "prefered_vol", True
        )
        manager.append(division, **division_spec)
    else:
        daughter = cell_division(sheet, face, division_spec["geom"])
        sheet.face_df.loc[daughter, "id"] = sheet.face_df.id.max() + 1

        sheet.face_df.loc[daughter, "unique_id"] = sheet.specs['face']['unique_id_max']+1
        sheet.specs['face']['unique_id_max'] += 1

        sheet.lineage.add_node(str(sheet.face_df.loc[daughter]['unique_id']),
                               color='grey')
        sheet.lineage.add_edge(str(sheet.face_df.loc[face]['unique_id']),
                               str(sheet.face_df.loc[daughter]['unique_id']))


default_contraction_spec = {
    "face_id": -1,
    "face": -1,
    "contractile_increase": 1.0,
    "critical_area": 1e-2,
    "max_contractility": 10,
    "multiply": False,
    "contraction_column": "contractility",
    "unique": True,
}


@face_lookup
def contraction(sheet, manager, **kwargs):
    """Single step contraction event."""
    contraction_spec = default_contraction_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]

    if (sheet.face_df.loc[face, "area"] < contraction_spec["critical_area"]) or (
        sheet.face_df.loc[face, contraction_spec["contraction_column"]]
        > contraction_spec["max_contractility"]
    ):
        return
    increase(
        sheet,
        "face",
        face,
        contraction_spec["contractile_increase"],
        contraction_spec["contraction_column"],
        contraction_spec["multiply"],
    )


default_type1_transition_spec = {
    "face_id": -1,
    "face": -1,
    "critical_length": 0.1,
    "geom": SheetGeometry,
}


@face_lookup
def type1_transition(sheet, manager, **kwargs):
    """Custom type 1 transition event that tests if
    the the shorter edge of the face is smaller than
    the critical length.
    """
    type1_transition_spec = default_type1_transition_spec
    type1_transition_spec.update(**kwargs)
    face = type1_transition_spec["face"]

    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    if min(edges["length"]) < type1_transition_spec["critical_length"]:
        exchange(sheet, face, type1_transition_spec["geom"])


default_face_elimination_spec = {"face_id": -1, "face": -1, "geom": SheetGeometry}


@face_lookup
def face_elimination(sheet, manager, **kwargs):
    """Removes the face with if face_id from the sheet."""
    face_elimination_spec = default_face_elimination_spec
    face_elimination_spec.update(**kwargs)
    remove(sheet, face_elimination_spec["face"], face_elimination_spec["geom"])


default_check_tri_face_spec = {"geom": SheetGeometry}


def check_tri_faces(sheet, manager, **kwargs):
    """Three neighbourghs cell elimination
    Add all cells with three neighbourghs in the manager
    to be eliminated at the next time step.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    """
    check_tri_faces_spec = default_check_tri_face_spec
    check_tri_faces_spec.update(**kwargs)

    tri_faces = sheet.face_df[(sheet.face_df["num_sides"] < 4)].id
    manager.extend(
        [
            (face_elimination, {"face_id": f, "geom": check_tri_faces_spec["geom"]})
            for f in tri_faces
        ]
    )


default_contraction_line_tension_spec = {
    "face_id": -1,
    "face": -1,
    "shrink_rate": 1.05,
    "contractile_increase": 1.0,
    "critical_area": 1e-2,
    "max_contractility": 10,
    "multiply": True,
    "contraction_column": "line_tension",
    "unique": True,
}


@face_lookup
def contraction_line_tension(sheet, manager, **kwargs):
    """
    Single step contraction event
    """
    contraction_spec = default_contraction_line_tension_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]

    if sheet.face_df.loc[face, "area"] < contraction_spec["critical_area"]:
        return

    # reduce prefered_area
    decrease(
        sheet,
        "face",
        face,
        contraction_spec["shrink_rate"],
        col="prefered_area",
        divide=True,
        bound=contraction_spec["critical_area"] / 2,
    )

    increase_linear_tension(
        sheet,
        face,
        contraction_spec["contractile_increase"],
        multiply=contraction_spec["multiply"],
        isotropic=True,
        limit=100,
    )
