"""
Small event module
=======================


"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


from ...utils.decorators import face_lookup
from ...geometry.sheet_geometry import SheetGeometry
from ...topology.sheet_topology import cell_division

from .actions import (
    grow,
    contract,
    exchange,
    remove,
    merge_vertices,
    detach_vertices,
    increase,
    decrease,
    increase_linear_tension,
)


def reconnect(sheet, manager, **kwargs):
    """Performs reconnections (vertex merging / splitting) following Finegan et al. 2019

    kwargs overwrite their corresponding `sheet.settings` entries

    Keyword Arguments
    -----------------
    threshold_length : the threshold length at which vertex merging is performed
    p_4 : the probability per unit time to perform a detachement from a rank 4 vertex
    p_5p : the probability per unit time to perform a detachement from a rank 5 or more vertex


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
    try:
        detach_vertices(sheet)
    except ValueError:
        logger.info(f"Failed to detach, skipping")
        pass
    if nv != sheet.Nv:
        logger.info(f"Detached {sheet.Nv - nv} vertices")

    manager.append(reconnect, **kwargs)


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

    print(sheet.face_df.loc[face, "vol"], division_spec["critical_vol"])

    if sheet.face_df.loc[face, "vol"] < division_spec["critical_vol"]:
        grow(sheet, face, division_spec["growth_rate"])
        manager.append(division, **division_spec)
    else:
        daughter = cell_division(sheet, face, division_spec["geom"])
        sheet.face_df.loc[daughter, "id"] = sheet.face_df.id.max() + 1


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
    """Single step contraction event
    """
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
    """Removes the face with if face_id from the sheet
    """
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
