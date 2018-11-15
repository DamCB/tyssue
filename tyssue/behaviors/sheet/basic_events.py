"""
Small event module
=======================


"""
from ..utils.decorators import face_lookup
from ..geometry.sheet_geometry import SheetGeometry
from ..topology.sheet_topology import cell_division

from ..actions import grow, contract, exchange, remove


@face_lookup
def division(
        sheet, manager, *, face_id, growth_rate=0.1, critical_vol=2.0, geom=SheetGeometry):
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
    critical_vol *= sheet.specs["face"]["prefered_vol"]
    print(sheet.face_df.loc[face_id, "vol"], critical_vol)
    if sheet.face_df.loc[face_id, "vol"] < critical_vol:
        grow(sheet, face_id, growth_rate)
        manager.append(division, face_id, args=(
            growth_rate, critical_vol, geom))
    else:
        daughter = cell_division(sheet, face_id, geom)
        sheet.face_df.loc[daughter, "id"] = sheet.face_df.id.max() + 1


@face_lookup
def contraction(
        sheet,
        manager, *,
        face_id,
        contractile_increase=1.0,
        critical_area=1e-2,
        max_contractility=10):
    """Single step contraction event
    """
    if (sheet.face_df.loc[face_id, "area"] < critical_area) or (
        sheet.face_df.loc[face_id, "contractility"] > max_contractility
    ):
        return
    contract(sheet, face_id, contractile_increase)


@face_lookup
def neighbors_contraction(
    sheet,
    manager, *,
    face_id=-1,
    contractile_increase=1.0,
    critical_area=1e-2,
    max_contractility=10,
    contraction_column="contractility"
):
    """Custom single step contraction event.
    """
    if (sheet.face_df.loc[face, "area"] < critical_area) or (
        sheet.face_df.loc[face, contraction_column] > max_contractility
    ):
        return
    contract(sheet, face_id, contractile_increase, True)


@face_lookup
def type1_transition(sheet, manager, *, face_id=-1, critical_length=0.3, geom=SheetGeometry):
    """Custom type 1 transition event that tests if
    the the shorter edge of the face is smaller than
    the critical length.
    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face_id]
    if min(edges["length"]) < critical_length:
        exchange(sheet, face_id, geom)


@face_lookup
def face_elimination(sheet, manager, *, face_id=-1, geom=SheetGeometry):
    """Removes the face with if face_id from the sheet
    """
    remove(sheet, face_id, geom)


def check_tri_faces(sheet, manager):
    """Three neighbourghs cell elimination
    Add all cells with three neighbourghs in the manager
    to be eliminated at the next time step.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    """

    tri_faces = sheet.face_df[
        (sheet.face_df["num_sides"] < 4) & (
            sheet.face_df["is_mesoderm"] is False)
    ]["id"]
    manager.extend(
        [
            (face_elimination, f, (), {
             "geom": sheet.settings["delamination"]["geom"]})
            for f in tri_faces
        ]
    )
