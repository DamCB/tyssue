"""
Event management module
=======================


"""

import logging
import pandas as pd
import warnings
import random
from collections import deque
from ..topology.sheet_topology import remove_face, type1_transition, cell_division
from ..geometry.sheet_geometry import SheetGeometry

logger = logging.getLogger(__name__)


def division(
    sheet, manager, face_id, growth_rate=0.1, critical_vol=2.0, geom=SheetGeometry
):
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

    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return
    critical_vol *= sheet.specs["face"]["prefered_vol"]
    print(sheet.face_df.loc[face, "vol"], critical_vol)
    if sheet.face_df.loc[face, "vol"] < critical_vol:
        grow(sheet, face, growth_rate)
        manager.append(division, face_id, args=(growth_rate, critical_vol, geom))
    else:
        daughter = cell_division(sheet, face, geom)
        sheet.face_df.loc[daughter, "id"] = sheet.face_df.id.max() + 1


def apoptosis(
    sheet,
    manager,
    face_id,
    shrink_rate=0.1,
    critical_area=1e-2,
    radial_tension=0.1,
    contractile_increase=0.1,
    contract_span=2,
    geom=SheetGeometry,
):
    """Apoptotic behavior

    While the cell's apical area is bigger than a threshold, the
    cell shrinks, and the contractility of its neighbors is increased.
    once the critical area is reached, the cell is eliminated
    from the apical surface through successive type 1 transition. Once
    only three sides are left, the cell is eliminated from the tissue.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    manager : a :class:`EventManager` object
    face_id : int,
        the id of the apoptotic cell
    shrink_rate : float, default 0.1
        the rate of reduction of the cell's prefered volume
        e.g. the prefered volume is devided by a factor 1+shrink_rate
    critical_area : area at which the face is eliminated from the sheet
    radial_tension : amount of radial tension added at each contraction steps
    contractile_increase : increase in contractility at the cell neighbors
    contract_span : number of neighbors affected by the contracitity increase
    geom : the geometry class used
    """

    settings = {
        "shrink_rate": shrink_rate,
        "critical_area": critical_area,
        "radial_tension": radial_tension,
        "contractile_increase": contractile_increase,
        "contract_span": contract_span,
        "geom": geom,
    }

    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return

    if sheet.face_df.loc[face, "area"] > critical_area:
        # Shrink and pull
        shrink(sheet, face, shrink_rate)
        ab_pull(sheet, face, radial_tension)
        # contract neighbors
        neighbors = sheet.get_neighborhood(face, contract_span).dropna()
        neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values
        manager.extend(
            [
                (
                    contraction,
                    neighbor["id"],
                    (contractile_increase / neighbor["order"],),
                )
                for _, neighbor in neighbors.iterrows()
            ]
        )
        done = False
    else:
        if sheet.face_df.loc[face, "num_sides"] > 3:
            type1_at_shorter(sheet, face, geom)
            done = False
        else:
            type3(sheet, face, geom)
            done = True
    if not done:
        manager.append(apoptosis, face_id, kwargs=settings)


def contraction(
    sheet,
    manager,
    face_id,
    contractile_increase=1.0,
    critical_area=1e-2,
    max_contractility=10,
):
    """Single step contraction event
    """
    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return
    if (sheet.face_df.loc[face, "area"] < critical_area) or (
        sheet.face_df.loc[face, "contractility"] > max_contractility
    ):
        return
    contract(sheet, face, contractile_increase)


def grow(sheet, face, growth_rate):
    """Multiplies the equilibrium volume of face by a
    a factor (1+growth_rate)
    """
    sheet.face_df.loc[face, "prefered_vol"] *= 1 + growth_rate


def shrink(sheet, face, shrink_rate):
    """Devides the equilibrium volume of face face by a
    a factor 1+shrink_rate
    """
    sheet.face_df.loc[face, "prefered_vol"] /= 1 + shrink_rate


def type1_at_shorter(sheet, face, geom, remove_tri_faces=True):
    """
    Execute a type1 transition on the shorter edge of a face.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class
    """
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    shorter = edges.length.idxmin()
    # type1_transition(sheet, shorter, 2 * min(edges.length), remove_tri_faces)
    type1_transition(sheet, shorter, 0.1, remove_tri_faces)
    geom.update_all(sheet)


def type3(sheet, face, geom):
    """Removes the face and updates the geometry

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    face : index of the face
    geom : a Geometry class

    """
    remove_face(sheet, face)
    geom.update_all(sheet)


def contract(sheet, face, contractile_increase, multiple=False):
    """
    Contract the face by increasing the 'contractility' parameter
    by contractile_increase

    Parameters
    ----------
    face : id face

    """
    if multiple:
        sheet.face_df.loc[face, "contractility"] *= contractile_increase
    else:
        new_contractility = contractile_increase
        sheet.face_df.loc[face, "contractility"] += new_contractility


def ab_pull(sheet, face, radial_tension, distributed=False):
    """ Adds radial_tension to the face's vertices radial_tension
    """
    verts = sheet.edge_df[sheet.edge_df["face"] == face]["srce"].unique()
    if distributed:
        radial_tension = radial_tension / len(verts)

    sheet.vert_df.loc[verts, "radial_tension"] += radial_tension
