"""
Apoptosis event module
=======================


"""

from ...geometry.sheet_geometry import SheetGeometry

from .actions import shrink, ab_pull, exchange, remove
from .basic_events import contraction


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
            exchange(sheet, face, geom)
            done = False
        else:
            remove(sheet, face, geom)
            done = True
    if not done:
        manager.append(apoptosis, face_id, kwargs=settings)
