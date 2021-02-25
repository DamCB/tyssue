"""
Apoptosis event module
=======================


"""

from ...utils.decorators import face_lookup
from ...geometry.sheet_geometry import SheetGeometry

from .actions import decrease, ab_pull, exchange, remove
from .basic_events import contraction

default_apoptosis_spec = {
    "face_id": -1,
    "face": -1,
    "shrink_rate": 1.1,
    "critical_area": 1e-2,
    "radial_tension": 0.1,
    "contractile_increase": 0.1,
    "contract_span": 2,
    "geom": SheetGeometry,
}


@face_lookup
def apoptosis(sheet, manager, **kwargs):
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

    apoptosis_spec = default_apoptosis_spec
    apoptosis_spec.update(**kwargs)
    face = apoptosis_spec["face"]

    if sheet.face_df.loc[face, "area"] > apoptosis_spec["critical_area"]:
        # Shrink and pull
        decrease(
            sheet, "face", face, apoptosis_spec["shrink_rate"], "prefered_vol", True
        )
        ab_pull(sheet, face, apoptosis_spec["radial_tension"])

        # contract neighbors
        neighbors = sheet.get_neighborhood(
            face, apoptosis_spec["contract_span"]
        ).dropna()
        neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values
        manager.extend(
            [
                (
                    contraction,
                    {
                        "face_id": neighbor["id"],
                        "contractile_increase": (
                            apoptosis_spec["contractile_increase"] / neighbor["order"],
                        ),
                    },
                )
                for _, neighbor in neighbors.iterrows()
            ]
        )
        done = False
    else:
        if sheet.face_df.loc[face, "num_sides"] > 3:
            exchange(sheet, face, apoptosis_spec["geom"])
            done = False
        else:
            remove(sheet, face, apoptosis_spec["geom"])
            done = True
    if not done:
        manager.append(apoptosis, **apoptosis_spec)
