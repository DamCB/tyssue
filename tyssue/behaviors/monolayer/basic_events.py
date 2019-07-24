"""
Small event module
=======================


"""
from ...utils.decorators import face_lookup, cell_lookup
from ...geometry.sheet_geometry import SheetGeometry

from .actions import grow, contract
from ..sheet.basic_events import reconnect

default_contraction_spec = {
    "cell_id": -1,
    "cell": -1,
    "side": "all",
    "contractile_increase": 1.0,
    "critical_area": 1e-2,
    "max_contractility": 10,
    "multiple": False,
    "contraction_column": "contractility",
    "unique": True,
}
# Side can be "apical", "basal", "lateral", "all"


@cell_lookup
def contraction(monolayer, manager, **kwargs):
    """
    Single step contraction event
    """
    contraction_spec = default_contraction_spec
    contraction_spec.update(**kwargs)

    cell = contraction_spec["cell"]
    list_face_in_cell = monolayer.get_orbits("cell", "face")

    # Pick face id in function of chosen side
    faces_id = (
        monolayer.face_df[
            (monolayer.face_df.index.isin(list_face_in_cell[cell]))
            & (monolayer.face_df.segment == contraction_spec["side"])
        ]
        .index[0]
        .values
    )

    for f in faces_id:
        if (monolayer.face_df.loc[f, "area"] < contraction_spec["critical_area"]) or (
            monolayer.face_df.loc[f, contraction_spec["contraction_column"]]
            > contraction_spec["max_contractility"]
        ):
            return

        contract(
            monolayer,
            f,
            contraction_spec["contractile_increase"],
            contraction_spec["multiple"],
            contraction_spec["contraction_column"],
        )
