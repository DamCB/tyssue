"""
Small event module
=======================


"""
from ...geometry.bulk_geometry import MonolayerGeometry
from ...topology.monolayer_topology import cell_division
from ...utils.decorators import cell_lookup
from .actions import contract, grow

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


default_division_spec = {
    "cell_id": -1,
    "cell": -1,
    "growth_rate": 0.1,
    "critical_vol": 2.0,
    "orientation": "vertical",
    "geom": MonolayerGeometry,
}


@cell_lookup
def division(monolayer, manager, **kwargs):
    """Cell division happens through cell growth up to a critical volume,
    followed by actual division of the cell.

    Monolayer version of :func:`tyssue.behaviors.sheet.basic_events.division`:
    the cell's ``prefered_vol`` grows until its volume reaches `critical_vol`.

    Parameters
    ----------
    monolayer : a :class:`Monolayer` (or bulk :class:`Epithelium`) object
    manager : an :class:`EventManager` instance
    cell_id : int
        index of the mother cell
    growth_rate : float, default 0.1
        rate of increase of the prefered volume (see
        :func:`tyssue.behaviors.monolayer.actions.grow`)
    critical_vol : float, default 2.
        volume, in units of the cell's ``prefered_vol``, at which the cell
        stops growing and divides
    orientation : {"vertical", "horizontal", "apical"}, default "vertical"
        orientation of the division plane, passed to
        :func:`tyssue.topology.monolayer_topology.cell_division`
    geom : a geometry class, default
        :class:`~tyssue.geometry.bulk_geometry.MonolayerGeometry`
        used to refresh the geometry after division

    """
    division_spec = default_division_spec.copy()
    division_spec.update(**kwargs)

    cell = division_spec["cell"]

    critical_vol = (
        division_spec["critical_vol"] * monolayer.specs["cell"]["prefered_vol"]
    )

    if monolayer.cell_df.loc[cell, "vol"] < critical_vol:
        grow(monolayer, cell, division_spec["growth_rate"])
        manager.append(division, **division_spec)
    else:
        daughter = cell_division(
            monolayer, cell, orientation=division_spec["orientation"]
        )
        monolayer.cell_df.loc[daughter, "id"] = monolayer.cell_df.id.max() + 1
        division_spec["geom"].update_all(monolayer)
