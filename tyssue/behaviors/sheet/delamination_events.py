"""
Mesoderm invagination event module
=======================


"""

import random
import numpy as np

from ...geometry.sheet_geometry import SheetGeometry
from ...utils.decorators import face_lookup
from .actions import relax, contract, ab_pull, exchange, remove
from .basic_events import contraction


def delamination(
    sheet,
    manager,
    face_id,
    contract_rate=2,
    critical_area=1e-2,
    radial_tension=1.0,
    nb_iteration=0,
    nb_iteration_max=20,
    contract_neighbors=True,
    critical_area_neighbors=10,
    contract_span=2,
    geom=SheetGeometry,
    contraction_column="contractility",
):
    """Delamination process
    This function corresponds to the process called "apical constriction"
    in the manuscript
    The cell undergoing delamination first contracts its apical
    area until it reaches a critical area, at which point it starts
    undergoing rearangements with its neighbors, performing
    successive type 1 transitions until the face has only 3 sides,
    when it disepears.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    face_id : int
       the Id of the face undergoing delamination.
    contract_rate : float, default 2
       rate of increase of the face contractility.
    critical_area : float, default 1e-2
       face's area under which the cell starts loosing sides.
    radial_tension : float, default 1.
       tension applied on the face vertices along the
       apical-basal axis.
    nb_iteration : int, default 0
       number of extra iterations where the apical-basal force is applied
       between each type 1 transition
    contract_neighbors : bool, default `False`
       if True, the face contraction triggers contraction of the neighbor
       faces.
    contract_span : int, default 2
       rank of neighbors contracting if contract_neighbor is True. Contraction
       rate for the neighbors is equal to `contract_rate` devided by
       the rank.
    """
    settings = {
        "contract_rate": contract_rate,
        "critical_area": critical_area,
        "radial_tension": radial_tension,
        "nb_iteration": nb_iteration,
        "nb_iteration_max": nb_iteration_max,
        "contract_neighbors": contract_neighbors,
        "critical_area_neighbors": critical_area_neighbors,
        "contract_span": contract_span,
        "geom": geom,
        "contraction_column": contraction_column,
    }

    face = sheet.idx_lookup(face_id, "face")
    if face is None:
        return

    if sheet.face_df.loc[face, "is_relaxation"]:
        relax(sheet, face, contract_rate, contraction_column)

    face_area = sheet.face_df.loc[face, "area"]
    num_sides = sheet.face_df.loc[face, "num_sides"]

    if face_area > critical_area:
        contract(sheet, face, contract_rate, True, contraction_column)

        if contract_neighbors & (face_area < critical_area_neighbors):
            neighbors = sheet.get_neighborhood(face, contract_span).dropna()
            neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values

            manager.extend(
                [
                    (
                        contraction,
                        neighbor["id"],
                        (
                            contract_rate ** (1 / 2 ** neighbor["order"]),
                            critical_area,
                            50,
                            contraction_column,
                        ),
                    )  # TODO: check this
                    for _, neighbor in neighbors.iterrows()
                ]
            )
        done = False

    elif face_area <= critical_area:
        if nb_iteration < nb_iteration_max:
            settings["nb_iteration"] = nb_iteration + 1
            ab_pull(sheet, face, radial_tension, True)
            done = False
        elif nb_iteration >= nb_iteration_max:
            done = True
    if not done:
        manager.append(delamination, face_id=face_id, **settings)


default_constriction_spec = {
    "face_id": -1,
    "face": -1,
    "contract_rate": 2,
    "critical_area": 1e-2,
    "radial_tension": 1.0,
    "nb_iteration": 0,
    "nb_iteration_max": 20,
    "contract_neighbors": True,
    "critical_area_neighbors": 10,
    "contract_span": 2,
    "basal_contract_rate": 1.001,
    "current_traction": 0,
    "max_traction": 30,
    "geom": SheetGeometry,
    "contraction_column": "contractility",
}

default_constriction_spec = {
    "face_id": -1,
    "face": -1,
    "contract_rate": 2,
    "critical_area": 1e-2,
    "radial_tension": 1.0,
    "nb_iteration": 0,
    "nb_iteration_max": 20,
    "contract_neighbors": True,
    "critical_area_neighbors": 10,
    "contract_span": 2,
    "basal_contract_rate": 1.001,
    "current_traction": 0,
    "max_traction": 30,
    "geom": SheetGeometry,
    "contraction_column": "contractility",
}


@face_lookup
def constriction(sheet, manager, **kwargs):
    """Constriction process
    This function corresponds to the process called "apical constriction"
    in the manuscript
    The cell undergoing delamination first contracts its apical
    area until it reaches a critical area. A probability
    dependent to the apical area allow an apico-basal
    traction of the cell. The cell can pull during max_traction
    time step, not necessarily consecutively.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    face_id : int
       the Id of the face undergoing delamination.
    contract_rate : float, default 2
       rate of increase of the face contractility.
    critical_area : float, default 1e-2
       face's area under which the cell starts loosing sides.
    radial_tension : float, default 1.
       tension applied on the face vertices along the
       apical-basal axis.
    nb_iteration : int, default 0
       number of extra iterations where the apical-basal force is applied
       between each type 1 transition
    contract_neighbors : bool, default `False`
       if True, the face contraction triggers contraction of the neighbor
       faces.
    contract_span : int, default 2
       rank of neighbors contracting if contract_neighbor is True. Contraction
       rate for the neighbors is equal to `contract_rate` devided by
       the rank.
    """
    constriction_spec = default_constriction_spec
    constriction_spec.update(**kwargs)

    # initialiser une variable face
    # aller chercher la valeur dans le dictionnaire à chaque fois ?
    face = constriction_spec["face"]
    contract_rate = constriction_spec["contract_rate"]
    current_traction = constriction_spec["current_traction"]

    if "is_relaxation" in sheet.face_df.columns:
        if sheet.face_df.loc[face, "is_relaxation"]:
            relax(sheet, face, contract_rate, constriction_spec["contraction_column"])

    if sheet.face_df.loc[face, "is_mesoderm"]:
        face_area = sheet.face_df.loc[face, "area"]

        if face_area > constriction_spec["critical_area"]:
            contract(
                sheet,
                face,
                contract_rate,
                True,
                constriction_spec["contraction_column"],
            )
            # if sheet.face_df.loc[face, 'prefered_area'] > 6:
            #    sheet.face_df.loc[face, 'prefered_area'] -= 0.5
            # increase_linear_tension(sheet, face, contract_rate)

            if (constriction_spec["contract_neighbors"]) & (
                face_area < constriction_spec["critical_area_neighbors"]
            ):
                neighbors = sheet.get_neighborhood(
                    face, constriction_spec["contract_span"]
                ).dropna()
                neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values

                # remove cell which are not mesoderm
                ectodermal_cell = sheet.face_df.loc[neighbors.face][
                    ~sheet.face_df.loc[neighbors.face, "is_mesoderm"]
                ].id.values

                neighbors = neighbors.drop(
                    neighbors[neighbors.id.isin(ectodermal_cell)].index
                )

                manager.extend(
                    [
                        (
                            contraction,
                            _neighbor_contractile_increase(
                                neighbor, contract_rate, constriction_spec
                            ),
                        )  # TODO: check this
                        for _, neighbor in neighbors.iterrows()
                    ]
                )

        proba_tension = np.exp(-face_area / constriction_spec["critical_area"])
        aleatory_number = random.uniform(0, 1)
        if constriction_spec["current_traction"] < constriction_spec["max_traction"]:
            if aleatory_number < proba_tension:
                current_traction = current_traction + 1
                ab_pull(sheet, face, constriction_spec["radial_tension"], True)
                constriction_spec.update(
                    {
                        "contract_rate": contract_rate,
                        "current_traction": current_traction,
                    }
                )

    manager.append(constriction, **constriction_spec)


def _neighbor_contractile_increase(neighbor, contract_rate, constriction_spec):

    increase = (
        -(contract_rate - constriction_spec["basal_contract_rate"])
        / constriction_spec["contract_span"]
    ) * neighbor["order"] + contract_rate

    specs = {
            "face_id": neighbor["id"],
            "contractile_increase": increase,
            "critical_area": constriction_spec["critical_area"],
            "max_contractility": 50,
            "contraction_column": constriction_spec["contraction_column"],
            "multiple": True,
        }

    return specs
