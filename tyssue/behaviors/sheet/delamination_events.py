"""
Mesoderm invagination event module
=======================


"""

import random
import numpy as np

from ...utils.decorators import face_lookup
from .actions import relax, contract, ab_pull
from .basic_events import contraction


default_constriction_spec = {
    "face_id": -1,
    "face": -1,
    "contract_rate": 2,
    "critical_area": 1e-2,
    "radial_tension": 1.0,
    "contract_neighbors": True,
    "critical_area_neighbors": 10,
    "contract_span": 2,
    "basal_contract_rate": 1.001,
    "current_traction": 0,
    "max_traction": 30,
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
                            _neighbor_contractile_increase(neighbor, constriction_spec),
                        )  # TODO: check this
                        for _, neighbor in neighbors.iterrows()
                    ]
                )

        proba_tension = np.exp(-face_area / constriction_spec["critical_area"])
        aleatory_number = random.uniform(0, 1)
        if current_traction < constriction_spec["max_traction"]:
            if aleatory_number < proba_tension:
                current_traction = current_traction + 1
                ab_pull(sheet, face, constriction_spec["radial_tension"], True)
                constriction_spec.update({"current_traction": current_traction})

    manager.append(constriction, **constriction_spec)


def _neighbor_contractile_increase(neighbor, constriction_spec):

    contract = constriction_spec["contract_rate"]
    basal_contract = constriction_spec["basal_contract_rate"]

    increase = (
        -(contract - basal_contract) / constriction_spec["contract_span"]
    ) * neighbor["order"] + contract

    specs = {
        "face_id": neighbor["id"],
        "contractile_increase": increase,
        "critical_area": constriction_spec["critical_area"],
        "max_contractility": 50,
        "contraction_column": constriction_spec["contraction_column"],
        "multiple": True,
        "unique": False,
    }

    return specs
