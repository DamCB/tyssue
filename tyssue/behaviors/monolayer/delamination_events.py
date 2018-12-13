import logging
import random
import numpy as np

from ...utils.decorators import cell_lookup
from ...topology.bulk_topology import IH_transition, HI_transition
from .actions import shrink, contract, relax, ab_pull

logger = logging.getLogger(__name__)


default_constriction_spec = {
    "cell_id": -1,
    "cell": -1,
    "contract_rate": 2.0,
    "critical_area": 1e-2,
    "shrink_rate": 0.4,
    "critical_volume": 0.1,
    "radial_tension": 1.0,
    "contract_neighbors": True,
    "critical_area_neighbors": 10,
    "contract_span": 2,
    "basal_contract_rate": 1.001,
    "current_traction": 0,
    "max_traction": 30,
    "contraction_column": "contractility",
}


@cell_lookup
def constriction(monolayer, manager, **kwargs):
    """Constriction behavior

    Parameters
    ----------
    monolayer : a :cass:`Monolayer` object
    manager : a :class:`EventManager` object
    """
    constriction_spec = default_constriction_spec
    constriction_spec.update(**kwargs)

    cell = constriction_spec["cell"]
    contract_rate = constriction_spec["contract_rate"]
    current_traction = constriction_spec["current_traction"]

    if "is_relaxation" in monolayer.cell_df.columns:
        if monolayer.cell_df.loc[cell, "is_relaxation"]:
            relax(
                monolayer, cell, contract_rate, constriction_spec[
                    "contraction_column"]
            )

    if monolayer.cell_df.loc[cell, "is_mesoderm"]:

        # Find the apical face id of the cell
        list_face_in_cell = monolayer.get_orbits("cell", "face")

        apical_face_id = monolayer.face_df[
            (monolayer.face_df.index.isin(list_face_in_cell[cell]))
            & (monolayer.face_df.segment == "apical")
        ].index[0]

        apical_face_area = monolayer.face_df.loc[apical_face_id, "area"]

        if apical_face_area > constriction_spec["critical_area"]:
            contract(
                monolayer,
                apical_face_id,
                contract_rate,
                True,
                constriction_spec["contraction_column"],
            )

            if (constriction_spec["contract_neighbors"]) & (
                apical_face_area < constriction_spec["critical_area_neighbors"]
            ):

                print("Contract neighbors is not implemented yet")

        proba_tension = np.exp(-apical_face_area /
                               constriction_spec["critical_area"])
        aleatory_number = random.uniform(0, 1)

        if current_traction < constriction_spec["max_traction"]:
            if aleatory_number < proba_tension:
                current_traction += 1

                ab_pull(monolayer, cell, constriction_spec[
                        "radial_tension"], True)

                constriction_spec.update(
                    {"current_traction": current_traction})
    manager.append(constriction, **constriction_spec)
