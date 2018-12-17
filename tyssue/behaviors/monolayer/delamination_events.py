import logging
import random
import numpy as np

from ...utils.decorators import cell_lookup
from ...topology.bulk_topology import IH_transition, HI_transition
from .actions import shrink, contract, relax, ab_pull
from .basic_events import contraction
from ..sheet.basic_events import contraction as sheet_contraction
from ..sheet.delamination_events import _neighbor_contractile_increase

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
    "with_rearrangement": False,
    "critical_area_reduction": 5,
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
        faces_in_cell = monolayer.face_df.loc[list_face_in_cell[cell].unique()]
        try:
            apical_face_id = monolayer.face_df[
                (monolayer.face_df.index.isin(list_face_in_cell[cell]))
                & (monolayer.face_df.segment == "apical")
            ].index[0]
        except Exception:  # TODO fix that
            apical_face_id = None


        if apical_face_id is None:
            print("No more apical constriction... ")

        else :
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

                    sheet = monolayer.get_sub_sheet("apical")

                    neighbors = sheet.get_neighborhood(
                        apical_face_id, constriction_spec["contract_span"]
                    ).dropna()
                    neighbors["id"] = sheet.face_df.loc[
                        neighbors.face, "id"].values

                    # remove cell which are not mesodermal
                    ectodermal_cell = sheet.face_df.loc[neighbors.face][
                        ~sheet.face_df.loc[neighbors.face, "is_mesoderm"]
                    ].id.values

                    neighbors = neighbors.drop(
                        neighbors[neighbors.id.isin(ectodermal_cell)].index
                    )

                    manager.extend(
                        [
                            (
                                sheet_contraction,
                                _neighbor_contractile_increase(
                                    neighbor, constriction_spec),
                            )
                            for _, neighbor in neighbors.iterrows()
                        ]
                    )

            if constriction_spec["with_rearrangement"]:
                # If apical face has already been removed
                # Need to eliminate lateral face until obtain cell with 4 faces

                # If apical face area are under threshold
                if apical_face_area < constriction_spec["critical_area_reduction"]:
                    # Reduce neighbours for the apical face (until 3)
                    if monolayer.face_df.loc[apical_face_id, "num_sides"] > 3:
                        #"Remove" the shortest edge
                        e_min = monolayer.edge_df[monolayer.edge_df[
                            "face"] == apical_face_id]["length"].idxmin()
                        prev_nums = {
                            "edge": monolayer.Ne,
                            "face": monolayer.Nf,
                            "vert": monolayer.Nv,
                        }

                        monolayer.settings["threshold_length"] = 1e-3
                        IH_transition(monolayer, e_min)
                        monolayer.face_df.loc[
                            prev_nums["face"]:, "contractility"] = 0

                    elif monolayer.face_df.loc[apical_face_id, "num_sides"] == 3:
                        prev_nums = {
                            "edge": monolayer.Ne,
                            "face": monolayer.Nf,
                            "vert": monolayer.Nv,
                        }
                        HI_transition(monolayer, apical_face_id)



        """proba_tension = np.exp(-apical_face_area /
                               constriction_spec["critical_area"])
        aleatory_number = random.uniform(0, 1)

        if current_traction < constriction_spec["max_traction"]:
            if aleatory_number < proba_tension:
                current_traction += 1

                if len(faces_in_cell) > 4:
                    print("No traction when there is still an apical face")
                    #ab_pull(monolayer, cell, constriction_spec[
                    #        "radial_tension"], True)
                if len(faces_in_cell) == 4:
                    print("need to be coded")

                constriction_spec.update(
                    {"current_traction": current_traction})"""
    manager.append(constriction, **constriction_spec)
