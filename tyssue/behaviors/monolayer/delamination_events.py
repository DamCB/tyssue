import logging
import random
import numpy as np

from ...utils.decorators import cell_lookup
from ...topology.bulk_topology import IH_transition, HI_transition
from .actions import shrink, contract, relax, ab_pull, ab_pull_edge
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

    # Find the apical face id of the cell
    list_face_in_cell = monolayer.get_orbits("cell", "face")
    faces_in_cell = monolayer.face_df.loc[list_face_in_cell[cell].unique()]
    if "apical" in faces_in_cell["segment"].values:
        apical_face_id = faces_in_cell[faces_in_cell["segment"] == "apical"].index[0]
    else:
        apical_face_id = None

    if apical_face_id is None and constriction_spec["with_rearrangement"]:
        _constrict_apical(faces_in_cell, monolayer, constriction_spec)

    else:
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
                neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values

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
                            _neighbor_contractile_increase(neighbor, constriction_spec),
                        )
                        for _, neighbor in neighbors.iterrows()
                    ]
                )

        if not constriction_spec["with_rearrangement"]:
            # we're done
            return

        # If apical face has already been removed
        # Need to eliminate lateral face until obtain cell with 4 faces

        # If apical face area are under threshold
        if apical_face_area < constriction_spec["critical_area_reduction"]:
            # Reduce neighbours for the apical face (until 3)
            if monolayer.face_df.loc[apical_face_id, "num_sides"] > 3:
                # "Remove" the shortest edge
                e_min = monolayer.edge_df[monolayer.edge_df["face"] == apical_face_id][
                    "length"
                ].idxmin()
                prev_nums = {
                    "edge": monolayer.Ne,
                    "face": monolayer.Nf,
                    "vert": monolayer.Nv,
                }

                monolayer.settings["threshold_length"] = 1e-3
                IH_transition(monolayer, e_min)
                monolayer.face_df.loc[prev_nums["face"] :, "contractility"] = 0

            elif monolayer.face_df.loc[apical_face_id, "num_sides"] == 3:
                prev_nums = {
                    "edge": monolayer.Ne,
                    "face": monolayer.Nf,
                    "vert": monolayer.Nv,
                }
                HI_transition(monolayer, apical_face_id)

    manager.append(constriction, **constriction_spec)


def _constrict_apical(faces_in_cell, monolayer, constriction_spec):

    cell = constriction_spec["cell"]
    current_traction = constriction_spec["current_traction"]

    if len(faces_in_cell) > 4:
        # Remove lateral face with 3 sides

        three_lateral = faces_in_cell[
            (faces_in_cell.segment == "lateral") & (faces_in_cell.num_sides == 3)
        ].index
        if len(three_lateral):
            face_to_eliminate = three_lateral[0]
        prev_Nf = monolayer.Nf
        HI_transition(monolayer, face_to_eliminate)
        monolayer.face_df.loc[prev_Nf:, "contractility"] = 0

    elif len(faces_in_cell) == 4:
        if monolayer.cell_df.loc[cell, "vol"] > constriction_spec["critical_volume"]:
            shrink(monolayer, cell, constriction_spec["shrink_rate"])

        if current_traction < constriction_spec["max_traction"]:
            list_vert_in_cell = monolayer.get_orbits("cell", "srce")
            vert_in_cell = monolayer.vert_df.loc[list_vert_in_cell[4].unique()]
            apical_vert = vert_in_cell[vert_in_cell["segment"] == "apical"].index[0]
            list_vert_connected_to_apical = monolayer.edge_df[
                monolayer.edge_df["srce"] == apical_vert
            ]["trgt"].unique()
            opposite_vert = -1
            for v in list_vert_connected_to_apical:
                if v not in (vert_in_cell.index):
                    opposite_vert = v
            if opposite_vert > -1:
                edges_to_pull = monolayer.edge_df[
                    (
                        (monolayer.edge_df.trgt == apical_vert)
                        & (monolayer.edge_df.srce == opposite_vert)
                    )
                    | (
                        (monolayer.edge_df.srce == apical_vert)
                        & (monolayer.edge_df.trgt == opposite_vert)
                    )
                ].index

                ab_pull_edge(
                    monolayer, edges_to_pull, constriction_spec["radial_tension"], True
                )
                current_traction += 1
                constriction_spec.update({"current_traction": current_traction})
