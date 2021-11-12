"""
Mesoderm invagination event module
=======================


"""

import random
import numpy as np

from ...utils.decorators import face_lookup
from .actions import increase
from .actions import ab_pull
from .basic_events import contraction
from ...topology.base_topology import collapse_edge
from ...topology.sheet_topology import split_vert


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

    if sheet.face_df.loc[face, "is_mesoderm"]:
        face_area = sheet.face_df.loc[face, "area"]

        if face_area > constriction_spec["critical_area"]:
            increase(
                sheet,
                "face",
                face,
                contract_rate,
                constriction_spec["contraction_column"],
                True,
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
    
def exchange_neighbour(sheet, manager, **kwargs):
    """
    Executes neighbour exchanges for a face:
    1) if face_df[face, 'try_collapse_edge'] is True, collapse that face's shortest edge into a vert
    2) if face_df[face, 'try_expand_vert'] is True, expand that face's highest-order vert into an edge
    
    If either or both was successful:
    1) Stores index of the new vert in face_df[face, 'edge_collapsed']
    2) Stores index of the new edge in face_df[face, 'vert_expanded']
    
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    **kwargs : parameters for the event, indices:
        face_id: id of the face being checked for exchange events
        critical_area: optional, default 1e-2, events are only executes if face area larger than this
        minimum_edges: optional, default 3, edge will not be collapsed if face with fewer sides would result
        maximum_edges: optional, default 10, vert will not be expanded if face with more sides would result
        verbose: optional, default False, print log messages if True
    """
    exchange_spec = {
        "face_id": -1,
        "critical_area": 1e-2,
        "minimum_edges": 3,
        "maximum_edges": 10
    }
    exchange_spec.update(**kwargs)
    verbose = exchange_spec.get("verbose", False)
    
    face_o = exchange_spec["face_id"]
    if len(sheet.face_df[sheet.face_df["face_o"]==face_o])!=1:
        if verbose:
            if sheet.face_df[sheet.face_df["face_o"]==face_o].empty:
                print(f"No face with original ID {face_o} exists.")
            else:
                print(f"More than one face with original ID {face_o} exists.")
        return

    face = sheet.face_df[sheet.face_df["face_o"]==face_o].index[0]
    
    do_collapse = sheet.face_df.loc[face,'try_collapse_edge']
    do_expand = sheet.face_df.loc[face,'try_expand_vert']
    if verbose and (do_collapse or do_expand):
        print(f"neighbour exchange triggered for face {face}, "
              f"originally {face_o}: collapse? {do_collapse} "
              f"expand? {do_expand} ",end="\r")
    
    if (
        do_collapse and 
        sheet.face_df.loc[face].area > exchange_spec["critical_area"]
    ):
        #attempt to collapse the face's shortest edge
        e_collapsilling = sheet.edge_df[
            sheet.edge_df["face"]==face 
        ]["length"].idxmin()
        #check first: does this face, and the one opposite that edge have enough sides (>3)?
        if verbose:
            print(f"checking conditions for edge {e_collapsilling}, "
                  f"face: {face}, originally {face_o} ",end="\r")
        if sheet.face_df.loc[face].num_sides > exchange_spec["minimum_edges"]:
            opposite_face = sheet.edge_df.loc[
                sheet.edge_df.loc[e_collapsilling].opposite
            ].face
            
            if sheet.face_df.loc[
                opposite_face
            ].num_sides > exchange_spec["minimum_edges"]:
                if verbose:
                    print(f"collapsing edge {e_collapsilling} (face {face})  ",
                          end="\r")
                remain_vert = collapse_edge(
                    sheet, 
                    e_collapsilling, 
                    reindex=True,
                    allow_two_sided=True
                )
                sheet.face_df.at[
                    face,"edge_collapsed"
                ] = sheet.vert_df.loc[remain_vert,"srce_o"]
                sheet.face_df.at[
                    opposite_face,"edge_collapsed"
                ] = sheet.vert_df.loc[remain_vert,"srce_o"]
            elif verbose:
                print(f"could not collapse edge {e_collapsilling} "
                      f"(face {face})  ",end="\r")
            
    if do_expand and sheet.face_df.loc[face].area > exchange_spec["critical_area"]:
        #attempt to expand the face's highest-order vertex
        #this works as if the cell pulled itself away from the rosette,
        #creating a new edge between it and the rosette
        verts = sheet.edge_df[sheet.edge_df["face"]==face]["srce"]
        v_expandilling = sheet.edge_df.srce.value_counts().loc[verts].idxmax()
        
        #now check if the rearrangement meets the criteria for this vertex
        # - is the vertex actually a rosette (more than tricellular)?
        # - does this create a face with too many sides?
        # nesting two if-clauses avoids calculating the second condition if the first fails
        if (len( sheet.edge_df[ sheet.edge_df["srce"]==v_expandilling ] ) > 3):
            neighbors_v_expandilling = sheet.get_neighbors(face).intersection(
                sheet.edge_df[
                    sheet.edge_df["trgt"]==v_expandilling
                ]["face"].values
            )
            
            if all(
                len(sheet.edge_df.query(f"face == {nf}")) < exchange_spec[
                    "maximum_edges"] for nf in neighbors_v_expandilling
            ):
                if verbose:
                    print(f"splitting vert {v_expandilling} (face {face})  ",end="\r")
                new_edges = split_vert(sheet, v_expandilling, face=face, multiplier=1.5, reindex=True, recenter=True)
                #the fact that the vert was expanded needs to be stored
                #for the two cells that are affected, i.e. the ones
                #bordering the new edge
                for e in new_edges:
                    sheet.face_df.at[sheet.edge_df.loc[e]["face"],"vert_expanded"] = sheet.edge_df.loc[e,"edge_o"]
    manager.append(exchange_neighbour, **exchange_spec)

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
