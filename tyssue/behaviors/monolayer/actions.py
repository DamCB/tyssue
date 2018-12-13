"""
Basic monolayer event module
=======================


"""

import logging

logger = logging.getLogger(__name__)


def grow(monolayer, cell, grow_rate):
    """Multiplies the equilibrium volume of face
    by a factor (1+shrink_rate)
    """
    factor = 1 + grow_rate
    monolayer.cell_df.loc[cell, "prefered_vol"] *= factor
    monolayer.cell_df.loc[cell, "prefered_area"] *= factor ** (2 / 3)


def shrink(monolayer, cell, shrink_rate):
    """Divides the equilibrium volume of the cell
    by a factor (1+shrink_rate) and its equilibrium area
    by (1+shrink_rate)^2/3
    """
    factor = 1 + shrink_rate
    monolayer.cell_df.loc[cell, "prefered_vol"] /= factor
    monolayer.cell_df.loc[cell, "prefered_area"] /= factor ** (2 / 3)


def contract(
        monolayer,
        face,
        contractile_increase,
        multiple=False,
        contraction_column="contractility"
):
    """
    Contract the face by increasing the 'contractility' parameter
    by contractile_increase
    """
    if multiple:
        monolayer.face_df.loc[face, contraction_column] *= contractile_increase
    else:
        monolayer.face_df.loc[face, contraction_column] += contractile_increase


def contract_apical_face(
    monolayer,
    face_id,
    contractile_increase=1.0,
    critical_area=1e-2,
    max_contractility=50,
    multiple=False,
    contraction_column="contractility"
):
    """Single step contraction event for apical face only
    """
    face = monolayer.idx_lookup(face_id, "face")
    if face is None:
        return
    if (
        (monolayer.face_df.loc[face, "segment"] != "apical")
        or (monolayer.face_df.loc[face, "area"] < critical_area)
        or (monolayer.face_df.loc[face, contraction_column] > max_contractility)
    ):
        return
    contract(monolayer, face, contractile_increase,
             multiple, contraction_column)


def ab_pull(monolayer, cell, radial_tension, distributed=False):
    """Adds a linear tension to the apical-to-basal edges
    of a cell
    """
    cell_edges = monolayer.edge_df[monolayer.edge_df["cell"] == cell]
    lateral_edges = cell_edges[cell_edges["segment"] == "lateral"]
    srce_segment = monolayer.upcast_srce(monolayer.vert_df["segment"]).loc[
        lateral_edges.index
    ]
    trgt_segment = monolayer.upcast_trgt(monolayer.vert_df["segment"]).loc[
        lateral_edges.index
    ]

    ab_edges = lateral_edges[
        (srce_segment == "apical") & (trgt_segment == "basal")
    ].index
    ba_edges = lateral_edges[
        (trgt_segment == "apical") & (srce_segment == "basal")
    ].index

    if distributed:
        new_tension = radial_tension / (len(ab_edges) + len(ba_edges))
    else:
        new_tension = radial_tension
    monolayer.edge_df.loc[ab_edges, "line_tension"] += new_tension
    monolayer.edge_df.loc[ba_edges, "line_tension"] += new_tension
