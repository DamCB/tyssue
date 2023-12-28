"""
Base gradients for bulk geometry
"""
import numpy as np
import pandas as pd


def volume_grad(eptm):

    coords = eptm.coords
    cell_pos = eptm.edge_df[["c" + c for c in coords]].values
    face_pos = eptm.edge_df[["f" + c for c in coords]].values
    srce_pos = eptm.edge_df[["s" + c for c in coords]].values
    trgt_pos = eptm.edge_df[["t" + c for c in coords]].values

    grad_v_srce = np.cross((trgt_pos - cell_pos), (face_pos - cell_pos)) / 4
    grad_v_trgt = -np.cross((srce_pos - cell_pos), (face_pos - cell_pos)) / 4
    return (
        pd.DataFrame(grad_v_srce, index=eptm.edge_df.index, columns=eptm.coords),
        pd.DataFrame(grad_v_trgt, index=eptm.edge_df.index, columns=eptm.coords),
    )


def lumen_volume_grad(eptm):
    """
    Calculates the gradient for the volume enclosed by the epithelium.

    For a monolayer, it will by default compute the volume enclosed
    by the basal side (edges whose 'segment' column is "basal").
    If the polarity is reversed and the apical side faces the lumen,
    this can be changed by setting eptm.settings["lumen_side"] to 'apical'

    """

    coords = eptm.coords
    if "segment" in eptm.edge_df:
        lumen_side = eptm.settings.get("lumen_side", "basal")
        basal_edges = eptm.edge_df[eptm.edge_df.segment == lumen_side]
        face_pos = basal_edges[["f" + c for c in coords]].values
        srce_pos = basal_edges[["s" + c for c in coords]].values
        trgt_pos = basal_edges[["t" + c for c in coords]].values

        grad_v_srce = pd.DataFrame(
            np.zeros((eptm.Ne, 3)), index=eptm.edge_df.index, columns=eptm.coords
        )
        grad_v_trgt = pd.DataFrame(
            np.zeros((eptm.Ne, 3)), index=eptm.edge_df.index, columns=eptm.coords
        )

        grad_v_srce.loc[basal_edges.index] = -np.cross((trgt_pos), (face_pos)) / 4
        grad_v_trgt.loc[basal_edges.index] = np.cross((srce_pos), (face_pos)) / 4

        return grad_v_srce, grad_v_trgt
    else:
        face_pos = eptm.edge_df[["f" + c for c in coords]].values
        srce_pos = eptm.edge_df[["s" + c for c in coords]].values
        trgt_pos = eptm.edge_df[["t" + c for c in coords]].values

        grad_v_srce = np.cross((trgt_pos), (face_pos)) / 4
        grad_v_trgt = -np.cross((srce_pos), (face_pos)) / 4
        return (
            pd.DataFrame(grad_v_srce, index=eptm.edge_df.index, columns=eptm.coords),
            pd.DataFrame(grad_v_trgt, index=eptm.edge_df.index, columns=eptm.coords),
        )
