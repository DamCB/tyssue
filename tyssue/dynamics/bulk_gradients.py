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


def all_volume_grad(eptm):
    """
    Calculate volume gradient for the all object.
    For example calculate the yolk volume of the embryo.
    """

    coords = eptm.coords
    if "segment" in eptm.edge_df:
        basal_edges = eptm.edge_df[eptm.edge_df.segment == "basal"].copy()
        face_pos = basal_edges[["f" + c for c in coords]].values
        srce_pos = basal_edges[["s" + c for c in coords]].values
        trgt_pos = basal_edges[["t" + c for c in coords]].values
        grad_v_srce = np.cross((trgt_pos), (face_pos)) / 4
        grad_v_trgt = -np.cross((srce_pos), (face_pos)) / 4
        return (
            pd.DataFrame(grad_v_srce, index=basal_edges.index, columns=eptm.coords),
            pd.DataFrame(grad_v_trgt, index=basal_edges.index, columns=eptm.coords),
        )
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
