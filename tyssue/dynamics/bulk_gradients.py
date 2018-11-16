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
