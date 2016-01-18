'''
Base gradients for bulk geometry
'''
import numpy as np
import pandas as pd


def volume_grad(eptm):
    cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords])
    face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
    srce_pos = eptm.upcast_srce(eptm.jv_df[eptm.coords])
    trgt_pos = eptm.upcast_trgt(eptm.jv_df[eptm.coords])
    grad_v_srce =  np.cross((srce_pos - cell_pos), (face_pos - cell_pos)) / 6
    grad_v_trgt =  np.cross((trgt_pos - cell_pos), (face_pos - cell_pos)) / 6
    return (pd.DataFrame(grad_v_srce, index=eptm.je_df.index,
                         columns=eptm.coords),
            pd.DataFrame(grad_v_trgt, index=eptm.je_df.index,
                         columns=eptm.coords))
