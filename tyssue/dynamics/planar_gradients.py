import pandas as pd

from ..utils.utils import _to_2d


def area_grad(sheet):

    coords = sheet.coords
    inv_area = sheet.edge_df.eval("1 / (4 * sub_area)")

    face_pos = sheet.edge_df[["f" + c for c in coords]]
    srce_pos = sheet.edge_df[["s" + c for c in coords]]
    trgt_pos = sheet.edge_df[["t" + c for c in coords]]

    r_ak = srce_pos - face_pos.values
    r_aj = trgt_pos - face_pos.values
    grad_a_srce = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_a_srce["gx"] = r_aj["ty"] * sheet.edge_df["nz"]
    grad_a_srce["gy"] = -r_aj["tx"] * sheet.edge_df["nz"]
    grad_a_trgt = pd.DataFrame(index=sheet.edge_df.index, columns=["gx", "gy"])
    grad_a_trgt["gx"] = -r_ak["sy"] * sheet.edge_df["nz"]
    grad_a_trgt["gy"] = r_ak["sx"] * sheet.edge_df["nz"]

    grad_a_srce = _to_2d(inv_area) * grad_a_srce
    grad_a_trgt = _to_2d(inv_area) * grad_a_trgt

    return grad_a_srce, grad_a_trgt


def lumen_area_grad(eptm):
    apical_pos = eptm.vert_df[["x", "y"]].copy()
    apical_pos.loc[eptm.basal_verts] = 0
    srce_pos = eptm.upcast_srce(apical_pos)
    trgt_pos = eptm.upcast_trgt(apical_pos)
    grad_srce = srce_pos.copy()
    grad_srce.columns = ["gx", "gy"]
    grad_trgt = grad_srce.copy()
    grad_srce["gx"] = trgt_pos["y"]
    grad_srce["gy"] = -trgt_pos["x"]
    grad_trgt["gx"] = -srce_pos["y"]
    grad_trgt["gy"] = srce_pos["x"]
    # minus sign due to the backward orientation
    return -grad_srce, -grad_trgt
