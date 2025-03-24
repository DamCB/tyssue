"""
Base gradients for sheet like geometries
"""

import numpy as np
import pandas as pd

from ..utils.utils import to_nd  # _to_3d, _to_2d


def height_grad(sheet):

    w = sheet.settings.get("height_axis", "z")
    u, v = (c for c in sheet.coords if c != w)
    coords = [u, v, w]

    if sheet.settings.get("geometry", "cylindrical") == "cylindrical":
        r_to_rho = sheet.vert_df[coords] / to_nd(sheet.vert_df["rho"], 3)
        r_to_rho[w] = 0.0

    elif sheet.settings.get("geometry") == "flat":
        r_to_rho = sheet.vert_df[coords].copy()
        r_to_rho[[u, v]] = 0.0
        r_to_rho[[w]] = 1.0

    elif sheet.settings.get("geometry") == "spherical":
        r_to_rho = sheet.vert_df[coords] / to_nd(sheet.vert_df["rho"], 3)

    return r_to_rho


def area_grad(sheet):

    coords = sheet.coords
    ncoords = sheet.ncoords
    inv_area = sheet.edge_df.eval("1 / (4 * sub_area)")
    # Some segmentations create null areas
    inv_area.replace(np.inf, 0, inplace=True)
    inv_area.replace(-np.inf, 0, inplace=True)

    face_pos = sheet.edge_df[["f" + c for c in coords]].values
    srce_pos = sheet.edge_df[["s" + c for c in coords]].values
    trgt_pos = sheet.edge_df[["t" + c for c in coords]].values

    r_ak = srce_pos - face_pos
    r_aj = trgt_pos - face_pos

    inv_area = to_nd(inv_area, 3)
    grad_a_srce = inv_area * np.cross(r_aj, sheet.edge_df[ncoords])
    grad_a_trgt = inv_area * np.cross(sheet.edge_df[ncoords], r_ak)
    return (
        pd.DataFrame(grad_a_srce, index=sheet.edge_df.index, columns=sheet.coords),
        pd.DataFrame(grad_a_trgt, index=sheet.edge_df.index, columns=sheet.coords),
    )
