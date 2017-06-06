"""
Base gradients for sheet like geometries
"""

import numpy as np
import pandas as pd

from ..utils.utils import _to_3d, _to_2d



def cyl_height_grad(vert_df, coords):

    r_to_rho = vert_df[coords] / _to_3d(vert_df['rho'])
    r_to_rho[coords[-1]] = 0.
    return r_to_rho



def height_grad(sheet):

    w = sheet.settings['height_axis']
    u, v = (c for c in sheet.coords if c != w)
    coords = [u, v, w]

    if sheet.settings['geometry'] == 'cylindrical':
        r_to_rho = sheet.vert_df[coords] / _to_3d(sheet.vert_df['rho'])
        r_to_rho[w] = 0.

    elif sheet.settings['geometry'] == 'flat':
        r_to_rho = sheet.vert_df[coords].copy()
        r_to_rho[[u, v]] = 0.
        r_to_rho[[w]] = 1.

    elif sheet.settings['geometry'] == 'spherical':
        r_to_rho = sheet.vert_df[coords] / _to_3d(sheet.vert_df['rho'])

    elif sheet.settings['geometry'] == 'rod':

        r_to_rho = sheet.vert_df[coords].copy()

        r_to_rho[[u, v]] = sheet.vert_df[[u, v]] / _to_2d(sheet.vert_df['rho'])
        r_to_rho[w] = 0.

        l_mask = sheet.vert_df[sheet.vert_df['left_tip'].astype(np.bool)].index
        r_mask = sheet.vert_df[sheet.vert_df['right_tip'].astype(np.bool)].index
        a, b = sheet.settings['ab']
        w0 = a - b

        l_rel_z = sheet.vert_df.loc[l_mask, 'z'] - w0
        r_to_rho.loc[l_mask, w] = l_rel_z / sheet.vert_df['rho']

        r_rel_z = sheet.vert_df.loc[r_mask, 'z'] + w0
        r_to_rho.loc[r_mask, w] = r_rel_z / sheet.vert_df['rho']

    return r_to_rho


def area_grad(sheet):

    coords = sheet.coords
    ncoords = sheet.ncoords
    inv_area = sheet.edge_df.eval('1 / (4 * sub_area)')
    # Some segmentations create null areas
    inv_area.replace(np.inf, 0, inplace=True)
    inv_area.replace(-np.inf, 0, inplace=True)

    face_pos = sheet.upcast_face(sheet.face_df[coords])
    srce_pos = sheet.upcast_srce(sheet.vert_df[coords])
    trgt_pos = sheet.upcast_trgt(sheet.vert_df[coords])

    r_ak = srce_pos - face_pos
    r_aj = trgt_pos - face_pos

    grad_a_srce = _to_3d(inv_area) * np.cross(r_aj, sheet.edge_df[ncoords])
    grad_a_trgt = _to_3d(inv_area) * np.cross(sheet.edge_df[ncoords], r_ak)
    return (pd.DataFrame(grad_a_srce,
                         index=sheet.edge_df.index,
                         columns=sheet.coords),
            pd.DataFrame(grad_a_trgt, index=sheet.edge_df.index,
                         columns=sheet.coords))
