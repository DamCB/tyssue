
import numpy as np
import pandas as pd

from ..utils.utils import _to_2d, _to_3d


def area_grad(sheet, coords):

    if coords is None:
        coords = sheet.coords
    inv_area = sheet.je_df.eval('1 / (4 * sub_area)')

    cell_pos = sheet.upcast_cell(sheet.cell_df[coords])
    srce_pos = sheet.upcast_srce(sheet.jv_df[coords])
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords])

    r_ak = srce_pos - cell_pos
    r_aj = trgt_pos - cell_pos
    grad_a_srce = pd.DataFrame(index=sheet.je_idx, columns=sheet.coords)
    grad_a_srce['x'] = r_aj['y'] * sheet.je_df['nz']
    grad_a_srce['y'] = -r_aj['x'] * sheet.je_df['nz']

    grad_a_trgt = pd.DataFrame(index=sheet.je_idx, columns=sheet.coords)
    grad_a_trgt['x'] = -r_ak['y'] * sheet.je_df['nz']
    grad_a_trgt['y'] = r_ak['x'] * sheet.je_df['nz']

    if len(sheet.coords) == 2:
        grad_a_srce = _to_2d(inv_area) * grad_a_srce
        grad_a_trgt = _to_2d(inv_area) * grad_a_trgt
    if len(sheet.coords) == 3:
        grad_a_srce = _to_3d(inv_area) * grad_a_srce
        grad_a_trgt = _to_3d(inv_area) * grad_a_trgt

    return grad_a_srce, grad_a_trgt
