import pandas as pd

from ..utils.utils import _to_2d


def area_grad(sheet):

    coords = sheet.coords
    inv_area = sheet.edge_df.eval('1 / (4 * sub_area)')

    face_pos = sheet.edge_df[['f'+c for c in coords]]
    srce_pos = sheet.edge_df[['s'+c for c in coords]]
    trgt_pos = sheet.edge_df[['t'+c for c in coords]]

    r_ak = srce_pos - face_pos
    r_aj = trgt_pos - face_pos
    grad_a_srce = pd.DataFrame(index=sheet.edge_df.index,
                               columns=['gx', 'gy'])
    grad_a_srce['gx'] = r_aj['y'] * sheet.edge_df['nz']
    grad_a_srce['gy'] = -r_aj['x'] * sheet.edge_df['nz']
    grad_a_trgt = pd.DataFrame(index=sheet.edge_df.index,
                               columns=['gx', 'gy'])
    grad_a_trgt['gx'] = -r_ak['y'] * sheet.edge_df['nz']
    grad_a_trgt['gy'] = r_ak['x'] * sheet.edge_df['nz']

    grad_a_srce = _to_2d(inv_area) * grad_a_srce
    grad_a_trgt = _to_2d(inv_area) * grad_a_trgt

    return grad_a_srce, grad_a_trgt
