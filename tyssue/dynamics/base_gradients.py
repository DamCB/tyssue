from ..utils.utils import _to_3d, _to_2d


def length_grad(sheet):
    '''returns -(dx/l, dy/l, dz/l), ie grad_i(l_ij))
    '''
    if sheet.dim == 2:
        grad_lij = - (sheet.edge_df[sheet.dcoords] /
                      _to_2d(sheet.edge_df['length']))
    elif sheet.dim == 3:
        grad_lij = - (sheet.edge_df[sheet.dcoords] /
                      _to_3d(sheet.edge_df['length']))
    grad_lij.columns = sheet.coords
    return grad_lij
