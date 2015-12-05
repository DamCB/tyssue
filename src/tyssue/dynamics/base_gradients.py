
from ..utils.utils import _to_3d

def length_grad(sheet):
    '''returns -(dx/l, dy/l, dz/l), ie grad_i(l_ij))
    '''
    dcoords = ['d'+c for c in sheet.coords]
    grad_lij = - (sheet.je_df[dcoords] /
                  _to_3d(sheet.je_df['length']))
    grad_lij.columns = sheet.coords
    return grad_lij
