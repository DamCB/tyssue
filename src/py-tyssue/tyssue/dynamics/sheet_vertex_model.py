
import pandas as pd
import numpy as np

from ..utils.utils import (_to_3d, set_data_columns,
                           update_default)

default_params = {
    "line_tension": 0.12,
    "contractility": 0.04,
    "vol_elasticity": 1.0,
    "prefered_height": 24.0,
    "prefered_area": 10.0
    }


cell_dtypes = ["contractility", 
               "vol_elasticity", 
               "prefered_height",
               "prefered_area",
               "prefered_vol"]

je_dtypes = ['line_tension',]


def get_dyn_data(paramters):

    dyn_data = {
        'cell': {key: (paramters[key], np.float)
                 for key in cell_dtypes},
        'je': {key: (paramters[key], np.float)
                 for key in je_dtypes},
        }
    return dyn_data

def set_dynamic_columns(sheet, parameters=None):
    
    parameters = update_default(default_params, parameters)
    dyn_data = get_dyn_data(parameters)
    set_data_columns(sheet, dyn_data)


def dimentionalize(parameters=None):

    parameters = update_default(default_params, parameters)
    dim_params = parameters.copy()
    Kv = parameters['vol_elasticity']
    A0 = parameters['prefered_area']
    h0 = parameters['prefered_height']
    dim_params['contractility'] = (parameters['contractility'] * 
                                   Kv * A0 * h0**2)
    dim_params['line_tension'] = (parameters['line_tension'] *
                                  Kv * A0**1.5 * h0**2)
    dim_params['prefered_vol'] = A0 * h0
    dim_params['norm_factor'] = Kv * (A0 * h0)**2

    return dim_params


def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * mesh: a :class:`tyssue.object.sheet.Sheet` instance
    * full_output: if True, returns the enery components
    '''
    E_t = sheet.je_df['line_tension'] * sheet.je_df['length']
    E_v = 0.5 * (sheet.cell_df['vol_elasticity'] *
                  (sheet.cell_df['vol'] -
                   sheet.cell_df['prefered_vol'])**2 *
                   sheet.cell_df['is_alive'])
    E_c = 0.5 * (sheet.cell_df['contractility'] *
                 sheet.cell_df['perimeter']**2 *
                 sheet.cell_df['is_alive'])
    if full_output:
        return E_t, E_c, E_v
    else:
        return E_t.sum() + (E_c+E_v).sum()


def compute_gradient(sheet, components=False, 
                     dcoords=None, ncoords=None):
    '''
    If components is True, returns the individual terms
    (grad_t, grad_c, grad_v)
    '''
    if dcoords is None:
        dcoords = ['d'+c for c in sheet.coords]
    if ncoords is None:
        ncoords = ['n'+c for c in sheet.coords]
    
    sheet.grad_i_lij = - (sheet.je_df[dcoords] /
                          _to_3d(sheet.je_df['length']))

    sheet.grad_i_lij.columns = sheet.coords
    grad_t = tension_grad(sheet)
    grad_c = contractile_grad(sheet)
    grad_v = volume_grad(sheet, dcoords, ncoords)

    grad_i = grad_t + grad_c + grad_v
    if components:
        return grad_i, grad_t, grad_c, grad_v
    return grad_i


def tension_grad(sheet):

    _grad_t = (sheet.grad_i_lij
               * _to_3d(sheet.je_df['line_tension']))

    grad_t = _grad_t.sum(level='srce').loc[sheet.jv_idx]
    return grad_t


def contractile_grad(sheet):

    contract = sheet.cell_df['contractility']
    perimeter = sheet.cell_df['perimeter']

    gamma_L = contract * perimeter
    gamma_L = sheet.upcast_cell(gamma_L)

    _grad_c = sheet.grad_i_lij * _to_3d(gamma_L)
    grad_c = _grad_c.sum(level='srce').loc[sheet.jv_idx]

    return grad_c


def volume_grad(sheet, dcoords=None, ncoords=None):
    '''
    Computes 
    :math:`\sum_\alpha\nabla_i \left(K (V_\alpha - V_0)^2\right)`
    '''
    if dcoords is None:
        dcoords = ['d'+c for c in sheet.coords]
    if ncoords is None:
        ncoords = ['n'+c for c in sheet.coords]

    # volumic elastic modulus
    elasticity = sheet.cell_df['vol_elasticity']
    pref_V = sheet.cell_df['prefered_vol']
    V = sheet.cell_df['vol']
    # this is K * (V - V0)
    KV_V0 = elasticity * (V - pref_V) * sheet.cell_df['is_alive']
    tri_KV_V0 = sheet.upcast_cell(KV_V0)
    h_nu = sheet.cell_df['height'] / (2 * sheet.cell_df['num_sides'])

    # edge vertices
    r_ijs = sheet.je_df[dcoords]
    
    cross_ur = pd.DataFrame(
        np.cross(sheet.je_df[ncoords], r_ijs),
        index=sheet.je_idx, columns=sheet.coords)

    cell_term_ = cross_ur.groupby(level='cell').sum() * _to_3d(KV_V0 * h_nu)

    cell_term = sheet.upcast_cell(cell_term_)
    grad_v = cell_term.groupby(level='srce').sum()
    
    r_to_rho = sheet.jv_df[sheet.coords] / _to_3d(sheet.jv_df['rho'])
    r_to_rho = sheet.upcast_srce(df=r_to_rho)
    r_to_rho.columns = sheet.coords
    
    r_aj = (sheet.upcast_trgt(sheet.jv_df[sheet.coords]) -
            sheet.upcast_cell(sheet.cell_df[sheet.coords]))
    r_aj.columns = sheet.coords
    
    normals = sheet.je_df[ncoords]
    cross_aj = pd.DataFrame(np.cross(r_aj, normals),
                            columns=sheet.coords, index=sheet.je_idx)

    tri_height = sheet.upcast_cell(sheet.cell_df['height'])

    sub_area = sheet.je_df['sub_area']

    ij_term = _to_3d(tri_KV_V0) * (_to_3d(sub_area / 2) * r_to_rho +
                                    _to_3d(tri_height / 2) * cross_aj)
    ij_term = pd.DataFrame(ij_term, index=sheet.je_idx,
                           columns=sheet.coords)
    grad_v = grad_v + ij_term.groupby(level='srce').sum()
    grad_v = grad_v
    return grad_v.loc[sheet.jv_idx]
