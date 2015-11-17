
import pandas as pd
import numpy as np

from ..utils import _to_3d

default_params = {"line_tension": 0.12,
                  "rho_lumen": 4.0,
                  "contractility": 0.04,
                  "vol_elasticity": 1.0,
                  "prefered_height": 24.0,
                  "prefered_area": 10.0}

cell_data = [
    "contractility",
    "vol_elasticity",
    "prefered_height",
    "prefered_area"
    ]

je_data = [
    "contractility",
    "vol_elasticity",
    "prefered_height",
    "prefered_area"
    ]

jv_data = []

def dimentionalize(parameters=None):

    if parameters is None:
        parameters = default_params.copy()
    parameters.update(default_params)

    dim_params = parameters.copy()
    Kv = parameters['vol_elasticity']
    A0 = parameters['prefered_area']
    h0 = parameters['prefered_height']
    dim_params['contractility'] = (parameters['contractility'] * 
                                   Kv * A0 * h0**2))
    dim_params['line_tension'] = (parameters['line_tension'] *
                                  Kv * A0**1.5 * h0**2)
    dim_params['prefered_vol'] = A0 * h0
    return dim_params


def set_dynamic_columns(sheet, parameters=None):
    '''
    parameters should be dimentionalized first (or should they?)
    '''
    if parameters is None:
        parameters = default_params.copy()
    parameters.update(default_params)

    for col in cell_data:
        sheet.cell_df[col] = parameters[col]
    for col in jv_data:
        sheet.jv_df[col] = parameters[col]
    for col in je_data:
        sheet.je_df[col] = parameters[col]


def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * mesh: a :class:`tyssue.object.sheet.Sheet` instance
    
    '''
    E_t = sheet.je_df['line_tension'] * sheet.je_df['length']

    E_v = 0.5 * (sheet.cell_df['vol_elasticity'] *
                 (sheet.cell_df['vol'] -
                  sheet.cell_df['prefered_vol'])**2
                 ) * sheet.cell_df['is_alive']
    E_c = 0.5 * (sheet.cell_df['contractility'] *
                 sheet.cell_df['perimeter']**2) * sheet.cell_df['is_alive']
    if full_output:
        return E_t, E_c, E_v
    else:
        return E_t.sum() + (E_c+E_v).sum()


def compute_gradient(mesh, components=False):
    '''
    If components is True, returns the individual terms
    (grad_t, grad_c, grad_v)
    '''
    mesh.grad_i_lij = - (mesh.fv_itoj[mesh.dcoords] /
                         _to_3d(mesh.fv_itoj['edge_length'])
                         ).loc[mesh.uix_itoj]

    grad_t = tension_grad(mesh)
    grad_c = contractile_grad(mesh)
    grad_v = volume_grad(mesh)

    grad_i = grad_t + grad_c + grad_v
    if components:
        return grad_i, grad_t, grad_c, grad_v
    return grad_i


def tension_grad(mesh):

    grad_t = mesh.grad_array.copy()

    tensions = mesh.fv_itoj['line_tension'].loc[mesh.uix_itoj]
    tensions.index.names = ('jv_i', 'jv_j')

    _grad_t = mesh.grad_i_lij * _to_3d(tensions)
    grad_t.loc[mesh.uix_active_i] = _grad_t.sum(
        level='jv_i').loc[mesh.uix_active_i].values
    grad_t.loc[mesh.uix_active_j] -= _grad_t.sum(
        level='jv_j').loc[mesh.uix_active_j].values
    return grad_t


def contractile_grad(mesh):

    grad_c = mesh.grad_array.copy()
    grad_c[:] = 0

    contract = mesh.udf_cell['contractility']
    contract.index.name = 'cell'
    perimeter = mesh.udf_cell['perimeter']

    gamma_L = contract * perimeter
    gamma_L = gamma_L.loc[mesh.tix_a]
    gamma_L.index = mesh.tix_aij

    # area_term = gamma_L.groupby(level='jv_i').apply(
    #     lambda df: df.sum(level='jv_j'))
    area_term = gamma_L.groupby(level=('jv_i', 'jv_j')).sum()

    _grad_c = mesh.grad_i_lij.loc[mesh.uix_ij] * _to_3d(
        area_term.loc[mesh.uix_ij])
    grad_c.loc[mesh.uix_active_i] = _grad_c.sum(
        level='jv_i').loc[mesh.uix_active_i].values
    grad_c.loc[mesh.uix_active_j] -= _grad_c.sum(
        level='jv_j').loc[mesh.uix_active_j].values
    return grad_c


def volume_grad(mesh):
    '''
    Computes :math:`\sum_\alpha\nabla_i \left(K (V_\alpha - V_0)^2\right)`
    '''
    grad_v = mesh.grad_array.copy()
    grad_v[:] = 0

    elasticity = mesh.udf_cell['vol_elasticity']
    pref_V = mesh.udf_cell['prefered_vol']
    V = mesh.udf_cell['vol']
    KV_V0 = elasticity * (V - pref_V)
    tri_KV_V0 = KV_V0.loc[mesh.tix_a]
    tri_KV_V0.index = mesh.tix_aij

    r_ijs = mesh.tdf_itoj[mesh.dcoords]
    cross_ur = pd.DataFrame(np.cross(mesh.faces[mesh.normal_coords], r_ijs),
                            index=mesh.tix_aij, columns=mesh.coords)

    h_nu = mesh.udf_cell['height'] / (2 * mesh.udf_cell['num_sides'])

    grad_i_V_cell = cross_ur.sum(level='cell') * _to_3d(KV_V0 * h_nu)

    cell_term_i = grad_i_V_cell.loc[mesh.tix_a].set_index(mesh.tix_ai)
    cell_term_j = grad_i_V_cell.loc[mesh.tix_a].set_index(mesh.tix_aj)

    grad_v.loc[mesh.uix_active_i] += cell_term_i.loc[mesh.uix_ai].sum(
        level='jv_i').loc[mesh.uix_active_i].values/2
    grad_v.loc[mesh.uix_active_j] += cell_term_j.loc[mesh.uix_aj].sum(
        level='jv_j').loc[mesh.uix_active_j].values/2

    _r_to_rho_i = mesh.udf_jv_i[mesh.coords] / _to_3d(mesh.udf_jv_i['rho'])
    _r_to_rho_j = mesh.udf_jv_j[mesh.coords] / _to_3d(mesh.udf_jv_j['rho'])
    r_to_rho_i = _r_to_rho_i.loc[mesh.tix_i].set_index(mesh.tix_aij)
    r_to_rho_j = _r_to_rho_j.loc[mesh.tix_j].set_index(mesh.tix_aij)
    r_ai = mesh.tdf_atoi[mesh.dcoords]
    r_aj = mesh.tdf_atoj[mesh.dcoords]
    normals = mesh.faces[mesh.normal_coords]
    cross_ai = pd.DataFrame(np.cross(normals, r_ai),
                            index=mesh.tix_aij, columns=mesh.coords)
    cross_aj = pd.DataFrame(np.cross(normals, r_aj),
                            index=mesh.tix_aij, columns=mesh.coords)

    tri_height = mesh.tdf_cell['height']
    tri_height.index = mesh.tix_aij
    sub_area = mesh.faces['sub_area']

    _ij_term = _to_3d(tri_KV_V0) * (_to_3d(sub_area / 2) * r_to_rho_i -
                                    _to_3d(tri_height / 2) * cross_aj)
    _jk_term = _to_3d(tri_KV_V0) * (_to_3d(sub_area / 2) * r_to_rho_j -
                                    _to_3d(tri_height / 2) * cross_ai)

    #ij_term = _ij_term.groupby(level=('jv_i', 'jv_j')).sum()
    #jk_term = _jk_term.groupby(level=('jv_j', 'jv_i')).sum()

    grad_v.loc[mesh.uix_active_i] += _ij_term.sum(
        level='jv_i').loc[mesh.uix_active_i].values
    grad_v.loc[mesh.uix_active_j] += _jk_term.sum(
        level='jv_j').loc[mesh.uix_active_j].values

    return grad_v
