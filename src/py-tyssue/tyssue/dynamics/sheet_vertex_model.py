

"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""

import pandas as pd
import numpy as np
from copy import deepcopy

from ..utils.utils import (_to_3d, set_data_columns,
                           update_default)

def get_default_mod_specs():
    default_mod_specs = {
        "cell": {
            "contractility": (0.04, np.float),
            "vol_elasticity": (1., np.float),
            "prefered_height": (10., np.float),
            "prefered_area": (24., np.float),
            "prefered_vol": (0., np.float),
            },
        "je": {
            "line_tension": (0.12, np.float),
            },
        "jv": {
            "radial_tension": (0., np.float),
            }
        }
    return default_mod_specs

def dimentionalize(mod_specs, **kwargs):

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)

    Kv = dim_mod_specs['cell']['vol_elasticity'][0]
    A0 = dim_mod_specs['cell']['prefered_area'][0]
    h0 = dim_mod_specs['cell']['prefered_height'][0]
    lbda = dim_mod_specs['je']['line_tension'][0]
    gamma = dim_mod_specs['cell']['contractility'][0]

    dim_mod_specs['cell']['contractility'] = (gamma * Kv*A0 * h0**2,
                                              np.float)

    dim_mod_specs['cell']['prefered_vol'] = (A0 * h0, np.float)
    dim_mod_specs['je']['line_tension'] = (lbda * Kv * A0**1.5 * h0**2,
                                           np.float)
    norm_factor = Kv*(A0*h0)**2

    return dim_mod_specs, norm_factor


def elastic_force(element_df,
                  var='vol',
                  elasticity='vol_elasticity',
                  prefered='prefered_vol'):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    force = element_df.eval('{K} * ({x} - {x0})'.format(**params))
    return force


def elastic_energy(element_df,
                   var='vol',
                   elasticity='vol_elasticity',
                   prefered='prefered_vol'):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    energy = element_df.eval('0.5 * {K} * ({x} - {x0}) ** 2'.format(**params))
    return energy


def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * mesh: a :class:`tyssue.object.sheet.Sheet` instance
    * full_output: if True, returns the enery components
    '''
    # consider only live cells:
    live_cell_df = sheet.cell_df[sheet.cell_df.is_alive==1]
    upcast_alive = sheet.upcast_cell(sheet.cell_df.is_alive)
    live_je_df = sheet.je_df[upcast_alive==1]

    E_t = live_je_df.eval('line_tension * length / 2')
    E_v = elastic_energy(live_cell_df,
                         var='vol',
                         elasticity='vol_elasticity',
                         prefered='prefered_vol')
    E_c = live_cell_df.eval('0.5 * contractility * perimeter ** 2')
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
    grad_v = volume_grad(sheet, sheet.coords)

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


def volume_grad(sheet, coords=None):
    ''' Computes
    :math:`\sum_\alpha\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
    '''
    if coords is None:
        coords = sheet.coords
    dcoords = ['d'+c for c in sheet.coords]
    ncoords = ['n'+c for c in sheet.coords]

    # volumic elastic force
    # this is K * (V - V0)
    kv_v0_ = elastic_force(sheet.cell_df,
                           var='vol',
                           elasticity='vol_elasticity',
                           prefered='prefered_vol')

    kv_v0_ = (kv_v0_ * sheet.cell_df['is_alive'])
    kv_v0 = sheet.upcast_cell(kv_v0_)

    # # First term of the gradient
    # ## cross product of normals with edge
    cross_ur = pd.DataFrame(
        np.cross(sheet.je_df[ncoords],
                 sheet.je_df[dcoords]),
        index=sheet.je_idx, columns=sheet.coords
        )
    # ## mutliplicative factor
    h_nu = sheet.cell_df.eval('height / 2 * num_sides')
    cell_term_ = cross_ur.groupby(level='cell').sum() * _to_3d(h_nu)
    cell_term = sheet.upcast_cell(cell_term_)

    # # Second term of the gradient
    # ## r_i / rho_i
    r_to_rho = sheet.jv_df[sheet.coords] / _to_3d(sheet.jv_df['rho'])
    r_to_rho = sheet.upcast_srce(df=r_to_rho)
    r_to_rho.columns = sheet.coords

    # ## cross product of cell to target vector with the normals
    r_aj = (sheet.upcast_trgt(sheet.jv_df[sheet.coords]) -
            sheet.upcast_cell(sheet.cell_df[sheet.coords]))
    r_aj.columns = sheet.coords
    normals = sheet.je_df[ncoords]
    cross_aj = pd.DataFrame(np.cross(r_aj, normals),
                            columns=sheet.coords, index=sheet.je_idx)
    # ## cell sub volume
    tri_height = sheet.upcast_cell(sheet.cell_df['height'])
    sub_area = sheet.je_df['sub_area']

    ij_term_ = (_to_3d(sub_area / 2) * r_to_rho +
                _to_3d(tri_height / 2) * cross_aj)
    ij_term = pd.DataFrame(ij_term_,
                           index=sheet.je_idx,
                           columns=sheet.coords)

    grad_v = (_to_3d(kv_v0) *
              (cell_term + ij_term)).groupby(level='srce').sum()
    return grad_v.loc[sheet.jv_idx]
