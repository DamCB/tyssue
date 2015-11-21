"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""

import pandas as pd
import numpy as np
from copy import deepcopy

from ..utils.utils import _to_3d, _to_2d


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
            },
        "settings": {
            'grad_norm_factor': 1.,
            'nrj_norm_factor': 1.,
            }
        }
    return default_mod_specs


def dimentionalize(mod_specs, **kwargs):

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)

    Kv = dim_mod_specs['cell']['vol_elasticity'][0]
    A0 = dim_mod_specs['cell']['prefered_area'][0]
    h0 = dim_mod_specs['cell']['prefered_height'][0]
    gamma = dim_mod_specs['cell']['contractility'][0]

    dim_mod_specs['cell']['contractility'] = (gamma * Kv*A0 * h0**2,
                                              np.float)

    dim_mod_specs['cell']['prefered_vol'] = (A0 * h0, np.float)

    lbda = dim_mod_specs['je']['line_tension'][0]
    dim_mod_specs['je']['line_tension'] = (lbda * Kv * A0**1.5 * h0**2,
                                           np.float)

    dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5 * h0**2
    dim_mod_specs['settings']['nrj_norm_factor'] = Kv * (A0*h0)**2

    return dim_mod_specs


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
        return (E / sheet.nrj_norm_factor for E in (E_t, E_c, E_v))
    else:
        return (E_t.sum() + (E_c+E_v).sum()) / sheet.nrj_norm_factor

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
    norm_factor = sheet.grad_norm_factor

    sheet.grad_i_lij = - (sheet.je_df[dcoords] /
                          _to_3d(sheet.je_df['length']))

    sheet.grad_i_lij.columns = sheet.coords
    grad_t = tension_grad(sheet)
    grad_c = contractile_grad(sheet)
    grad_v = volume_grad(sheet, sheet.coords)

    grad_i = grad_t + grad_c + grad_v
    live_je = sheet.upcast_cell(sheet.cell_df['is_alive'])
    grad_i = grad_i * _to_3d(live_je)
    if components:
        return grad_i, grad_t, grad_c, grad_v
    return grad_i.sum(level='srce') / norm_factor


def tension_grad(sheet):

    grad_t = (sheet.grad_i_lij
              * _to_3d(sheet.je_df['line_tension']))

    #grad_t = _grad_t.sum(level='srce').loc[sheet.jv_idx]
    return grad_t


def contractile_grad(sheet):

    gamma_ = sheet.cell_df.eval('contractility * perimeter')
    gamma = sheet.upcast_cell(gamma_)

    grad_c = sheet.grad_i_lij * _to_3d(gamma)
    # grad_c = _grad_c.sum(level='srce').loc[sheet.jv_idx]

    return grad_c

def height_grad(sheet, coords):

    r_to_rho = sheet.jv_df[coords] / _to_3d(sheet.jv_df['rho'])
    ### Cyl. geom
    r_to_rho['z'] = 0.

    r_to_rho = sheet.upcast_srce(df=r_to_rho)
    r_to_rho.columns = sheet.coords
    return r_to_rho / 2

def area_grad(sheet, coords):

    if coords is None:
        coords = sheet.coords
    ncoords = ['n'+c for c in sheet.coords]
    dcoords = ['d'+c for c in sheet.coords]
    inv_area = sheet.je_df.eval('1 / (2 * sub_area)')

    # ## cross product of normals with edge
    n_sides_cor_ = sheet.upcast_cell(1/sheet.cell_df['num_sides'])
    n_sides_cor = _to_3d(n_sides_cor_)
    cross_aij_ij = n_sides_cor * np.cross(sheet.je_df[ncoords],
                                          sheet.je_df[dcoords])

    cell_pos = sheet.upcast_cell(sheet.cell_df[coords])
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords])
    r_aj = trgt_pos - cell_pos

    cross_aij_aj = np.cross(sheet.je_df[ncoords], r_aj)

    grad_a = _to_3d(inv_area) * (cross_aij_ij - cross_aij_aj)
    #grad_a = sheet.upcast_cell(grad_a_).sum(level='cell')

    return grad_a


def volume_grad(sheet, coords=None):
    ''' Computes
    :math:`\sum_\alpha\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
    '''
    if coords is None:
        coords = sheet.coords

    # volumic elastic force
    # this is K * (V - V0)
    kv_v0_ = elastic_force(sheet.cell_df,
                           var='vol',
                           elasticity='vol_elasticity',
                           prefered='prefered_vol')

    kv_v0_ = (kv_v0_ * sheet.cell_df['is_alive'])
    kv_v0 = _to_3d(sheet.upcast_cell(kv_v0_))

    je_h = _to_3d(sheet.upcast_srce(sheet.jv_df['height']))
    area_ = sheet.je_df['sub_area']
    area = _to_3d(area_)
    grad_v = kv_v0 * (
        je_h * area_grad(sheet, coords) +
        area * height_grad(sheet, coords)
        )
    return grad_v
