import numpy as np
import pandas as pd

from ..utils.utils import _to_3d

from copy import deepcopy

from .base_gradients import length_grad
from .planar_gradients import area_grad
from .effectors import elastic_force, elastic_energy


def get_default_mod_specs():
    """
    Loads the default model specifications

    Returns
    -------
    defautl_mod_spec: dict, the default values for the model
      specifications
    """
    default_mod_specs = {
        "cell": {
            "contractility": (0.04, np.float),
            "area_elasticity": (1., np.float),
            "prefered_area": (1., np.float),
            },
        "je": {
            "line_tension": (0.12, np.float),
            },
        "settings": {
            'grad_norm_factor': 1.,
            'nrj_norm_factor': 1.,
            }
        }
    return default_mod_specs

def dimentionalize(mod_specs, **kwargs):
    """
    Changes the values of the input gamma and lambda parameters
    from the values of the prefered height and area.
    Computes the norm factor.
    """

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)

    Kv = dim_mod_specs['cell']['area_elasticity'][0]
    A0 = dim_mod_specs['cell']['prefered_area'][0]
    gamma = dim_mod_specs['cell']['contractility'][0]

    dim_mod_specs['cell']['contractility'] = (gamma * Kv*A0,
                                              np.float)

    dim_mod_specs['cell']['prefered_area'] = (A0, np.float)

    lbda = dim_mod_specs['je']['line_tension'][0]
    dim_mod_specs['je']['line_tension'] = (lbda * Kv * A0**1.5,
                                           np.float)

    dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5
    dim_mod_specs['settings']['nrj_norm_factor'] = Kv * A0**2

    return dim_mod_specs

def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * sheet: a 2D :class:`tyssue.object.sheet.Sheet` instance
    * full_output: if True, returns the enery components
    '''
    # consider only live cells:
    live_cell_df = sheet.cell_df[sheet.cell_df.is_alive == 1]
    upcast_alive = sheet.upcast_cell(sheet.cell_df.is_alive)
    live_je_df = sheet.je_df[upcast_alive == 1]

    E_t = live_je_df.eval('line_tension * length / 2')
    E_v = elastic_energy(live_cell_df,
                         var='area',
                         elasticity='area_elasticity',
                         prefered='prefered_area')
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
    norm_factor = sheet.nrj_norm_factor

    grad_lij = length_grad(sheet)

    grad_t = tension_grad(sheet, grad_lij)
    grad_c = contractile_grad(sheet, grad_lij)
    grad_v_srce, grad_v_trgt = elastic_grad(sheet, sheet.coords)
    grad_i = ((grad_t.sum(level='srce') - grad_t.sum(level='trgt'))/2 +
              grad_c.sum(level='srce') - grad_c.sum(level='trgt') +
              grad_v_srce.sum(level='srce') + grad_v_trgt.sum(level='trgt'))
    if components:
        return grad_t, grad_c, grad_v_srce, grad_v_trgt
    return grad_i / norm_factor

def tension_grad(sheet, grad_lij):

    live_je = sheet.upcast_cell(sheet.cell_df['is_alive'])
    grad_t = (grad_lij
              * _to_3d(sheet.je_df['line_tension'] * live_je))

    #grad_t = _grad_t.sum(level='srce').loc[sheet.jv_idx]
    return grad_t


def contractile_grad(sheet, grad_lij):

    gamma_ = sheet.cell_df.eval('contractility * perimeter * is_alive')
    gamma = sheet.upcast_cell(gamma_)
    grad_c = grad_lij * _to_3d(gamma)
    return grad_c


def elastic_grad(sheet, coords=None):
    ''' Computes
    :math:`\nabla_i \left(K (A_\alpha - A_0)^2\right)`:
    '''
    if coords is None:
        coords = sheet.coords

    # volumic elastic force
    # this is K * (A - A0)
    ka_a0_ = elastic_force(sheet.cell_df,
                           var='area',
                           elasticity='area_elasticity',
                           prefered='prefered_area')

    ka_a0_ = ka_a0_ * sheet.cell_df['is_alive']
    ka_a0 = _to_3d(sheet.upcast_cell(ka_a0_))

    grad_a_srce, grad_a_trgt = area_grad(sheet, coords)
    grad_v_srce = ka_a0 * grad_a_srce
    grad_v_trgt = ka_a0 * grad_a_trgt

    return grad_v_srce, grad_v_trgt
