"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""

import numpy as np

from copy import deepcopy

from .base_gradients import length_grad
from .sheet_gradients import height_grad, area_grad
from .planar_vertex_model import tension_grad, contractile_grad
from .effectors import elastic_force, elastic_energy

from ..utils.utils import _to_3d

def get_default_mod_specs():
    """
    Loads the default model specifications

    Returns
    -------
    defautl_mod_spec: dict, the default values for the model
      specifications
    """
    default_mod_specs = {
        "face": {
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
    """
    Changes the values of the input gamma and lambda parameters
    from the values of the prefered height and area.
    Computes the norm factor.
    """

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)

    Kv = dim_mod_specs['face']['vol_elasticity'][0]
    A0 = dim_mod_specs['face']['prefered_area'][0]
    h0 = dim_mod_specs['face']['prefered_height'][0]
    gamma = dim_mod_specs['face']['contractility'][0]

    dim_mod_specs['face']['contractility'] = (gamma * Kv*A0 * h0**2,
                                              np.float)

    dim_mod_specs['face']['prefered_vol'] = (A0 * h0, np.float)

    lbda = dim_mod_specs['je']['line_tension'][0]
    dim_mod_specs['je']['line_tension'] = (lbda * Kv * A0**1.5 * h0**2,
                                           np.float)

    dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5 * h0**2
    dim_mod_specs['settings']['nrj_norm_factor'] = Kv * (A0*h0)**2

    return dim_mod_specs


def compute_energy(sheet, full_output=False):
    '''
    Computes the tissue sheet mesh energy.

    Parameters
    ----------
    * mesh: a :class:`tyssue.object.sheet.Sheet` instance
    * full_output: if True, returns the enery components
    '''
    # consider only live faces:
    live_face_df = sheet.face_df[sheet.face_df.is_alive == 1]
    upcast_alive = sheet.upcast_face(sheet.face_df.is_alive)
    live_je_df = sheet.je_df[upcast_alive == 1]

    E_t = live_je_df.eval('line_tension * length / 2')
    E_v = elastic_energy(live_face_df,
                         var='vol',
                         elasticity='vol_elasticity',
                         prefered='prefered_vol')
    E_c = live_face_df.eval('0.5 * contractility * perimeter ** 2')
    if full_output:
        return (E / sheet.nrj_norm_factor for E in (E_t, E_c, E_v))
    else:
        return (E_t.sum() + (E_c+E_v).sum()) / sheet.nrj_norm_factor

def compute_gradient(sheet, components=False):
    '''
    If components is True, returns the individual terms
    (grad_t, grad_c, grad_v)
    '''

    norm_factor = sheet.nrj_norm_factor

    grad_lij = length_grad(sheet)

    grad_t = tension_grad(sheet, grad_lij)
    grad_c = contractile_grad(sheet, grad_lij)
    grad_v_srce, grad_v_trgt = elastic_grad(sheet)

    grad_i = ((sheet.sum_srce(grad_t) - sheet.sum_trgt(grad_t))/2 +
              sheet.sum_srce(grad_c) - sheet.sum_trgt(grad_c) +
              sheet.sum_srce(grad_v_srce) + sheet.sum_trgt(grad_v_trgt))
    if components:
        return grad_t, grad_c, grad_v_srce, grad_v_trgt
    return grad_i / norm_factor

def elastic_grad(sheet):
    ''' Computes
    :math:`\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
    '''
    coords = sheet.coords

    # volumic elastic force
    # this is K * (V - V0)
    kv_v0_ = elastic_force(sheet.face_df,
                           var='vol',
                           elasticity='vol_elasticity',
                           prefered='prefered_vol')

    kv_v0_ = kv_v0_ * sheet.face_df['is_alive']
    kv_v0 = _to_3d(sheet.upcast_face(kv_v0_))

    je_h = _to_3d(sheet.upcast_srce(sheet.jv_df['height']))
    area_ = sheet.je_df['sub_area']
    area = _to_3d(area_)
    grad_a_srce, grad_a_trgt = area_grad(sheet)

    grad_v_srce = kv_v0 * (je_h * grad_a_srce +
                           area * height_grad(sheet))
    grad_v_trgt = kv_v0 * (je_h * grad_a_trgt)

    return grad_v_srce, grad_v_trgt
