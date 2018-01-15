"""Deprecated module to represent cell-cell interaction.

Usage is not recommended
"""


import numpy as np

from copy import deepcopy
from ..utils.utils import _to_3d, _to_2d
from .effectors import elastic_force, elastic_energy



def length_grad(ccmesh):
    '''returns -(dx/l, dy/l, dz/l), ie grad_i(l_ij))
    '''
    if ccmesh.dim == 2:
        grad_lij = - (ccmesh.cc_df[ccmesh.dcoords] /
                      _to_2d(ccmesh.cc_df['length']))
    elif ccmesh.dim == 3:
        grad_lij = - (ccmesh.cc_df[ccmesh.dcoords] /
                      _to_3d(ccmesh.cc_df['length']))
    grad_lij.columns = ccmesh.coords
    return grad_lij


def dimentionalize(mod_specs, **kwargs):
    """
    Changes the values of the input gamma and lambda parameters
    from the values of the prefered height and area.
    Computes the norm factor.
    """

    dim_mod_specs = deepcopy(mod_specs)
    dim_mod_specs.update(**kwargs)
    K = dim_mod_specs['cc']['elasticity']
    l0 = dim_mod_specs['cc']['prefered_length']
    dim_mod_specs['settings']['grad_norm_factor'] = K * l0
    dim_mod_specs['settings']['nrj_norm_factor'] = K * l0**2
    return dim_mod_specs


def compute_energy(ccmesh, full_output=False):

    nrj_norm_factor = ccmesh.specs['settings']['nrj_norm_factor']

    upcast_alive = ccmesh.upcast_srce(ccmesh.cell_df.is_alive)
    live_cc_df = ccmesh.cc_df[upcast_alive == 1]
    energy = elastic_energy(live_cc_df, var='length', elasticity='elasticity',
                            prefered='prefered_length')
    if full_output:
        return energy

    return energy.sum() / nrj_norm_factor

def compute_gradient(ccmesh, components=False):

    nrj_norm_factor = ccmesh.specs['settings']['nrj_norm_factor']
    upcast_alive = ccmesh.upcast_srce(ccmesh.cell_df.is_alive)
    grad_lij = length_grad(ccmesh)
    kl_l0 = elastic_force(ccmesh.cc_df, var='length',
                          elasticity='elasticity',
                          prefered='prefered_length')

    grad = _to_3d(kl_l0 * upcast_alive) * grad_lij
    if components:
        return grad
    return ccmesh.sum_srce(grad) / nrj_norm_factor
