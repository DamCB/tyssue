"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""

import numpy as np

from copy import deepcopy

from .base_gradients import length_grad
from .sheet_gradients import height_grad, area_grad
from .effectors import elastic_force, elastic_energy
from .planar_vertex_model import PlanarModel

from ..utils.utils import _to_3d


class SheetModel(PlanarModel):
    """
    Model for a 2D junction network  in 3D, aka 2.5D

    """
    energy_labels = ['tension', 'contractility', 'volume']


    @staticmethod
    def dimentionalize(mod_specs):
        """
        Changes the values of the input gamma and lambda parameters
        from the values of the prefered height and area.
        Computes the norm factor.
        """

        dim_mod_specs = deepcopy(mod_specs)

        Kv = dim_mod_specs['face']['vol_elasticity']
        A0 = dim_mod_specs['face']['prefered_area']
        h0 = dim_mod_specs['face']['prefered_height']
        gamma = dim_mod_specs['face']['contractility']

        dim_mod_specs['face']['contractility'] = gamma * Kv * A0 * h0**2
        dim_mod_specs['face']['prefered_vol'] = A0 * h0

        lbda = dim_mod_specs['edge']['line_tension']
        dim_mod_specs['edge']['line_tension'] = lbda * Kv * A0**1.5 * h0**2

        dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5 * h0**2
        dim_mod_specs['settings']['nrj_norm_factor'] = Kv * (A0*h0)**2

        if 'anchor_tension' in dim_mod_specs['edge']:
            t_a = dim_mod_specs['edge']['anchor_tension']
            dim_mod_specs['edge']['anchor_tension'] = (t_a * Kv *
                                                       A0**1.5 * h0**2)

        return dim_mod_specs

    @staticmethod
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
        upcast_active = sheet.upcast_srce(sheet.vert_df.is_active)
        upcast_alive = sheet.upcast_face(sheet.face_df.is_alive)
        sheet.edge_df['is_active'] = (upcast_alive.values &
                                      upcast_active.values).astype(np.bool)
        live_edge_df = sheet.edge_df[sheet.edge_df['is_active']]

        E_t = live_edge_df.eval('line_tension * length / 2')
        E_c = live_face_df.eval('0.5 * contractility * perimeter**2')
        E_v = elastic_energy(live_face_df,
                             var='vol',
                             elasticity='vol_elasticity',
                             prefered='prefered_vol')

        energies = (E_t, E_c, E_v)
        if 'is_anchor' in sheet.edge_df.columns:
            E_a = sheet.edge_df.eval(
                'anchor_tension * length * is_anchor'
                )
            energies = energies + (E_a,)

        nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']
        if full_output:
            return [E / nrj_norm_factor for E in energies]
        else:
            return sum(E.sum() for E in energies) / nrj_norm_factor

    @classmethod
    def compute_gradient(cls, sheet, components=False):
        '''
        If components is True, returns the individual terms
        (grad_t, grad_c, grad_v)
        '''
        norm_factor = sheet.specs['settings']['nrj_norm_factor']

        grad_lij = length_grad(sheet)

        grad_t = cls.tension_grad(sheet, grad_lij)
        grad_c = cls.contractile_grad(sheet, grad_lij)
        grad_v_srce, grad_v_trgt = cls.elastic_grad(sheet)
        grads = (grad_t, grad_c, grad_v_srce, grad_v_trgt)

        if 'is_anchor' in sheet.edge_df.columns:
            grad_a = cls.anchor_grad(sheet, grad_lij)
            grads = grads + (grad_a,)

        if components:
            return grads

        grad_i = (
            sheet.sum_srce(grad_t + grad_c + grad_v_srce) +
            sheet.sum_trgt(-grad_c + grad_v_trgt)
            )
        if 'is_anchor' in sheet.edge_df.columns:
            grad_i = grad_i + sheet.sum_srce(grad_a)

        return grad_i * _to_3d(sheet.vert_df.is_active) / norm_factor

    @staticmethod
    def elastic_grad(sheet):
        ''' Computes
        :math:`\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
        '''
        # volumic elastic force
        # this is K * (V - V0)
        kv_v0_ = elastic_force(sheet.face_df,
                               var='vol',
                               elasticity='vol_elasticity',
                               prefered='prefered_vol')

        kv_v0_ = kv_v0_ * sheet.face_df['is_alive']
        kv_v0 = _to_3d(sheet.upcast_face(kv_v0_))

        edge_h = _to_3d(sheet.upcast_srce(sheet.vert_df['height']))
        area_ = sheet.edge_df['sub_area']
        area = _to_3d(area_)
        grad_a_srce, grad_a_trgt = area_grad(sheet)
        grad_h = sheet.upcast_srce(height_grad(sheet))

        grad_v_srce = kv_v0 * (edge_h * grad_a_srce +
                               area * grad_h)
        grad_v_trgt = kv_v0 * (edge_h * grad_a_trgt)

        return grad_v_srce, grad_v_trgt
