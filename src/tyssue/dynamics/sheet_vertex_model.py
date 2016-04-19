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
        upcast_alive = sheet.upcast_face(sheet.face_df.is_alive)
        live_edge_df = sheet.edge_df[upcast_alive == 1]

        E_t = live_edge_df.eval('line_tension * length / 2')
        E_v = elastic_energy(live_face_df,
                             var='vol',
                             elasticity='vol_elasticity',
                             prefered='prefered_vol')
        E_c = live_face_df.eval('0.5 * contractility * perimeter ** 2')
        if full_output:
            nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']

            return (E / nrj_norm_factor for E in (E_t, E_c, E_v))
        else:
            return (E_t.sum() + (E_c+E_v).sum()) / nrj_norm_factor

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
        if components:
            return grad_t, grad_c, grad_v_srce, grad_v_trgt

        grad_i = (
            (sheet.sum_srce(grad_t) - sheet.sum_trgt(grad_t))/2 +
            sheet.sum_srce(grad_c) - sheet.sum_trgt(grad_c) +
            sheet.sum_srce(grad_v_srce) + sheet.sum_trgt(grad_v_trgt)
            ) * _to_3d(sheet.vert_df.is_active)
        return grad_i / norm_factor

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
