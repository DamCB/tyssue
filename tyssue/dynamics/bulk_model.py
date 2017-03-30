"""
Vertex model for an Epithelial sheet (see definitions).

Depends on the sheet vertex geometry functions.
"""
import numpy as np
from copy import deepcopy

from .base_gradients import length_grad
from .sheet_gradients import area_grad
from .bulk_gradients import volume_grad
from .effectors import elastic_force, elastic_energy
from ..utils.utils import _to_3d
from .sheet_vertex_model import SheetModel


class BulkModel(SheetModel):

    @staticmethod
    def dimentionalize(mod_specs):
        """
        Changes the values of the input gamma and lambda parameters
        from the values of the prefered height and area.
        Computes the norm factor.
        """

        dim_mod_specs = deepcopy(mod_specs)

        Kv = dim_mod_specs['cell']['vol_elasticity']
        V0 = dim_mod_specs['cell']['prefered_vol']
        gamma = dim_mod_specs['face']['contractility']
        Ka = dim_mod_specs['face']['area_elasticity']

        dim_mod_specs['face']['contractility'] = gamma * Kv * V0
        dim_mod_specs['face']['area_elasticity'] = Ka * Kv * V0**(2/3)

        lbda = dim_mod_specs['edge']['line_tension']
        dim_mod_specs['edge']['line_tension'] = lbda * Kv * V0**(5/3)

        dim_mod_specs['settings']['grad_norm_factor'] = Kv * V0**(5/3)
        dim_mod_specs['settings']['nrj_norm_factor'] = Kv * V0**2

        if 'anchor_tension' in dim_mod_specs['edge']:
            t_a = dim_mod_specs['edge']['anchor_tension']
            dim_mod_specs['edge']['anchor_tension'] = (t_a * Kv * V0**(5/3))

        return dim_mod_specs

    @staticmethod
    def compute_energy(eptm, full_output=False):
        '''
        Computes the tissue sheet mesh energy.

        Parameters
        ----------
        * eptm: a :class:`tyssue.object.Epithelium` instance
        * full_output: if True, returns the enery components
        '''

        E_t = eptm.edge_df.eval('line_tension * length / 2')
        E_c = eptm.face_df.eval(
            '0.5 * is_alive * contractility * perimeter**2')
        E_a = elastic_energy(eptm.face_df,
                             var='area',
                             elasticity='area_elasticity',
                             prefered='prefered_area')
        E_v = elastic_energy(eptm.cell_df,
                             var='vol',
                             elasticity='vol_elasticity',
                             prefered='prefered_vol')
        E_v *= eptm.cell_df['is_alive']
        E_a *= eptm.face_df['is_alive']
        energies = (E_t, E_c, E_a, E_v)
        if 'is_anchor' in eptm.edge_df.columns:
            E_anc = eptm.edge_df.eval(
                'anchor_tension * length * is_anchor'
                )
            energies = energies + (E_anc,)

        nrj_norm_factor = eptm.specs['settings']['nrj_norm_factor']
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
        grad_v_srce, grad_v_trgt = cls.elastic_grad_v(sheet)
        grad_a_srce, grad_a_trgt = cls.elastic_grad_a(sheet)
        grads = (grad_t, grad_c,
                 grad_v_srce, grad_v_trgt,
                 grad_a_srce, grad_a_trgt)

        if 'is_anchor' in sheet.edge_df.columns:
            grad_anc = cls.anchor_grad(sheet, grad_lij)
            grads = grads + (grad_anc,)

        if components:
            return grads

        grad_i = (
            sheet.sum_srce(grad_t/2 + grad_c +
                           grad_a_srce + grad_v_srce) +
            sheet.sum_trgt(grad_t/2 + grad_c +
                           grad_a_trgt + grad_v_trgt)
            )
        # grad_i = (
        #     (sheet.sum_srce(grad_t) - sheet.sum_trgt(grad_t))/2 +
        #     sheet.sum_srce(grad_c) - sheet.sum_trgt(grad_c) +
        #     sheet.sum_srce(grad_v_srce) + sheet.sum_trgt(grad_v_trgt) +
        #     sheet.sum_srce(grad_a_srce) + sheet.sum_trgt(grad_a_trgt))

        if 'is_anchor' in sheet.edge_df.columns:
            grad_i = grad_i + sheet.sum_srce(grad_anc)

        return grad_i * _to_3d(sheet.vert_df.is_active) / norm_factor

    @staticmethod
    def elastic_grad_v(eptm):
        ''' Computes
        :math:`\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
        '''
        # volumic elastic force
        # this is K * (V - V0)
        kv_v0_ = elastic_force(eptm.cell_df,
                               var='vol',
                               elasticity='vol_elasticity',
                               prefered='prefered_vol')

        kv_v0_ = kv_v0_ * eptm.cell_df['is_alive']
        kv_v0 = _to_3d(eptm.upcast_cell(kv_v0_))
        grad_v_srce, grad_v_trgt = volume_grad(eptm)
        grad_v_srce = kv_v0 * grad_v_srce
        grad_v_trgt = kv_v0 * grad_v_trgt

        return grad_v_srce, grad_v_trgt

    @staticmethod
    def elastic_grad_a(eptm):
        ''' Computes
        :math:`\nabla_i \left(K (V_\alpha - V_0)^2\right)`:
        '''
        # volumic elastic force
        # this is K * (V - V0)
        kv_a0_ = elastic_force(eptm.face_df,
                               var='area',
                               elasticity='area_elasticity',
                               prefered='prefered_area')

        kv_a0_ = kv_a0_ * eptm.face_df['is_alive']
        kv_a0 = _to_3d(eptm.upcast_face(kv_a0_))
        grad_a_srce, grad_a_trgt = area_grad(eptm)
        grad_a_srce = kv_a0 * grad_a_srce
        grad_a_trgt = kv_a0 * grad_a_trgt

        return grad_a_srce, grad_a_trgt


class LaminaModel(BulkModel):

    @classmethod
    def compute_energy(cls, eptm, full_output=False):
        E_b = BulkModel.compute_energy(eptm)
        lamina = eptm.edge_df.loc[eptm.lamina_edges]
        E_l = lamina.eval('0.5 * length_elasticity'
                          '* (length - prefered_length)**2').sum()
        return E_b + E_l

    @staticmethod
    def compute_gradient(eptm, components=False):
        grad = BulkModel.compute_gradient(eptm)
        lamina = eptm.edge_df.loc[eptm.lamina_edges]

        grad_lij = length_grad(eptm).loc[eptm.lamina_edges]
        kl_l0 = elastic_force(lamina,
                              var='length',
                              elasticity='length_elasticity',
                              prefered='prefered_length')
        grad_l_ = _to_3d(kl_l0) * grad_lij
        grad_l_ = grad_l_.set_index(lamina['srce'])

        grad_l_srce = grad_l_.sum(level='srce')
        grad_l_ = grad_l_.set_index(lamina['trgt'])
        grad_l_trgt = grad_l_.sum(level='trgt')
        grad_l = (grad_l_srce - grad_l_trgt)/2
        grad_l.replace(np.nan, 0, inplace=True)
        grad.loc[grad_l.index] += grad_l
        return grad


def set_model(eptm, model, apical_spec, modifiers):

    # apical_spec = model.dimentionalize(apical_spec)
    eptm.update_specs(apical_spec, reset=True)

    for segment, spec in modifiers.items():
        for element, parameters in spec.items():
            idx = eptm.segment_index(segment, element)
            for param_name, param_value in parameters.items():
                eptm.datasets[element].loc[idx,
                                           param_name] = \
                    param_value * eptm.specs[element][param_name]
