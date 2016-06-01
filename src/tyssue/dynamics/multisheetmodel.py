
from copy import deepcopy

from tyssue.dynamics.sheet_vertex_model import SheetModel
from tyssue.dynamics.planar_vertex_model import PlanarModel
from tyssue.dynamics.sheet_gradients import area_grad
from tyssue.dynamics.base_gradients import length_grad
from tyssue.dynamics.effectors import elastic_force
from tyssue.utils.utils import _to_3d
import pandas as pd


class MultiSheetModel(SheetModel):

    @staticmethod
    def dimentionalize(mod_specs):
        """
        Changes the values of the input gamma and lambda parameters
        from the values of the prefered height and area.
        Computes the norm factor.
        """

        dim_mod_specs = deepcopy(mod_specs)

        Kv = dim_mod_specs['face']['area_elasticity']
        A0 = dim_mod_specs['face']['prefered_area']
        gamma = dim_mod_specs['face']['contractility']
        kappa_d = dim_mod_specs['vert']['d_elasticity']

        dim_mod_specs['face']['contractility'] = gamma * Kv * A0

        lbda = dim_mod_specs['edge']['line_tension']
        dim_mod_specs['edge']['line_tension'] = lbda * Kv * A0**1.5

        dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5
        dim_mod_specs['settings']['nrj_norm_factor'] = Kv * A0**2

        dim_mod_specs['vert']['d_elasticity'] = kappa_d * Kv * A0

        if 'anchor_tension' in dim_mod_specs['edge']:
            t_a = dim_mod_specs['edge']['anchor_tension']
            dim_mod_specs['edge']['anchor_tension'] = t_a * Kv * A0**1.5
        return dim_mod_specs

    @classmethod
    def compute_energy(cls, msheet, full_output=False):

        E = 0
        for sheet in msheet:
            E += PlanarModel.compute_energy(sheet)
        E += cls.desmosome_energy(msheet)
        return E

    @staticmethod
    def desmosome_energy(msheet):

        base_sheet = msheet[0]
        norm_factor = base_sheet.specs['settings']['nrj_norm_factor']
        basal = base_sheet.vert_df.eval(
            '0.5 * d_elasticity * ((z - basal_shift) - prefered_height)**2')
        E_d = basal.sum()/norm_factor

        for sheet in msheet[1:]:
            norm_factor = sheet.specs['settings']['nrj_norm_factor']
            upward = sheet.vert_df.eval(
                '0.5 * d_elasticity * (height - prefered_height)**2')
            E_d += upward.sum()/norm_factor

        for sheet in msheet[:-1]:
            norm_factor = sheet.specs['settings']['nrj_norm_factor']
            downward = sheet.vert_df.eval(
                '0.5 * d_elasticity * (depth - prefered_height)**2')
            E_d += downward.sum()/norm_factor

        return E_d

    @classmethod
    def compute_gradient(cls, msheet, components=False):

        grads = [pd.DataFrame(0.0, index=sheet.vert_df.index,
                              columns=sheet.coords)
                 for sheet in msheet]

        for sheet, (i, grad) in zip(msheet, enumerate(grads)):
            norm_factor = sheet.specs['settings']['nrj_norm_factor']

            grad_lij = length_grad(sheet)
            grad_t = PlanarModel.tension_grad(sheet, grad_lij)
            grad_c = PlanarModel.contractile_grad(sheet, grad_lij)
            grad_a_srce, grad_a_trgt = cls.elastic_grad(sheet)
            grad_i = ((sheet.sum_srce(grad_t) - sheet.sum_trgt(grad_t))/2 +
                      sheet.sum_srce(grad_c) - sheet.sum_trgt(grad_c) +
                      sheet.sum_srce(grad_a_srce) +
                      sheet.sum_trgt(grad_a_trgt))
            grad += grad_i
            grads[i] = grad/norm_factor

        cls.desmosome_gradient(msheet, grads)
        return grads

    @staticmethod
    def desmosome_gradient(msheet, grads):

        base_sheet = msheet[0]
        grads[0]['z'] += base_sheet.vert_df.eval(
            'd_elasticity * ((z - basal_shift) - prefered_height)')

        for i, sheet in enumerate(msheet[1:]):
            norm_factor = sheet.specs['settings']['nrj_norm_factor']
            upward = sheet.vert_df.eval(
                'd_elasticity * (height - prefered_height)')
            grads[i+1]['z'] += upward/norm_factor

        for i, sheet in enumerate(msheet[:-1]):
            norm_factor = sheet.specs['settings']['nrj_norm_factor']
            downward = sheet.vert_df.eval(
                'd_elasticity * (prefered_height - depth)')
            grads[i]['z'] += downward/norm_factor
        # return grads

    @staticmethod
    def elastic_grad(sheet):
        ''' Computes
        :math:`\nabla_i \left(K (A_\alpha - A_0)^2\right)`:
        '''

        # volumic elastic force
        # this is K * (A - A0)
        ka_a0_ = elastic_force(sheet.face_df,
                               var='area',
                               elasticity='area_elasticity',
                               prefered='prefered_area')

        ka_a0_ = ka_a0_ * sheet.face_df['is_alive']
        ka_a0 = _to_3d(sheet.upcast_face(ka_a0_))
        grad_a_srce, grad_a_trgt = area_grad(sheet)
        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt

        return grad_a_srce, grad_a_trgt
