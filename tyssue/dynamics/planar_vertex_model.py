# from ..utils.utils import _to_2d, _to_3d

# from copy import deepcopy

# from .base_gradients import length_grad
# from .planar_gradients import area_grad
# from .effectors import elastic_force, elastic_energy

from . import effectors
from .factory import model_factory

PlanarModel = model_factory(
    [effectors.LineTension,
     effectors.FaceContractility,
     effectors.FaceAreaElasticity],
    effectors.FaceAreaElasticity)



# class PlanarModel():
#     """
#     Model for a 2D junction network  in 2D.

#     """
#     energy_labels = ['tension', 'contractility', 'area']

#     @staticmethod
#     def dimentionalize(mod_specs, **kwargs):
#         """
#         Changes the values of the input gamma and lambda parameters
#         from the values of the prefered height and area.
#         Computes the norm factor.
#         """

#         dim_mod_specs = deepcopy(mod_specs)
#         dim_mod_specs.update(**kwargs)

#         Kv = dim_mod_specs['face']['area_elasticity']
#         A0 = dim_mod_specs['face']['prefered_area']
#         gamma = dim_mod_specs['face']['contractility']

#         dim_mod_specs['face']['contractility'] = gamma * Kv*A0

#         dim_mod_specs['face']['prefered_area'] = A0

#         lbda = dim_mod_specs['edge']['line_tension']
#         dim_mod_specs['edge']['line_tension'] = lbda * Kv * A0**1.5

#         dim_mod_specs['settings']['grad_norm_factor'] = Kv * A0**1.5
#         dim_mod_specs['settings']['nrj_norm_factor'] = Kv * A0**2

#         if 'anchor_tension' in dim_mod_specs['edge']:
#             t_a = dim_mod_specs['edge']['anchor_tension']
#             dim_mod_specs['edge']['anchor_tension'] = t_a * Kv * A0**1.5

#         return dim_mod_specs

#     @staticmethod
#     def compute_energy(sheet, full_output=False):
#         '''
#         Computes the tissue sheet mesh energy.

#         Parameters
#         ----------
#         * sheet: a 2D :class:`tyssue.obedgect.sheet.Sheet` instance
#         * full_output: if True, returns the enery components
#         '''
#         # consider only live faces:
#         live_face_df = sheet.face_df[sheet.face_df.is_alive == 1]
#         upcast_alive = sheet.upcast_face(sheet.face_df.is_alive)
#         live_edge_df = sheet.edge_df[upcast_alive == 1]

#         E_t = live_edge_df.eval('line_tension * length / 2')
#         E_a = elastic_energy(live_face_df,
#                              var='area',
#                              elasticity='area_elasticity',
#                              prefered='prefered_area')
#         E_c = live_face_df.eval('0.5 * contractility * perimeter ** 2')
#         nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']

#         if full_output:
#             return (E / nrj_norm_factor for E in (E_t, E_c, E_a))
#         else:
#             return (E_t.sum() + (E_c + E_a).sum()) / nrj_norm_factor

#     @classmethod
#     def compute_gradient(cls, sheet, components=False):
#         '''
#         If components is True, returns the individual terms
#         (grad_t, grad_c, grad_v)
#         '''
#         norm_factor = sheet.specs['settings']['nrj_norm_factor']
#         grad_lij = length_grad(sheet)

#         grad_t = cls.tension_grad(sheet, grad_lij)
#         grad_c = cls.contractile_grad(sheet, grad_lij)
#         grad_a_srce, grad_a_trgt = cls.elastic_grad(sheet)
#         grad_i = (sheet.sum_srce(grad_t + grad_c + grad_a_srce)
#                   + sheet.sum_trgt(grad_a_trgt - grad_c))
#         if components:
#             return grad_t, grad_c, grad_a_srce, grad_a_trgt
#         return grad_i / norm_factor

#     @staticmethod
#     def tension_grad(sheet, grad_lij):

#         live_edge = sheet.upcast_face(sheet.face_df['is_alive'])
#         if len(sheet.coords) == 2:
#             grad_t = (grad_lij *
#                       _to_2d(sheet.edge_df['line_tension'] * live_edge))
#         elif len(sheet.coords) == 3:
#             grad_t = (grad_lij *
#                       _to_3d(sheet.edge_df['line_tension'] * live_edge))

#         return grad_t

#     @staticmethod
#     def contractile_grad(sheet, grad_lij):

#         gamma_ = sheet.face_df.eval('contractility * perimeter * is_alive')
#         gamma = sheet.upcast_face(gamma_)
#         if len(sheet.coords) == 2:
#             grad_c = grad_lij * _to_2d(gamma)
#         elif len(sheet.coords) == 3:
#             grad_c = grad_lij * _to_3d(gamma)

#         return grad_c

#     @staticmethod
#     def elastic_grad(sheet):
#         ''' Computes
#         :math:`\nabla_i \left(K (A_\alpha - A_0)^2\right)`:
#         '''
#         coords = sheet.coords

#         # volumic elastic force
#         # this is K * (A - A0)
#         ka_a0_ = elastic_force(sheet.face_df,
#                                var='area',
#                                elasticity='area_elasticity',
#                                prefered='prefered_area')

#         ka_a0_ = ka_a0_ * sheet.face_df['is_alive']
#         if len(coords) == 2:
#             ka_a0 = _to_2d(sheet.upcast_face(ka_a0_))
#         elif len(coords) == 3:
#             ka_a0 = _to_3d(sheet.upcast_face(ka_a0_))
#         grad_a_srce, grad_a_trgt = area_grad(sheet)
#         grad_a_srce = ka_a0 * grad_a_srce
#         grad_a_trgt = ka_a0 * grad_a_trgt

#         return grad_a_srce, grad_a_trgt

#     @staticmethod
#     def relax_anchors(sheet):
#         '''reset the anchor positions of the border vertices
#         to the positions of said vertices.

#         '''
#         at_border = sheet.vert_df[sheet.vert_df['at_border'] == 1].index
#         anchors = sheet.vert_df[sheet.vert_df['is_anchor'] == 1].index
#         sheet.vert_df.loc[
#             anchors, sheet.cords] = sheet.vert_df.loc[at_border,
#                                                       sheet.coords]

#     @staticmethod
#     def anchor_grad(sheet, grad_lij):

#         ka = sheet.edge_df.eval('anchor_tension * is_anchor')
#         return grad_lij * _to_3d(ka)
