'''
Generic forces and energies
'''
from ..utils import to_nd
from . import units

from .planar_gradients import area_grad as area_grad2d
from .sheet_gradients import height_grad, area_grad
from .bulk_gradients import volume_grad


def elastic_force(element_df, var, elasticity, prefered):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    force = element_df.eval('{K} * ({x} - {x0})'.format(**params))
    return force


def elastic_energy(element_df, var, elasticity, prefered):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    energy = element_df.eval(
        '0.5 * {K} * ({x} - {x0}) ** 2'.format(**params))
    return energy


class AbstractEffector:
    """ The effector class is used by model factories
    to construct a model.


    """
    dimentionless = False
    dimentions = None
    label = 'Abstract effector'
    element = None  # cell, face, edge or vert
    specs = {}

    @staticmethod
    def energy(eptm):
        raise NotImplemented

    @staticmethod
    def gradient(eptm):
        raise NotImplemented


class LengthElasticity(AbstractEffector):

    dimentionless = False
    dimentions = units.line_elasticity
    label = 'Length elasticity'
    element = 'edge'
    specs = {'is_active',
             'length',
             'length_elasticity',
             'prefered_length'}

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.edge_df,
                              'length',
                              'length_elasticity * is_active / 2',
                              'prefered_length')

    @staticmethod
    def gradient(eptm):
        kl_l0 = elastic_force(eptm.edge_df,
                              'length',
                              'length_elasticity * is_active / 2',
                              'prefered_length')
        grad = eptm.edge_df[eptm.ucoords] * to_nd(kl_l0, len(eptm.coords))
        grad.columns = ['g'+u for u in eptm.coords]
        return grad, None


class FaceAreaElasticity(AbstractEffector):

    dimentionless = False
    dimentions = units.area_elasticity
    label = 'Area elasticity'
    element = 'face'
    specs = {'is_alive',
             'area',
             'area_elasticity',
             'prefered_area'}

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.face_df,
                              'area',
                              'area_elasticity * is_alive',
                              'prefered_area')

    @staticmethod
    def gradient(eptm):
        ka_a0_ = elastic_force(eptm.face_df,
                               'area',
                               'area_elasticity * is_alive',
                               'prefered_area')
        ka_a0 = to_nd(eptm.upcast_face(ka_a0_),
                      len(eptm.coords))

        if len(eptm.coords) == 2:
            grad_a_srce, grad_a_trgt = area_grad2d(eptm)
        elif len(eptm.coords) == 3:
            grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt

        grad_a_srce.columns = ['g'+u for u in eptm.coords]
        grad_a_trgt.columns = ['g'+u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


class FaceVolumeElasticity(AbstractEffector):
    dimentionless = False
    dimentions = units.vol_elasticity
    label = 'Volume elasticity'
    element = 'face'
    specs = {'is_alive',
             'vol',
             'vol_elasticity',
             'prefered_vol'}

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.face_df,
                              'vol',
                              'vol_elasticity * is_alive',
                              'prefered_vol')

    @staticmethod
    def gradient(eptm):
        kv_v0_ = elastic_force(eptm.face_df,
                               'vol',
                               'vol_elasticity * is_alive',
                               'prefered_vol')

        kv_v0 = to_nd(eptm.upcast_face(kv_v0_), 3)

        edge_h = to_nd(eptm.upcast_srce(eptm.vert_df['height']), 3)
        area_ = eptm.edge_df['sub_area']
        area = to_nd(area_, 3)
        grad_a_srce, grad_a_trgt = area_grad(eptm)
        grad_h = eptm.upcast_srce(height_grad(eptm))

        grad_v_srce = kv_v0 * (edge_h * grad_a_srce +
                               area * grad_h)
        grad_v_trgt = kv_v0 * (edge_h * grad_a_trgt)

        grad_v_srce.columns = ['g'+u for u in eptm.coords]
        grad_v_trgt.columns = ['g'+u for u in eptm.coords]

        return grad_v_srce, grad_v_trgt


class CellAreaElasticity(AbstractEffector):

    dimentionless = False
    dimentions = units.area_elasticity
    label = 'Area elasticity'
    element = 'cell'
    specs = {'is_alive',
             'area',
             'area_elasticity',
             'prefered_area'}

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.cell_df,
                              'area',
                              'area_elasticity',
                              'prefered_area')

    @staticmethod
    def gradient(eptm):
        ka_a0_ = elastic_force(eptm.cell_df,
                               'area',
                               'area_elasticity * is_alive',
                               'prefered_area')

        ka_a0 = to_nd(eptm.upcast_cell(ka_a0_), 3)

        grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt
        grad_a_srce.columns = ['g'+u for u in eptm.coords]
        grad_a_trgt.columns = ['g'+u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


class CellVolumeElasticity(AbstractEffector):

    dimentionless = False
    dimentions = units.vol_elasticity
    label = 'Volume elasticity'
    element = 'cell'
    specs = {'is_alive',
             'vol',
             'vol_elasticity',
             'prefered_vol'}

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.cell_df,
                              'vol',
                              'vol_elasticity',
                              'prefered_vol')

    @staticmethod
    def gradient(eptm):
        kv_v0_ = elastic_force(eptm.cell_df,
                               'vol',
                               'vol_elasticity * is_alive',
                               'prefered_vol')

        kv_v0 = to_nd(eptm.upcast_cell(kv_v0_), 3)
        grad_v_srce, grad_v_trgt = volume_grad(eptm)
        grad_v_srce = kv_v0 * grad_v_srce
        grad_v_trgt = kv_v0 * grad_v_trgt

        grad_v_srce.columns = ['g'+u for u in eptm.coords]
        grad_v_trgt.columns = ['g'+u for u in eptm.coords]

        return grad_v_srce, grad_v_trgt


class LineTension(AbstractEffector):

    dimentionless = False
    dimentions = units.line_tension
    label = 'Line tension'
    element = 'edge'
    specs = {'is_active',
             'line_tension'}

    @staticmethod
    def energy(eptm):
        return eptm.edge_df.eval('line_tension'
                                 '* is_active'
                                 '* length / 2')  # accounts for half edges

    @staticmethod
    def gradient(eptm):
        grad_srce = - eptm.edge_df[eptm.ucoords] * to_nd(
            eptm.edge_df.eval('line_tension * is_active'), len(eptm.coords))
        grad_srce.columns = ['g'+u for u in eptm.coords]
        return grad_srce, None


class FaceContractility(AbstractEffector):

    dimentionless = False
    dimentions = units.line_elasticity
    label = 'Contractility'
    element = 'face'
    specs = {'is_alive',
             'perimeter',
             'contractility'}

    @staticmethod
    def energy(eptm):
        return eptm.face_df.eval('0.5 * is_alive'
                                 '* contractility'
                                 '* perimeter ** 2')

    @classmethod
    def gradient(cls, eptm):

        gamma_ = eptm.face_df.eval('contractility * perimeter * is_alive')
        gamma = eptm.upcast_face(gamma_)

        grad_srce = - eptm.edge_df[eptm.ucoords] * to_nd(gamma,
                                                         len(eptm.coords))
        grad_srce.columns = ['g'+u for u in eptm.coords]
        grad_trgt = - grad_srce
        return grad_srce, grad_trgt


class SurfaceTension(AbstractEffector):

    dimentionless = False
    dimentions = units.area_tension
    label = 'Surface tension'
    element = 'face'
    specs = {'is_active',
             'line_tension'}

    @staticmethod
    def energy(eptm):
        raise NotImplemented

    @staticmethod
    def gradient(eptm):
        raise NotImplemented
