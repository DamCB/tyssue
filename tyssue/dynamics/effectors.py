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
    dimensions = None
    magnitude = None
    spatial_ref = None, None
    temporal_ref = None, None

    label = 'Abstract effector'
    element = None  # cell, face, edge or vert
    specs = {}

    @staticmethod
    def energy(eptm):
        raise NotImplemented

    @staticmethod
    def gradient(eptm):
        raise NotImplemented

    @staticmethod
    def get_nrj_norm(specs):
        raise NotImplemented



class LengthElasticity(AbstractEffector):

    dimensions = units.line_elasticity
    label = 'Length elasticity'
    magnitude = 'length_elasticity'
    element = 'edge'
    spatial_ref = 'prefered_length', units.length

    specs = {'is_active',
             'length',
             'length_elasticity',
             'prefered_length'}

    @staticmethod
    def get_nrj_norm(specs):
        return (specs['edge']['length_elasticity']
                * specs['edge']['prefered_length']**2)

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

    dimensionless = False
    dimensions = units.area_elasticity
    magnitude = 'area_elasticity'
    label = 'Area elasticity'
    element = 'face'
    specs = {'is_alive',
             'area',
             'area_elasticity',
             'prefered_area'}

    spatial_ref = 'prefered_area', units.area

    @staticmethod
    def get_nrj_norm(specs):
        return (specs['face']['area_elasticity']
                * specs['face']['prefered_area']**2)

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

    dimensions = units.vol_elasticity
    magnitude = 'vol_elasticity'
    label = 'Volume elasticity'
    element = 'face'
    specs = {'is_alive',
             'vol',
             'vol_elasticity',
             'prefered_vol'}

    spatial_ref = 'prefered_vol', units.vol

    @staticmethod
    def get_nrj_norm(specs):
        return (specs['face']['vol_elasticity']
                * specs['face']['prefered_vol']**2)

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

    dimensions = units.area_elasticity
    magnitude = 'area_elasticity'
    label = 'Area elasticity'
    element = 'cell'
    specs = {'is_alive',
             'area',
             'area_elasticity',
             'prefered_area'}

    spatial_ref = 'prefered_area', units.area

    @staticmethod
    def get_nrj_norm(specs):
        return (specs['cell']['area_elasticity']
                * specs['cell']['prefered_area']**2)

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

    dimensions = units.vol_elasticity
    magnitude = 'vol_elasticity'
    label = 'Volume elasticity'
    element = 'cell'
    spatial_ref = 'prefered_vol', units.vol

    specs = {'is_alive',
             'vol',
             'vol_elasticity',
             'prefered_vol'}

    @staticmethod
    def get_nrj_norm(specs):
        return (specs['cell']['vol_elasticity']
                * specs['cel']['prefered_vol']**2)

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

    dimensions = units.line_tension
    magnitude = 'line_tension'
    label = 'Line tension'
    element = 'edge'
    specs = {'is_active',
             'line_tension'}

    spatial_ref = 'mean_length', units.length


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

    dimensions = units.line_elasticity
    magnitude = 'contractility'
    label = 'Contractility'
    element = 'face'
    specs = {'is_alive',
             'perimeter',
             'contractility'}

    spatial_ref = 'mean_perimeter', units.length

    @staticmethod
    def energy(eptm):
        return eptm.face_df.eval('0.5 * is_alive'
                                 '* contractility'
                                 '* perimeter ** 2')

    @staticmethod
    def gradient(eptm):

        gamma_ = eptm.face_df.eval('contractility * perimeter * is_alive')
        gamma = eptm.upcast_face(gamma_)

        grad_srce = - eptm.edge_df[eptm.ucoords] * to_nd(gamma,
                                                         len(eptm.coords))
        grad_srce.columns = ['g'+u for u in eptm.coords]
        grad_trgt = - grad_srce
        return grad_srce, grad_trgt


class SurfaceTension(AbstractEffector):

    dimensions = units.area_tension
    magnitude = 'surf_tension'

    spatial_ref = 'prefered_area', units.area

    label = 'Surface tension'
    element = 'face'
    specs = {'is_active',
             'surf_tension'}



class LineViscosity(AbstractEffector):

    dimensions = units.line_viscosity
    magnitude = 'edge_viscosity'

    label = 'Linear viscosity'
    element = 'edge'
    spatial_ref = 'mean_length', units.length
    temporal_ref = 'dt', units.time
    specs = {'is_active',
             'edge_viscosity'}

    @staticmethod
    def gradient(eptm):
        grad_srce = eptm.edge_df[['vx', 'vy', 'vz']] * to_nd(
            eptm.edge_df['edge_viscosity'], len(eptm.coords))
        grad_srce.columns = ['g'+u for u in eptm.coords]
        return grad_srce, None


def _exponants(dimensions, ref_dimensions,
               spatial_unit=None,
               temporal_unit=None):

    spatial_exponant = time_exponant = 0
    rel_dimensionality = (dimensions/ref_dimensions).dimensionality

    if spatial_unit is not None:
        spatial_exponant = (
            rel_dimensionality.get(units.length, 0)
            / spatial_unit.dimensionality[units.length])

    if temporal_unit is not None:
        time_exponant = (
            rel_dimensionality.get(units.time, 0)
            / temporal_unit.dimensionality[units.time])
    print(spatial_exponant, time_exponant)
    return spatial_exponant, time_exponant


def scaler(nondim_specs, dim_specs,
           effector, ref_effector):
    spatial_val, spatial_unit = ref_effector.spatial_ref
    temporal_val, temporal_unit = ref_effector.temporal_ref

    s_expo, t_expo = _exponants(effector.dimensions,
                                ref_effector.dimensions,
                                spatial_unit,
                                temporal_unit)

    ref_magnitude = ref_effector.magnitude
    ref_element = ref_effector.element
    factor = (dim_specs[ref_element][ref_magnitude]
              * dim_specs[ref_element].get(spatial_val, 1)**s_expo
              * dim_specs[ref_element].get(temporal_val, 1)**t_expo)
    return factor


def dimensionalize(nondim_specs, dim_specs,
                   effector, ref_effector):
    magnitude = effector.magnitude
    element = effector.element
    factor = scaler(nondim_specs, dim_specs,
                    effector, ref_effector)
    dim_specs[element][magnitude] = factor * nondim_specs[element][magnitude]
    return dim_specs


def normalize(dim_specs, nondim_specs,
              effector, ref_effector):
    magnitude = effector.magnitude
    element = effector.element
    factor = scaler(nondim_specs, dim_specs,
                    effector, ref_effector)
    dim_specs[element][magnitude] = nondim_specs[element][magnitude] / factor
    return dim_specs
