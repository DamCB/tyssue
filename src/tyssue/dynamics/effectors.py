"""
Generic forces and energies
"""
import pandas as pd
import numpy as np

from ..utils import to_nd
from . import units

from .planar_gradients import area_grad as area_grad2d
from .planar_gradients import lumen_area_grad
from .sheet_gradients import height_grad, area_grad
from .bulk_gradients import volume_grad, lumen_volume_grad


def elastic_force(element_df, var, elasticity, prefered):
    """
    K can be:
      - column name (str)
      - callable: f(df) -> array
    """
    if callable(elasticity):
        K_val = elasticity(element_df)
    else:
        K_val = element_df[elasticity]

    return K_val * (element_df[var] - element_df[prefered])


def _elastic_force(element_df, x, elasticity, prefered):
    force = element_df[elasticity] * (element_df[x] - element_df[prefered])
    return force


def elastic_energy(element_df, var, elasticity, prefered):
    """
    elasticity can be:
      - str: column name
      - callable: f(df) -> array-like
    """
    x = element_df[var]
    x0 = element_df[prefered]

    if callable(elasticity):
        K = elasticity(element_df)
    else:
        K = element_df[elasticity]

    energy = 0.5 * K * (x - x0) ** 2
    return energy


def _elastic_energy(element_df, x, elasticity, prefered):
    energy = 0.5 * element_df[elasticity] * (element_df[x] - element_df[prefered]) ** 2
    return np.array(energy)


class AbstractEffector:
    """ The effector class is used by model factories
    to construct a model.


    """

    dimensions = None
    magnitude = None
    spatial_ref = None, None
    temporal_ref = None, None

    label = "Abstract effector"
    element = None  # cell, face, edge or vert
    specs = {"cell": {}, "face": {}, "edge": {}, "vert": {}}

    @staticmethod
    def energy(eptm):
        raise NotImplementedError

    @staticmethod
    def gradient(eptm):
        raise NotImplementedError

    @staticmethod
    def get_nrj_norm(specs):
        raise NotImplementedError


#     @classmethod
#     @property
#     def __doc__(cls):
#         f"""Effector implementing {cls.label} with a magnitude factor
#  {cls.magnitude}.

# Works on an `Epithelium` object's {cls.element} elements.
# """


class LengthElasticity(AbstractEffector):
    """Elastic half edge
    """

    dimensions = units.line_elasticity
    label = "Length elasticity"
    magnitude = "length_elasticity"
    element = "edge"
    spatial_ref = "prefered_length", units.length

    specs = {
        "edge": {
            "is_active": 1,
            "length": 1.0,
            "length_elasticity": 1.0,
            "prefered_length": 1.0,
            "ux": (1 / 3) ** 0.5,
            "uy": (1 / 3) ** 0.5,
            "uz": (1 / 3) ** 0.5,
        }
    }

    @staticmethod
    def get_nrj_norm(specs):
        return (
            specs["edge"]["length_elasticity"] * specs["edge"]["prefered_length"] ** 2
        )

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.edge_df, "length", lambda df: df["length_elasticity"] * df["is_active"], "prefered_length"
        )

    @staticmethod
    def gradient(eptm):
        kl_l0 = elastic_force(
            eptm.edge_df, "length", lambda df: df["length_elasticity"] * df["is_active"], "prefered_length"
        )
        grad = eptm.edge_df[eptm.ucoords] * to_nd(kl_l0, eptm.dim)
        grad.columns = ["g" + u for u in eptm.coords]
        return -grad, grad


class PerimeterElasticity(AbstractEffector):
    """From Mapeng Bi et al. https://doi.org/10.1038/nphys3471
    """

    dimensions = units.line_elasticity
    magnitude = "perimeter_elasticity"
    label = "Perimeter Elasticity"
    element = "face"
    specs = {
        "face": {
            "is_alive": 1,
            "perimeter": 1.0,
            "perimeter_elasticity": 0.1,
            "prefered_perimeter": 3.81,
        }
    }

    spatial_ref = "prefered_perimeter", units.length

    @staticmethod
    def energy(eptm):
        df = eptm.face_df
        diff = df["perimeter"] - df["prefered_perimeter"]
        return 0.5 * df["is_alive"] * df["perimeter_elasticity"] * diff * diff

    @staticmethod
    def gradient(eptm):
        df = eptm.face_df
        gamma_ = df["perimeter_elasticity"] * df["is_alive"] * (df["perimeter"] - df["prefered_perimeter"])

        gamma = eptm.upcast_face(gamma_)

        gamma_nd = to_nd(gamma, len(eptm.coords))

        grad_srce = -eptm.edge_df[eptm.ucoords].to_numpy() * gamma_nd
        grad_srce = pd.DataFrame(grad_srce, columns=["g" + u for u in eptm.coords])

        grad_trgt = -grad_srce
        return grad_srce, grad_trgt


class FaceAreaElasticity(AbstractEffector):

    dimensionless = False
    dimensions = units.area_elasticity
    magnitude = "area_elasticity"
    label = "Area elasticity"
    element = "face"
    specs = {
        "face": {
            "is_alive": 1,
            "area": 1.0,
            "area_elasticity": 1.0,
            "prefered_area": 1.0,
        },
        "edge": {"sub_area": 1 / 6.0},
    }

    spatial_ref = "prefered_area", units.area

    @staticmethod
    def get_nrj_norm(specs):
        return specs["face"]["area_elasticity"] * specs["face"]["prefered_area"] ** 2

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.face_df, "area", lambda df: df["area_elasticity"] * df["is_alive"], "prefered_area"
        )

    @staticmethod
    def gradient(eptm):
        ka_a0_ = elastic_force(
            eptm.face_df, "area", lambda df: df["area_elasticity"] * df["is_alive"], "prefered_area"
        )
        ka_a0 = to_nd(eptm.upcast_face(ka_a0_), len(eptm.coords))

        if len(eptm.coords) == 2:
            grad_a_srce, grad_a_trgt = area_grad2d(eptm)
        elif len(eptm.coords) == 3:
            grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt

        grad_a_srce.columns = ["g" + u for u in eptm.coords]
        grad_a_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


class FaceVolumeElasticity(AbstractEffector):

    dimensions = units.vol_elasticity
    magnitude = "vol_elasticity"
    label = "Volume elasticity"
    element = "face"
    specs = {
        "face": {"is_alive": 1, "vol": 1.0, "vol_elasticity": 1.0, "prefered_vol": 1.0},
        "vert": {"height": 1.0},
        "edge": {"sub_area": 1 / 6},
    }

    spatial_ref = "prefered_vol", units.vol

    @staticmethod
    def get_nrj_norm(specs):
        return specs["face"]["vol_elasticity"] * specs["face"]["prefered_vol"] ** 2

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.face_df, "vol", lambda df: df["vol_elasticity"] * df["is_alive"], "prefered_vol"
        )

    @staticmethod
    def gradient(eptm):
        kv_v0_ = elastic_force(
            eptm.face_df, "vol", lambda df: df["vol_elasticity"] * df["is_alive"], "prefered_vol"
        )

        kv_v0 = to_nd(eptm.upcast_face(kv_v0_), 3)

        edge_h = to_nd(eptm.upcast_srce(eptm.vert_df["height"]), 3)
        area_ = eptm.edge_df["sub_area"]
        area = to_nd(area_, 3)
        grad_a_srce, grad_a_trgt = area_grad(eptm)
        grad_h = eptm.upcast_srce(height_grad(eptm))

        grad_v_srce = kv_v0 * (edge_h * grad_a_srce + area * grad_h)
        grad_v_trgt = kv_v0 * (edge_h * grad_a_trgt)

        grad_v_srce.columns = ["g" + u for u in eptm.coords]
        grad_v_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_v_srce, grad_v_trgt


class CellAreaElasticity(AbstractEffector):

    dimensions = units.area_elasticity
    magnitude = "area_elasticity"
    label = "Area elasticity"
    element = "cell"
    specs = {
        "cell": {
            "is_alive": 1,
            "area": 1.0,
            "area_elasticity": 1.0,
            "prefered_area": 1.0,
        }
    }
    spatial_ref = "prefered_area", units.area

    @staticmethod
    def get_nrj_norm(specs):
        return specs["cell"]["area_elasticity"] * specs["cell"]["prefered_area"] ** 2

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.cell_df, "area", "area_elasticity", "prefered_area")

    @staticmethod
    def gradient(eptm):
        ka_a0_ = elastic_force(
            eptm.cell_df, "area", lambda df: df["area_elasticity"] * df["is_alive"], "prefered_area"
        )

        ka_a0 = to_nd(eptm.upcast_cell(ka_a0_), 3)

        grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = ka_a0 * grad_a_srce
        grad_a_trgt = ka_a0 * grad_a_trgt
        grad_a_srce.columns = ["g" + u for u in eptm.coords]
        grad_a_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


class CellVolumeElasticity(AbstractEffector):

    dimensions = units.vol_elasticity
    magnitude = "vol_elasticity"
    label = "Volume elasticity"
    element = "cell"
    spatial_ref = "prefered_vol", units.vol

    specs = {
        "cell": {"is_alive": 1, "vol": 1.0, "vol_elasticity": 1.0, "prefered_vol": 1.0}
    }

    @staticmethod
    def get_nrj_norm(specs):
        return specs["cell"]["vol_elasticity"] * specs["cell"]["prefered_vol"] ** 2

    @staticmethod
    def energy(eptm):
        return elastic_energy(eptm.cell_df, "vol", "vol_elasticity", "prefered_vol")

    @staticmethod
    def gradient(eptm):
        kv_v0_ = elastic_force(
            eptm.cell_df, "vol", lambda df: df["vol_elasticity"] * df["is_alive"], "prefered_vol"
        )

        kv_v0 = to_nd(eptm.upcast_cell(kv_v0_), 3)
        grad_v_srce, grad_v_trgt = volume_grad(eptm)
        grad_v_srce = kv_v0 * grad_v_srce
        grad_v_trgt = kv_v0 * grad_v_trgt

        grad_v_srce.columns = ["g" + u for u in eptm.coords]
        grad_v_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_v_srce, grad_v_trgt


class LumenVolumeElasticity(AbstractEffector):
    """
    Global volume elasticity of the object.
    For example the volume of the yolk in the Drosophila embryo
    """

    dimensions = units.vol_elasticity
    magnitude = "lumen_vol_elasticity"
    label = "Lumen volume elasticity"
    element = "settings"
    spatial_ref = "lumen_prefered_vol", units.vol

    specs = {
        "settings": {
            "lumen_vol": 1.0,
            "lumen_vol_elasticity": 1.0,
            "lumen_prefered_vol": 1.0,
        }
    }

    @staticmethod
    def get_nrj_norm(specs):
        return (
            specs["settings"]["lumen_vol_elasticity"]
            * specs["settings"]["lumen_prefered_vol"] ** 2
        )

    @staticmethod
    def energy(eptm):

        return _elastic_energy(
            eptm.settings, "lumen_vol", "lumen_vol_elasticity", "lumen_prefered_vol"
        )

    @staticmethod
    def gradient(eptm):
        kv_v0 = _elastic_force(
            eptm.settings, "lumen_vol", "lumen_vol_elasticity", "lumen_prefered_vol"
        )

        grad_v_srce, grad_v_trgt = lumen_volume_grad(eptm)
        grad_v_srce = kv_v0 * grad_v_srce
        grad_v_trgt = kv_v0 * grad_v_trgt

        grad_v_srce.columns = ["g" + u for u in eptm.coords]
        grad_v_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_v_srce, grad_v_trgt


class LineTension(AbstractEffector):

    dimensions = units.line_tension
    magnitude = "line_tension"
    label = "Line tension"
    element = "edge"
    specs = {"edge": {"is_active": 1, "line_tension": 1.0}}

    spatial_ref = "mean_length", units.length

    @staticmethod
    def energy(eptm):
        df = eptm.edge_df
        return 0.5 * df["line_tension"] * df["is_active"] * df["length"]  # accounts for half edges

    @staticmethod
    def gradient(eptm):
        edge_df = eptm.edge_df

        coeff = (edge_df["line_tension"] * edge_df["is_active"]) * 0.5

        grad_srce = -edge_df[eptm.ucoords] * to_nd(
            coeff, len(eptm.coords)
        )

        grad_srce.columns = ["g" + u for u in eptm.coords]
        grad_trgt = -grad_srce
        return grad_srce, grad_trgt


class FaceContractility(AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = "contractility"
    label = "Contractility"
    element = "face"
    specs = {"face": {"is_alive": 1, "perimeter": 1.0, "contractility": 1.0}}

    spatial_ref = "mean_perimeter", units.length

    @staticmethod
    def energy(eptm):
        df = eptm.face_df
        return 0.5 * df["is_alive"] * df["contractility"] * df["perimeter"] * df["perimeter"]

    @staticmethod
    def gradient(eptm):
        df = eptm.face_df
        gamma_ = df["contractility"] * df["perimeter"] * df["is_alive"]

        gamma = eptm.upcast_face(gamma_)

        gamma_nd = to_nd(gamma, len(eptm.coords))

        grad_srce = -eptm.edge_df[eptm.ucoords].to_numpy() * gamma_nd
        grad_srce = pd.DataFrame(grad_srce, columns=["g" + u for u in eptm.coords])

        grad_trgt = -grad_srce
        return grad_srce, grad_trgt


class SurfaceTension(AbstractEffector):

    dimensions = units.area_tension
    magnitude = "surface_tension"

    spatial_ref = "prefered_area", units.area

    label = "Surface tension"
    element = "face"
    specs = {"face": {"is_active": 1, "surface_tension": 1.0, "area": 1.0}}

    @staticmethod
    def energy(eptm):
        df = eptm.face_df
        return df["surface_tension"] * df["area"]

    @staticmethod
    def gradient(eptm):

        G = to_nd(eptm.upcast_face(eptm.face_df["surface_tension"]), len(eptm.coords))
        grad_a_srce, grad_a_trgt = area_grad(eptm)

        grad_a_srce = G * grad_a_srce
        grad_a_trgt = G * grad_a_trgt
        grad_a_srce.columns = ["g" + u for u in eptm.coords]
        grad_a_trgt.columns = ["g" + u for u in eptm.coords]

        return grad_a_srce, grad_a_trgt


class LineViscosity(AbstractEffector):

    dimensions = units.line_viscosity
    magnitude = "edge_viscosity"

    label = "Linear viscosity"
    element = "edge"
    spatial_ref = "mean_length", units.length
    temporal_ref = "dt", units.time
    specs = {"edge": {"is_active": 1, "edge_viscosity": 1.0}}

    @staticmethod
    def gradient(eptm):
        grad_srce = eptm.edge_df[["vx", "vy", "vz"]] * to_nd(
            eptm.edge_df["edge_viscosity"], len(eptm.coords)
        )
        grad_srce.columns = ["g" + u for u in eptm.coords]
        return grad_srce, None


class BorderElasticity(AbstractEffector):
    dimensions = units.line_elasticity
    label = "Border edges elasticity"
    magnitude = "border_elasticity"
    element = "edge"
    spatial_ref = "prefered_length", units.length

    specs = {
        "edge": {
            "is_active": 1,
            "length": 1.0,
            "border_elasticity": 1.0,
            "prefered_length": 1.0,
            "is_border": 1.0,
        }
    }

    @staticmethod
    def get_nrj_norm(specs):
        return (
            specs["edge"]["border_elasticity"] * specs["edge"]["prefered_length"] ** 2
        )

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.edge_df,
            "length",
            lambda df: df["border_elasticity"] * df["is_active"] * df["is_border"]/2,
            "prefered_length",
        )

    @staticmethod
    def gradient(eptm):

        kl_l0 = elastic_force(
            eptm.edge_df,
            var="length",
            elasticity= lambda df: df["border_elasticity"] * df["is_active"] * df["is_border"],
            prefered="prefered_length",
        )
        grad = eptm.edge_df[eptm.ucoords] * to_nd(kl_l0, eptm.dim)
        grad.columns = ["g" + u for u in eptm.coords]
        return grad / 2, -grad / 2


class LumenAreaElasticity(AbstractEffector):

    dimensions = units.area_elasticity
    label = "Lumen volume constraint"
    magnitude = "lumen_elasticity"
    element = "settings"
    spatial_ref = "lumen_prefered_vol", units.area

    specs = {
        "settings": {
            "lumen_elasticity": 1.0,
            "lumen_prefered_vol": 1.0,
            "lumen_vol": 1.0,
        }
    }

    @staticmethod
    def energy(eptm):
        Ky = eptm.settings["lumen_elasticity"]
        V0 = eptm.settings["lumen_prefered_vol"]
        Vy = eptm.settings["lumen_vol"]
        return np.array([Ky * (Vy - V0) ** 2 / 2])

    @staticmethod
    def gradient(eptm):
        Ky = eptm.settings["lumen_elasticity"]
        V0 = eptm.settings["lumen_prefered_vol"]
        Vy = eptm.settings["lumen_vol"]
        grad_srce, grad_trgt = lumen_area_grad(eptm)
        return (Ky * (Vy - V0) * grad_srce, Ky * (Vy - V0) * grad_trgt)


class RadialTension(AbstractEffector):
    """
    Apply a tension perpendicular to a face.
    """

    dimensions = units.line_tension
    magnitude = "radial_tension"
    label = "Apical basal tension"
    element = "face"
    specs = {"face": {"height": 1.0, "radial_tension": 1.0}}

    @staticmethod
    def energy(eptm):
        df = eptm.face_df
        return df["height"] * df["radial_tension"]

    @staticmethod
    def gradient(eptm):
        upcast_tension = eptm.upcast_face(
            eptm.face_df["radial_tension"] / eptm.face_df["num_sides"]
        )

        upcast_height = eptm.upcast_srce(height_grad(eptm))
        grad_srce = to_nd(upcast_tension, 3) * upcast_height
        grad_srce.columns = ["g" + u for u in eptm.coords]
        return grad_srce / 2, grad_srce / 2


class BarrierElasticity(AbstractEffector):
    """
    Barrier use to maintain the tissue integrity.
    """

    dimensions = units.line_elasticity
    magnitude = "barrier_elasticity"
    label = "Barrier elasticity"
    element = "vert"
    specs = {
        "vert": {"barrier_elasticity": 1.0, "is_active": 1, "delta_rho": 0.0}
    }  # distance to a barrier membrane

    @staticmethod
    def energy(eptm):
        df = eptm.vert_df
        return 0.5 * df["barrier_elasticity"] * df["delta_rho"] * df["delta_rho"]

    @staticmethod
    def gradient(eptm):
        df = eptm.vert_df
        factor = df["barrier_elasticity"] * df["delta_rho"]

        factor_nd = to_nd(factor, 3)

        grad = height_grad(eptm) * factor_nd
        grad.columns = ["g" + c for c in eptm.coords]

        return grad, None


class ChiralTorque(AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = "torque_coef"
    label = "Apply Chiral Torque to Cells"
    element = "face"
    specs = {
        "face": {"torque_coef": 0.0, "is_alive": 1}
    }

    @staticmethod
    def energy(eptm):
        return np.zeros(eptm.Nv)

    @staticmethod
    def gradient(eptm):
        torque = eptm.face_df['torque_coef']
        torque = to_nd(eptm.upcast_face(torque), len(eptm.coords))
        grad_srce = np.multiply(eptm.edge_df[["r" + z for z in eptm.coords]].values, torque)
        normal = eptm.edge_df[["n" + u for u in eptm.coords]].values
        grad_srce = np.cross(grad_srce, normal)
        srce_active = eptm.upcast_srce(eptm.vert_df['is_active'])
        grad_srce = grad_srce * \
            to_nd(srce_active, len(eptm.coords)) #* srce_bound_coords
        grad_srce = pd.DataFrame(grad_srce)
        grad_srce.columns = ["g" + u for u in eptm.coords]

        return grad_srce, None


class ActiveMigration(AbstractEffector):
    """Active cell migration force along a specified direction.

    This is a non-conservative force that drives cells to migrate
    along a specified vector direction with constant magnitude.
    The direction is read from the face columns ``m`` + coordinate
    (``mx``, ``my``, and ``mz`` in 3D); it need not be normalized.
    """

    dimensions = units.force
    magnitude = "migration_strength"
    label = "Active Migration"
    element = "face"
    specs = {
        "face": {
            "is_alive": 1,
            "migration_strength": 0.1,  # magnitude of migration force
            "mx": 1.0,  # migration direction
            "my": 0.0,
            "mz": 0.0,
        }
    }

    spatial_ref = "migration_strength", units.force

    @staticmethod
    def energy(eptm):
        """Non-conservative force - return zero energy."""
        return np.zeros(eptm.Nv)

    @staticmethod
    def gradient(eptm):
        """Each cell exerts a force of magnitude `migration_strength` along its
        migration direction, split equally between its vertices."""
        df = eptm.face_df

        force_magnitude = df["migration_strength"] * df["is_alive"] / df["num_sides"]

        migration_dirs = df[[f"m{c}" for c in eptm.coords]].to_numpy()

        norms = np.linalg.norm(migration_dirs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # avoid division by zero
        migration_dirs_normalized = migration_dirs / norms

        force_vectors = migration_dirs_normalized * force_magnitude.to_numpy()[:, np.newaxis]

        force_per_coord = {
            coord: eptm.upcast_face(force_vectors[:, i])
            for i, coord in enumerate(eptm.coords)
        }

        force_nd = np.column_stack([
            to_nd(force_per_coord[coord], len(eptm.coords))
            for coord in eptm.coords
        ])

        grad_srce = pd.DataFrame(force_nd, columns=["g" + u for u in eptm.coords])

        return -grad_srce, None


class SurfaceElasticity(AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = "surface_elasticity"
    label = "Apply Surface Elasticity to vertices such that a flat surface is prefered"
    element = "vert"
    specs = {
        "vert": {"torque_coef": 0.0, "is_alive": 1}
    }

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.vert_df, "dev_length", lambda df: df["surface_elasticity"] * df["is_active"], "prefered_deviation"
        )

    @staticmethod
    def gradient(eptm):
        ka_a0_ = elastic_force(
            eptm.vert_df, "dev_length", lambda df: df["surface_elasticity"] * df["is_active"], "prefered_deviation"
        )

        ka_a0 = to_nd(ka_a0_, len(eptm.coords))

        grad = eptm.vert_df[["d" + x for x in eptm.coords]].to_numpy()

        grad = pd.DataFrame(grad * ka_a0)

        grad.columns = ["g" + u for u in eptm.coords]

        return grad, None

class VesselSurfaceElasticity(AbstractEffector):
    """
    Applies an elastic force to maintain vertex distance from xy origin
    """

    dimensions = units.line_elasticity
    magnitude = "surface_elasticity"
    label = "Apply Surface Elasticity to vertices such that a flat surface is prefered"
    element = "vert"
    specs = {
        "vert": {"torque_coef": 0.0, "is_alive": 1}
    }

    @staticmethod
    def energy(eptm):
        return elastic_energy(
            eptm.vert_df, "distance_origin", lambda df: df["vessel_elasticity"] * df["is_alive"], "prefered_radius"
        )

    @staticmethod
    def gradient(eptm):
        coords = ["x", "y", "z"]
        if "axis" in eptm.settings.keys():
            axis = eptm.settings["axis"]
        else:
            axis = "z"
        coords.remove(axis)
        ka_a0_ = elastic_force(
            eptm.vert_df, "distance_origin", lambda df: df["vessel_elasticity"] * df["is_alive"], "prefered_radius"
        )

        ka_a0 = to_nd(ka_a0_, len(eptm.coords))

        grad = eptm.vert_df[["o" + x for x in coords]].copy()
        grad["o" + f"{axis}"] = 0
        grad = grad.to_numpy()

        grad = pd.DataFrame(grad * ka_a0)

        grad.columns = ["g" + u for u in eptm.coords]

        return grad, None

def _exponants(dimensions, ref_dimensions, spatial_unit=None, temporal_unit=None):

    spatial_exponant = time_exponant = 0
    rel_dimensionality = (dimensions / ref_dimensions).dimensionality

    if spatial_unit is not None:
        spatial_exponant = (
            rel_dimensionality.get(units.length, 0)
            / spatial_unit.dimensionality[units.length]
        )

    if temporal_unit is not None:
        time_exponant = (
            rel_dimensionality.get(units.time, 0)
            / temporal_unit.dimensionality[units.time]
        )
    return spatial_exponant, time_exponant


def scaler(nondim_specs, dim_specs, effector, ref_effector):
    spatial_val, spatial_unit = ref_effector.spatial_ref
    temporal_val, temporal_unit = ref_effector.temporal_ref

    s_expo, t_expo = _exponants(
        effector.dimensions, ref_effector.dimensions, spatial_unit, temporal_unit
    )

    ref_magnitude = ref_effector.magnitude
    ref_element = ref_effector.element
    factor = (
        dim_specs[ref_element][ref_magnitude]
        * dim_specs[ref_element].get(spatial_val, 1) ** s_expo
        * dim_specs[ref_element].get(temporal_val, 1) ** t_expo
    )
    return factor


def dimensionalize(nondim_specs, dim_specs, effector, ref_effector):
    magnitude = effector.magnitude
    element = effector.element
    factor = scaler(nondim_specs, dim_specs, effector, ref_effector)
    dim_specs[element][magnitude] = factor * nondim_specs[element][magnitude]
    return dim_specs


def normalize(dim_specs, nondim_specs, effector, ref_effector):
    magnitude = effector.magnitude
    element = effector.element
    factor = scaler(nondim_specs, dim_specs, effector, ref_effector)
    dim_specs[element][magnitude] = nondim_specs[element][magnitude] / factor
    return dim_specs
