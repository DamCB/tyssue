import math
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

from .. import config
from ..core.sheet import Sheet, get_outer_sheet
from ..core.objects import get_prev_edges
from ..core.objects import Epithelium
from .from_voronoi import from_3d_voronoi
from ..geometry.bulk_geometry import BulkGeometry
from ..geometry.sheet_geometry import EllipsoidGeometry

from ..utils import single_cell


class AnnularSheet(Sheet):
    """2D annular model of a cylinder-like monolayer.

    Provides syntactic sugar to access the apical, basal and
    lateral segments of the epithlium
    """

    def segment_index(self, segment, element):
        df = getattr(self, "{}_df".format(element))
        return df[df["segment"] == segment].index

    @property
    def lateral_edges(self):
        return self.segment_index("lateral", "edge")

    @property
    def apical_edges(self):
        return self.segment_index("apical", "edge")

    @property
    def basal_edges(self):
        return self.segment_index("basal", "edge")

    @property
    def apical_verts(self):
        return self.segment_index("apical", "vert")

    @property
    def basal_verts(self):
        return self.segment_index("basal", "vert")

    def reset_topo(self):
        super().reset_topo()
        self.edge_df["prev"] = get_prev_edges(self)


def generate_ring(Nf, R_in, R_out, R_vit=None, apical="in"):
    """ Generates a 2D tyssue object aranged in a ring of Nf tetragonal cells
    with inner diameter R_in and outer diameter R_out

    Parameters
    ----------
    Nf : int
        The number of cells in the tissue
    R_in : float
        The inner ring diameter
    R_out : float
        The outer ring diameter
    R_vit : float
        The vitelline membrane diameter
        (a non strechable membrane around the annulus)
    apical : str {'in' | 'out'}
        The side of the apical surface
        if "in", the apical surface is inside the annulus, facing
        the lumen as in an organoid; if 'out': the apical side is facing the
        exterior of the tissue, as in an embryo
    Returns
    -------
    eptm : :class:`AnnularSheet`
        2D annular tissue. The `R_in` and `R_out` parameters
        are stored in the class `settings` attribute.
    """
    specs = config.geometry.planar_spec()
    specs["settings"] = specs.get("settings", {})
    specs["settings"]["R_in"] = R_in
    specs["settings"]["R_out"] = R_out
    specs["settings"]["R_vit"] = R_vit

    Ne = Nf * 4
    Nv = Nf * 2
    vert_df = pd.DataFrame(
        index=pd.Index(range(Nv), name="vert"),
        columns=specs["vert"].keys(),
        dtype=float,
    )
    edge_df = pd.DataFrame(
        index=pd.Index(range(Ne), name="edge"),
        columns=specs["edge"].keys(),
        dtype=float,
    )
    face_df = pd.DataFrame(
        index=pd.Index(range(Nf), name="face"),
        columns=specs["face"].keys(),
        dtype=float,
    )

    inner_edges = np.array(
        [
            [f0, v0, v1]
            for f0, v0, v1 in zip(range(Nf), range(Nf), np.roll(range(Nf), -1))
        ]
    )

    outer_edges = np.zeros_like(inner_edges)
    outer_edges[:, 0] = inner_edges[:, 0]
    outer_edges[:, 1] = inner_edges[:, 2] + Nf
    outer_edges[:, 2] = inner_edges[:, 1] + Nf

    left_spokes = np.zeros_like(inner_edges)
    left_spokes[:, 0] = inner_edges[:, 0]
    left_spokes[:, 1] = outer_edges[:, 2]
    left_spokes[:, 2] = inner_edges[:, 1]

    right_spokes = np.zeros_like(inner_edges)
    right_spokes[:, 0] = inner_edges[:, 0]
    right_spokes[:, 1] = inner_edges[:, 2]
    right_spokes[:, 2] = outer_edges[:, 1]

    edges = np.concatenate([inner_edges, outer_edges, left_spokes, right_spokes])

    edge_df[["face", "srce", "trgt"]] = edges
    edge_df[["face", "srce", "trgt"]] = edge_df[["face", "srce", "trgt"]].astype(int)

    thetas = np.linspace(0, 2 * np.pi, Nf, endpoint=False)
    thetas += thetas[1] / 2

    thetas = thetas[::-1]
    # Setting vertices position (turning clockwise for correct orientation)
    vert_df.loc[range(Nf), "x"] = R_in * np.cos(thetas)
    vert_df.loc[range(Nf), "y"] = R_in * np.sin(thetas)
    vert_df.loc[range(Nf, 2 * Nf), "x"] = R_out * np.cos(thetas)
    vert_df.loc[range(Nf, 2 * Nf), "y"] = R_out * np.sin(thetas)

    vert_df["segment"] = "basal"
    edge_df["segment"] = "basal"
    if apical == "out":
        edge_df.loc[range(Nf, 2 * Nf), "segment"] = "apical"
        vert_df.loc[range(Nf, 2 * Nf), "segment"] = "apical"
    elif apical == "in":
        edge_df.loc[range(Nf), "segment"] = "apical"
        vert_df.loc[range(Nf), "segment"] = "apical"
    else:
        raise ValueError(
            f"apical argument not understood,"
            'should be either "in" or "out", got {apical}'
        )
    edge_df.loc[range(2 * Nf, 4 * Nf), "segment"] = "lateral"

    datasets = {"vert": vert_df, "edge": edge_df, "face": face_df}
    ring = AnnularSheet("ring", datasets, specs, coords=["x", "y"])

    ring.reset_topo()
    return ring


"""
Ellipsoid 2.5D sheet
"""


def ellipse_rho(theta, a, b):
    return ((a * math.sin(theta)) ** 2 + (b * math.cos(theta)) ** 2) ** 0.5


def get_ellipsoid_centers(a, b, c, n_zs, pos_err=0.0, phase_err=0.0):
    """
    Creates hexagonaly organized points on the surface of an ellipsoid

    Parameters
    ----------
    a, b, c: float
      ellipsoid radii along the x, y and z axes, respectively
      i.e the ellipsoid boounding box will be
      `[[-a, a], [-b, b], [-c, c]]`
    n_zs :  float
      number of cells on the z axis, typical
    pos_err : float, default 0.
      normaly distributed noise of std. dev. pos_err is added
      to the centers positions
    phase_err : float, default 0.
      normaly distributed noise of std. dev. phase_err is added
      to the centers angle ϕ

    """
    dist = c / (n_zs)
    theta = -np.pi / 2
    thetas = []
    while theta < np.pi / 2:
        theta = theta + dist / ellipse_rho(theta, a, c)
        thetas.append(theta)

    thetas = np.array(thetas).clip(-np.pi / 2, np.pi / 2)
    zs = c * np.sin(thetas)

    # np.linspace(-c, c, n_zs, endpoint=False)
    # thetas = np.arcsin(zs/c)
    av_rhos = (a + b) * np.cos(thetas) / 2
    n_cells = np.ceil(av_rhos / dist).astype(np.int)

    phis = np.concatenate(
        [
            np.linspace(-np.pi, np.pi, nc, endpoint=False) + (np.pi / nc) * (i % 2)
            for i, nc in enumerate(n_cells)
        ]
    )

    if phase_err > 0:
        phis += np.random.normal(scale=phase_err * np.pi, size=phis.shape)

    zs = np.concatenate([z * np.ones(nc) for z, nc in zip(zs, n_cells)])
    thetas = np.concatenate([theta * np.ones(nc) for theta, nc in zip(thetas, n_cells)])

    xs = a * np.cos(thetas) * np.cos(phis)
    ys = b * np.cos(thetas) * np.sin(phis)

    if pos_err > 0.0:
        xs += np.random.normal(scale=pos_err, size=thetas.shape)
        ys += np.random.normal(scale=pos_err, size=thetas.shape)
        zs += np.random.normal(scale=pos_err, size=thetas.shape)
    centers = pd.DataFrame.from_dict(
        {"x": xs, "y": ys, "z": zs, "theta": thetas, "phi": phis}
    )
    return centers


def ellipsoid_sheet(a, b, c, n_zs, **kwargs):
    """Creates an ellipsoidal apical mesh.

    Parameters
    ----------
    a, b, c : floats
       Size of the ellipsoid half axes in
       the x, y, and z directions, respectively
    n_zs : int
       The (approximate) number of faces along the z axis.

    kwargs are passed to `get_ellipsoid_centers`

    Returns
    -------
    eptm : a :class:`Epithelium` object

    The mesh returned is an `Epithelium` and not a simpler `Sheet`
    so that a unique cell data can hold information on the
    whole volume of the ellipsoid.

    """
    centers = get_ellipsoid_centers(a, b, c, n_zs, **kwargs)

    centers = centers.append(
        pd.Series({"x": 0, "y": 0, "z": 0, "theta": 0, "phi": 0}), ignore_index=True
    )

    centers["x"] /= a
    centers["y"] /= b
    centers["z"] /= c

    vor3d = Voronoi(centers[list("xyz")].values)
    vor3d.close()
    dsets = from_3d_voronoi(vor3d)
    veptm = Epithelium("v", dsets, config.geometry.bulk_spec())
    eptm = get_outer_sheet(single_cell(veptm, centers.shape[0] - 1))
    eptm.vert_df["rho"] = np.linalg.norm(eptm.vert_df[eptm.coords], axis=1)
    eptm.vert_df["theta"] = np.arcsin(eptm.vert_df.eval("z/rho"))
    eptm.vert_df["phi"] = np.arctan2(eptm.vert_df["y"], eptm.vert_df["x"])

    eptm.vert_df["x"] = a * (
        np.cos(eptm.vert_df["theta"]) * np.cos(eptm.vert_df["phi"])
    )
    eptm.vert_df["y"] = b * (
        np.cos(eptm.vert_df["theta"]) * np.sin(eptm.vert_df["phi"])
    )
    eptm.vert_df["z"] = c * np.sin(eptm.vert_df["theta"])
    eptm.settings["abc"] = [a, b, c]
    EllipsoidGeometry.update_all(eptm)
    return eptm


def spherical_sheet(radius, Nf, **kwargs):
    """Returns a spherical sheet with the given radius and (approximately)
    the given number of cells
    """

    n_zs = int(np.ceil(np.roots([2, 1.0, -Nf])[-1]))  # determined experimentaly ;p
    eptm = ellipsoid_sheet(radius, radius, radius, n_zs, **kwargs)
    eptm.settings.pop("abc")
    eptm.settings["radius"] = radius
    return eptm
