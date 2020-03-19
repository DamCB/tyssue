import math
import warnings
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi
from scipy import interpolate

from .. import config
from ..core.sheet import Sheet, get_outer_sheet
from ..core.objects import get_prev_edges
from ..core.objects import Epithelium
from ..core.monolayer import Monolayer

from ..topology import type1_transition
from .from_voronoi import from_3d_voronoi
from ..geometry.bulk_geometry import BulkGeometry, ClosedMonolayerGeometry
from ..geometry.sheet_geometry import (
    EllipsoidGeometry,
    SheetGeometry,
    ClosedSheetGeometry,
)

try:
    from .cpp import mesh_generation
except ImportError:
    "CGAL-based mesh generation utilities not found, you may need to install"
    " CGAL and build from source"
    mesh_generation = None

from .modifiers import extrude
from ..utils import single_cell, swap_apico_basal


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
    if b != a:
        warnings.warn(
            "Different half axes length along x and y"
            " axes are not supported at the moment"
        )

    dist = c / n_zs
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

    centers = np.vstack([xs, ys, zs]).T
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

    eptm = sheet_from_cell_centers(centers)
    eptm.settings["abc"] = [a, b, c]
    EllipsoidGeometry.update_all(eptm)
    return eptm


def spherical_sheet(radius, Nf, **kwargs):
    """Returns a spherical sheet with the given radius and (approximately)
    the given number of cells
    """

    centers = np.array(mesh_generation.make_spherical(Nf))
    eptm = sheet_from_cell_centers(centers, **kwargs)

    rhos = (eptm.vert_df[eptm.coords] ** 2).sum(axis=1).mean()
    ClosedSheetGeometry.scale(eptm, radius / rhos, eptm.coords)

    ClosedSheetGeometry.update_all(eptm)
    return eptm


def spherical_monolayer(R_in, R_out, Nc, apical="out"):
    """Returns a spherical monolayer with the given inner and
    outer radii, and approximately the gieven number of cells.

    The `apical` argument can be 'in' out 'out' to specify wether
    the apical face of the cells faces inward or outward, reespectively.
    """
    sheet = spherical_sheet(R_in, Nc)
    delta_R = R_out - R_in
    mono = Monolayer("mono", extrude(sheet.datasets, method="normals", scale=-delta_R))
    if apical == "out":
        swap_apico_basal(mono)
    else:
        mono.settings["lumen_side"] = "apical"

    ClosedMonolayerGeometry.update_all(mono)
    return mono


def sheet_from_cell_centers(points, noise=0):
    """Returns a Sheet object from the Voronoï tessalation
    of the cell centers.

    The strategy is to project the points on a sphere, get the Voronoï
    tessalation on this sphere and reproject the vertices on the
    original (implicit) surface through linear interpolation of the cell centers.

    Works for relatively smooth surfaces (at the very minimum star convex).

    Parameters
    ----------

    points : np.ndarray of shape (Nf, 3)
        the x, y, z coordinates of the cell centers
    noise : float, default 0.0
        addiditve normal noise stdev

    Returns
    -------
    sheet : a :class:`Sheet` object with Nf faces


    """
    points = points.copy()
    if noise:
        points += np.random.normal(0, scale=noise, size=points.shape)
    points -= points.mean(axis=0)
    bbox = np.ptp(points, axis=0)
    points /= bbox

    rhos = np.linalg.norm(points, axis=1)
    thetas = np.arcsin(points[:, 2] / rhos)
    phis = np.arctan2(points[:, 0], points[:, 1])

    sphere_rad = rhos.max() * 1.1

    points_sphere = np.vstack(
        (
            sphere_rad * np.cos(thetas) * np.cos(phis),
            sphere_rad * np.cos(thetas) * np.sin(phis),
            sphere_rad * np.sin(thetas),
        )
    ).T
    points_sphere = np.concatenate(([[0, 0, 0]], points_sphere))

    vor3D = Voronoi(points_sphere)

    dsets = from_3d_voronoi(vor3D)
    eptm_ = Epithelium("v", dsets)

    eptm_ = single_cell(eptm_, 0)

    eptm = get_outer_sheet(eptm_)
    eptm.reset_index()
    eptm.reset_topo()
    eptm.vert_df["rho"] = np.linalg.norm(eptm.vert_df[eptm.coords], axis=1)
    mean_rho = eptm.vert_df["rho"].mean()

    SheetGeometry.scale(eptm, sphere_rad / mean_rho, ["x", "y", "z"])
    SheetGeometry.update_all(eptm)

    eptm.face_df["phi"] = np.arctan2(eptm.face_df.y, eptm.face_df.x)
    eptm.face_df["rho"] = np.linalg.norm(eptm.face_df[["x", "y", "z"]], axis=1)
    eptm.face_df["theta"] = np.arcsin(eptm.face_df.z / eptm.face_df["rho"])
    _itrp = interpolate.SmoothSphereBivariateSpline(
        thetas + np.pi / 2, phis + np.pi, rhos, s=1e-4
    )
    eptm.face_df["rho"] = _itrp(
        eptm.face_df["theta"] + np.pi / 2, eptm.face_df["phi"] + np.pi, grid=False
    )
    eptm.face_df["x"] = eptm.face_df.eval("rho * cos(theta) * cos(phi)")
    eptm.face_df["y"] = eptm.face_df.eval("rho * cos(theta) * sin(phi)")
    eptm.face_df["z"] = eptm.face_df.eval("rho * sin(theta)")

    eptm.edge_df[["fx", "fy", "fz"]] = eptm.upcast_face(eptm.face_df[["x", "y", "z"]])
    eptm.vert_df[["x", "y", "z"]] = eptm.edge_df.groupby("srce")[
        ["fx", "fy", "fz"]
    ].mean()
    for i, c in enumerate("xyz"):
        eptm.vert_df[c] *= bbox[i]

    SheetGeometry.update_all(eptm)

    eptm.sanitize(trim_borders=True)

    eptm.reset_index()
    eptm.reset_topo()
    SheetGeometry.update_all(eptm)
    null_length = eptm.edge_df.query("length == 0")

    while null_length.shape[0]:
        type1_transition(eptm, null_length.index[0])
        SheetGeometry.update_all(eptm)
        null_length = eptm.edge_df.query("length == 0")

    return eptm
