"""

Hexagonal grids
---------------
"""
import math
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi

from ..config.geometry import flat_sheet, bulk_spec
from ..geometry.bulk_geometry import BulkGeometry
from ..core.objects import Epithelium
from .utils import make_df
from .from_voronoi import from_3d_voronoi


def hexa_grid2d(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:ny, 0:nx]
    cx = cx.astype(np.float)
    cy = cy.astype(np.float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(),
                         cy.flatten()]).astype(np.float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers


def hexa_grid3d(nx, ny, nz, distx=1., disty=1., distz=1., noise=None):
    """Creates an hexagonal grid of points
    """
    cz, cy, cx = np.mgrid[0:nz, 0:ny, 0:nx]
    cx = cx.astype(np.float)
    cy = cy.astype(np.float)
    cz = cz.astype(np.float)
    cx[:, ::2] += 0.5
    cy[::2, :] += 0.5
    cy *= np.sqrt(3) / 2
    cz *= np.sqrt(3) / 2

    centers = np.vstack([cx.flatten(),
                         cy.flatten(),
                         cz.flatten()]).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    centers[:, 2] *= distz
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers



def ellipse_rho(theta, a, b):
    return ((a * math.sin(theta))**2 +
            (b * math.cos(theta))**2)**0.5


def get_ellipsoid_centers(a, b, c, n_zs,
                          pos_err=0., phase_err=0.):
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



    """
    dist = c / (n_zs)
    theta = -np.pi/2
    thetas = [theta]
    while theta < np.pi/2:
        theta = theta + dist/ellipse_rho(theta, a, c)
        thetas.append(theta)

    thetas = np.array(thetas).clip(-np.pi/2, np.pi/2)
    zs = c*np.sin(thetas)

    #np.linspace(-c, c, n_zs, endpoint=False)
    #thetas = np.arcsin(zs/c)
    av_rhos = (a + b) * np.cos(thetas) / 2
    n_cells = np.ceil(av_rhos/dist).astype(np.int)

    phis = np.concatenate(
        [np.linspace(-np.pi, np.pi, nc, endpoint=False)
         + (np.pi/nc) * (i%2) for i, nc in enumerate(n_cells)])

    if phase_err > 0:
        phis += np.random.normal(scale=phase_err*np.pi,
                                 size=phis.shape)

    zs = np.concatenate(
        [z * np.ones(nc) for z, nc in zip(zs, n_cells)])
    thetas = np.concatenate(
        [theta * np.ones(nc) for theta, nc in zip(thetas, n_cells)])

    xs = a * np.cos(thetas) * np.cos(phis)
    ys = b * np.cos(thetas) * np.sin(phis)

    if pos_err > 0.:
        xs += np.random.normal(scale=pos_err,
                               size=thetas.shape)
        ys += np.random.normal(scale=pos_err,
                               size=thetas.shape)
        zs += np.random.normal(scale=pos_err,
                               size=thetas.shape)
    centers = pd.DataFrame.from_dict(
        {'x': xs, 'y': ys, 'z': zs,
         'theta': thetas, 'phi': phis})
    return centers


def ellipsoid_sheet(a, b, c, n_zs, **kwargs):

    centers = get_ellipsoid_centers(a, b, c, n_zs,
                                    **kwargs)

    centers = centers.append(pd.Series(
        {'x':0, 'y':0, 'z':0,
         'theta':0, 'phi':0,}),
         ignore_index=True)

    centers['x'] /= a
    centers['y'] /= b
    centers['z'] /= c

    vor3d = Voronoi(centers[list('xyz')].values)
    vor3d.close()
    dsets = from_3d_voronoi(vor3d)
    veptm = Epithelium('v', dsets, config.geometry.bulk_spec())
    eptm = single_cell(veptm, centers.shape[0]-1)

    eptm.vert_df['rho'] = np.linalg.norm(eptm.vert_df[eptm.coords], axis=1)
    eptm.vert_df['theta'] = np.arcsin(eptm.vert_df.eval('z/rho'))
    eptm.vert_df['phi'] = np.arctan2(eptm.vert_df['y'], eptm.vert_df['x'])

    eptm.vert_df['x'] = a * (np.cos(eptm.vert_df['theta'])
                             * np.cos(eptm.vert_df['phi']))
    eptm.vert_df['y'] = b * (np.cos(eptm.vert_df['theta'])
                             * np.sin(eptm.vert_df['phi']))
    eptm.vert_df['z'] = c * np.sin(eptm.vert_df['theta'])
    eptm.settings['abc'] = [a, b, c]
    BulkGeometry.update_all(eptm)
    return eptm



"""

Three cells sheet
-----------------
"""


def three_faces_sheet_array():
    '''
    Creates the apical junctions mesh of three packed hexagonal faces.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Faces have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    points: (13, ndim) np.array of floats
      the positions, where ndim is 2 or 3 depending on `zaxis`
    edges: (15, 2)  np.array of ints
      indices of the edges
    (Nc, Nv, Ne): triple of ints
      number of faces, vertices and edges (3, 13, 15)

    '''
    Nc = 3  # Number of faces
    points = np.array([[0., 0.],
                       [1.0, 0.0],
                       [1.5, 0.866],
                       [1.0, 1.732],
                       [0.0, 1.732],
                       [-0.5, 0.866],
                       [-1.5, 0.866],
                       [-2, 0.],
                       [-1.5, -0.866],
                       [-0.5, -0.866],
                       [0, -1.732],
                       [1, -1.732],
                       [1.5, -0.866]])

    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5],
                      [5, 0],
                      [5, 6],
                      [6, 7],
                      [7, 8],
                      [8, 9],
                      [9, 0],
                      [9, 10],
                      [10, 11],
                      [11, 12],
                      [12, 1]])

    Nv, Ne = len(points), len(edges)
    return points, edges, (Nc, Nv, Ne)


def three_faces_sheet(zaxis=False):
    '''
    Creates the apical junctions mesh of three packed hexagonal faces.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Faces have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    face_df: the faces `DataFrame` indexed from 0 to 2
    vert_df: the junction vertices `DataFrame`
    edge_df: the junction edges `DataFrame`

    '''
    points, _, (Nc, Nv, Ne) = three_faces_sheet_array()

    if zaxis:
        coords = ['x', 'y', 'z']
    else:
        coords = ['x', 'y']

    face_idx = pd.Index(range(Nc), name='face')
    vert_idx = pd.Index(range(Nv), name='vert')

    _edge_e_idx = np.array([[0, 1, 0],
                            [1, 2, 0],
                            [2, 3, 0],
                            [3, 4, 0],
                            [4, 5, 0],
                            [5, 0, 0],
                            [0, 5, 1],
                            [5, 6, 1],
                            [6, 7, 1],
                            [7, 8, 1],
                            [8, 9, 1],
                            [9, 0, 1],
                            [0, 9, 2],
                            [9, 10, 2],
                            [10, 11, 2],
                            [11, 12, 2],
                            [12, 1, 2],
                            [1, 0, 2]])

    edge_idx = pd.Index(range(_edge_e_idx.shape[0]), name='edge')

    specifications = flat_sheet()

    # ## Faces DataFrame
    face_df = make_df(index=face_idx,
                      spec=specifications['face'])

    # ## Junction vertices and edges DataFrames
    vert_df = make_df(index=vert_idx,
                      spec=specifications['vert'])
    edge_df = make_df(index=edge_idx,
                      spec=specifications['edge'])

    edge_df['srce'] = _edge_e_idx[:, 0]
    edge_df['trgt'] = _edge_e_idx[:, 1]
    edge_df['face'] = _edge_e_idx[:, 2]

    vert_df.loc[:, coords[:2]] = points
    if zaxis:
        vert_df.loc[:, coords[2:]] = 0.

    datasets = {'face': face_df, 'vert': vert_df, 'edge': edge_df}
    return datasets, specifications
