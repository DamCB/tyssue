"""

Hexagonal grids
---------------
"""
import numpy as np
import pandas as pd

from ..config.geometry import flat_sheet
from .utils import make_df


def hexa_grid2d(nx, ny, distx, disty, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:ny, 0:nx]
    cx = cx.astype(np.float)
    cy = cy.astype(np.float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(), cy.flatten()]).astype(np.float).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers


def hexa_grid3d(nx, ny, nz, distx=1.0, disty=1.0, distz=1.0, noise=None):
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

    centers = np.vstack([cx.flatten(), cy.flatten(), cz.flatten()]).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    centers[:, 2] *= distz
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers


"""

Three cells sheet
-----------------
"""


def three_faces_sheet_array():
    """
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

    """
    Nc = 3  # Number of faces
    points = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.5, 0.866],
            [1.0, 1.732],
            [0.0, 1.732],
            [-0.5, 0.866],
            [-1.5, 0.866],
            [-2, 0.0],
            [-1.5, -0.866],
            [-0.5, -0.866],
            [0, -1.732],
            [1, -1.732],
            [1.5, -0.866],
        ]
    )

    edges = np.array(
        [
            [0, 1],
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
            [12, 1],
        ]
    )

    Nv, Ne = len(points), len(edges)
    return points, edges, (Nc, Nv, Ne)


def three_faces_sheet(zaxis=False):
    """
    Creates the apical junctions mesh of three packed hexagonal faces.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Faces have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    face_df: the faces `DataFrame` indexed from 0 to 2
    vert_df: the junction vertices `DataFrame`
    edge_df: the junction edges `DataFrame`

    """
    points, _, (Nc, Nv, Ne) = three_faces_sheet_array()

    if zaxis:
        coords = ["x", "y", "z"]
    else:
        coords = ["x", "y"]

    face_idx = pd.Index(range(Nc), name="face")
    vert_idx = pd.Index(range(Nv), name="vert")

    _edge_e_idx = np.array(
        [
            [0, 1, 0],
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
            [1, 0, 2],
        ]
    )

    edge_idx = pd.Index(range(_edge_e_idx.shape[0]), name="edge")

    specifications = flat_sheet()

    # ## Faces DataFrame
    face_df = make_df(index=face_idx, spec=specifications["face"])

    # ## Junction vertices and edges DataFrames
    vert_df = make_df(index=vert_idx, spec=specifications["vert"])
    edge_df = make_df(index=edge_idx, spec=specifications["edge"])

    edge_df["srce"] = _edge_e_idx[:, 0]
    edge_df["trgt"] = _edge_e_idx[:, 1]
    edge_df["face"] = _edge_e_idx[:, 2]

    vert_df.loc[:, coords[:2]] = points
    if zaxis:
        vert_df.loc[:, coords[2:]] = 0.0

    datasets = {"face": face_df, "vert": vert_df, "edge": edge_df}
    return datasets, specifications
