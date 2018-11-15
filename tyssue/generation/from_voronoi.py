import pandas as pd
import numpy as np

from ..config.geometry import planar_spec, bulk_spec, flat_sheet

from .utils import make_df

"""
Generate datasets and epithelia from Voronoi tessalations
-------------------------
"""


def from_3d_voronoi(voro):
    """ Creates 3D (bulk geometry) datasets from a Voronoï  tessalation

    Parameters
    ----------
    voro: a :class:`scipy.spatial.Voronoi` object

    Returns
    -------
    datasets: dict
      datasets suitable for :class:`Epithelium` implementation

    """
    specs3d = bulk_spec()

    el_idx = []

    for f_idx, (rv, rp) in enumerate(zip(voro.ridge_vertices, voro.ridge_points)):

        if -1 in rv:
            continue
        face_verts = voro.vertices[rv]
        f_center = face_verts.mean(axis=0)
        c0 = voro.points[rp[0]]
        ctof = f_center - c0

        for rv0, rv1 in zip(rv, np.roll(rv, 1, axis=0)):
            fv0 = voro.vertices[rv0]
            fv1 = voro.vertices[rv1]
            edge_v = fv1 - fv0
            fto0 = fv0 - f_center
            normal = np.cross(fto0, edge_v)
            dotp = np.dot(ctof, normal)
            if np.sign(dotp) > 0:
                el_idx.append([rv0, rv1, f_idx, rp[0]])
                el_idx.append([rv1, rv0, f_idx, rp[1]])
            else:
                el_idx.append([rv1, rv0, f_idx, rp[0]])
                el_idx.append([rv0, rv1, f_idx, rp[1]])

    el_idx = np.array(el_idx)

    coords = ["x", "y", "z"]
    edge_idx = pd.Index(range(el_idx.shape[0]), name="edge")
    edge_df = make_df(edge_idx, specs3d["edge"])

    for i, elem in enumerate(["srce", "trgt", "face", "cell"]):
        edge_df[elem] = el_idx[:, i]

    vert_idx = pd.Index(range(voro.vertices.shape[0]), name="vert")
    vert_df = make_df(vert_idx, specs3d["vert"])
    vert_df[coords] = voro.vertices
    included_verts = edge_df["srce"].unique()
    vert_df = vert_df.loc[included_verts].copy()

    cell_idx = pd.Index(range(voro.points.shape[0]), name="cell")
    cell_df = make_df(cell_idx, specs3d["cell"])
    cell_df[coords] = voro.points
    included_cells = edge_df["cell"].unique()
    cell_df = cell_df.loc[included_cells].copy()

    nfaces = len(voro.ridge_vertices)
    face_idx = pd.Index(np.arange(nfaces), name="face")
    face_df = make_df(face_idx, specs3d["face"])
    included_faces = edge_df["face"].unique()
    face_df = face_df.loc[included_faces].copy()

    edge_df.sort_values(by="cell", inplace=True)

    datasets = {"vert": vert_df, "edge": edge_df, "face": face_df, "cell": cell_df}
    return datasets


def from_2d_voronoi(voro, specs=None):
    """ Creates 2D (sheet geometry) datasets from a Voronoï  tessalation

    Parameters
    ----------
    voro: a :class:`scipy.spatial.Voronoi` object

    Returns
    -------
    datasets: dict
      datasets suitable for :class:`Epithelium` implementation

    """
    if specs is None:
        specs = planar_spec()
    el_idx = []

    for rv, rp in zip(voro.ridge_vertices, voro.ridge_points):

        if -1 in rv:
            continue
        f_center = voro.points[rp[0]]
        for rv0, rv1 in zip(rv, np.roll(rv, 1, axis=0)):
            fv0 = voro.vertices[rv0]
            fv1 = voro.vertices[rv1]
            edge_v = fv1 - fv0
            fto0 = fv0 - f_center
            normal = np.cross(fto0, edge_v)
            if np.sign(normal) > 0:
                el_idx.append([rv0, rv1, rp[0]])
            else:
                el_idx.append([rv0, rv1, rp[1]])

    el_idx = np.array(el_idx)
    coords = ["x", "y"]
    edge_idx = pd.Index(range(el_idx.shape[0]), name="edge")
    edge_df = make_df(edge_idx, specs["edge"])

    for i, elem in enumerate(["srce", "trgt", "face"]):
        edge_df[elem] = el_idx[:, i]

    vert_idx = pd.Index(range(voro.vertices.shape[0]), name="vert")
    vert_df = make_df(vert_idx, specs["vert"])

    vert_df[coords] = voro.vertices

    face_idx = pd.Index(range(voro.points.shape[0]), name="face")
    face_df = make_df(face_idx, specs["face"])
    face_df[coords] = voro.points

    datasets = {"vert": vert_df, "edge": edge_df, "face": face_df}
    return datasets
