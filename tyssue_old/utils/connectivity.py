"""Connectivity matrix computation
"""

import pandas as pd
import numpy as np

from scipy import sparse


def _index_mesh(df):
    ii1, ii2 = np.meshgrid(df.index, df.index)
    return pd.DataFrame({"row": ii1.ravel(), "col": ii2.ravel()})


def _elements_mesh(df, elem1, elem2):
    ee1, ee2 = np.meshgrid(df[elem1], df[elem2])
    return pd.DataFrame({"row": ee1.ravel(), "col": ee2.ravel()})


def edge_in_face_connectivity(eptm):
    """Returns an array of shape (eptm.Ne, eptm.Ne) with
    C_ij = 1 iff edges i and j belong to the same face.

    """
    mesh = eptm.edge_df.groupby("face").apply(_index_mesh)
    ef_connect = sparse.coo_matrix(
        (np.ones(mesh.shape[0]), (mesh["row"], mesh["col"])),
        shape=(eptm.Ne, eptm.Ne),
        dtype=int,
    ).toarray()
    return ef_connect


def face_face_connectivity(eptm, exclude_opposites=False):
    """Returns an array of shape (eptm.Nf, eptm.Nf) with
    C_ij = n, where n is the number of shared vertices
    between the faces i and j.

    Parameters
    ----------
    eptm: a :class:`tyssue.Epithelium` instance
    exclude_opposites: bool, default `False`
        if True, opposite faces are not included in the
        resulting connectivity matrix

    """
    mesh = eptm.edge_df.groupby("srce").apply(_elements_mesh, "face", "face")

    ff_connect = sparse.coo_matrix(
        (np.ones(mesh.shape[0]), (mesh["row"], mesh["col"])),
        shape=(eptm.Nf, eptm.Nf),
        dtype=int,
    ).toarray()

    ff_connect[np.arange(eptm.Nf), np.arange(eptm.Nf)] = 0
    if exclude_opposites:
        eptm.get_opposite_faces()
        oppos = eptm.face_df.query("opposite >= 0")["opposite"]
        ff_connect[oppos.index, oppos.values] = 0
    return ff_connect


def cell_cell_connectivity(eptm):
    """Returns an array of shape (eptm.Nc, eptm.Nc) with
    C_ij = n, where n is the number of connections
    between the cells i and j.

    """
    mesh = eptm.edge_df.groupby("srce").apply(_elements_mesh, "cell", "cell")

    cc_connect = sparse.coo_matrix(
        (np.ones(mesh.shape[0]), (mesh["row"], mesh["col"])),
        shape=(eptm.Nc, eptm.Nc),
        dtype=int,
    ).toarray()
    cc_connect[np.arange(eptm.Nc), np.arange(eptm.Nc)] = 0

    return cc_connect


def srce_trgt_connectivity(eptm):
    """Returns an array of shape (eptm.Nv, eptm.Nv) with
    C_ij = n, where n is the number of shared edges
    between the vertices i and j.

    """
    srce, trgt = eptm.edge_df["srce"], eptm.edge_df["trgt"]

    st_connect = sparse.coo_matrix(
        (np.ones(eptm.Ne), (srce, trgt)), shape=(eptm.Nv, eptm.Nv), dtype=int
    ).toarray()
    return st_connect


def verts_in_face_connectivity(eptm):
    """Returns an array of shape (eptm.Nv, eptm.Nv) with
    C_ij = n, where n is the number of shared faces
    between the vertices i and j.
    """
    mesh = eptm.edge_df.groupby("face").apply(_elements_mesh, "srce", "srce")

    fst_connect = sparse.coo_matrix(
        (np.ones(mesh.shape[0]), (mesh["row"], mesh["col"])),
        shape=(eptm.Nv, eptm.Nv),
        dtype=int,
    ).toarray()
    fst_connect[np.arange(eptm.Nv), np.arange(eptm.Nv)] = 0
    return fst_connect


def verts_in_cell_connectivity(eptm):
    """Returns an array of shape (eptm.Nv, eptm.Nv) with
    C_ij = n, where n is the number of shared cells
    between the vertices i and j.
    """
    mesh = eptm.edge_df.groupby("cell").apply(_elements_mesh, "srce", "srce")

    cst_connect = sparse.coo_matrix(
        (np.ones(mesh.shape[0]), (mesh["row"], mesh["col"])),
        shape=(eptm.Nv, eptm.Nv),
        dtype=int,
    ).toarray()
    cst_connect[np.arange(eptm.Nv), np.arange(eptm.Nv)] = 0
    return cst_connect
