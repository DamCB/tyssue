import numpy as np
import pandas as pd

from vispy.io import write_mesh
import logging

logger = logging.getLogger(name=__name__)


def save_triangulated(filename, eptm):

    vertices, faces, _ = eptm.triangular_mesh(eptm.coords)
    write_mesh(filename,
               vertices=vertices,
               faces=faces, normals=None,
               texcoords=None, overwrite=True)
    logger.info('Saved %s as a trianglulated .OBJ file', eptm.identifier)


def save_junction_mesh(filename, eptm):

    vertices, faces, normals = eptm.verterts_mesh(eptm.coords,
                                                  vertex_normals=True)
    write_mesh(filename,
               vertices=vertices.values,
               faces=faces.values,
               normals=normals.values,
               texcoords=None, overwrite=True,
               reshape_faces=False)  # GH 1155
    logger.info('Saved %s as a junction mesh .OBJ file', eptm.identifier)


def write_splitted_cells(sheet, fname, epsilon=0.1):

    coords = sheet.coords
    up_srce = sheet.upcast_srce(sheet.vert_df[coords])
    up_trgt = sheet.upcast_trgt(sheet.vert_df[coords])
    up_face = sheet.upcast_face(sheet.face_df[coords])
    up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
    up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    cell_faces = pd.concat([sheet.face_df[coords], up_srce, up_trgt],
                           ignore_index=True)
    Ne, Nf = sheet.Ne, sheet.Nf

    triangles = np.vstack([sheet.edge_df['face'],
                           np.arange(Ne)+Nf,
                           np.arange(Ne)+Ne+Nf]).T
    write_mesh(fname, cell_faces.values, triangles,
               normals=None, texcoords=None, overwrite=True)
