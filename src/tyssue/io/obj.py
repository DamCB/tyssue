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
    ## Assumes https://github.com/vispy/vispy/issues/1155 is closed and my PR passed, inch Allah

    vertices, faces, normals = eptm.verterts_mesh(eptm.coords,
                                                vertex_normals=True)

    write_mesh(filename,
               vertices=vertices.values,
               faces=faces.values,
               normals=normals.values,
               texcoords=None, overwrite=True)
    logger.info('Saved %s as a junction mesh .OBJ file', eptm.identifier)
