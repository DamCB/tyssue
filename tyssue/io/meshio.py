import meshio
import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(name=__name__)


def save_mesh(filename, eptm):
    coords = eptm.coords
    eptm.reset_index(order=True)

    if (filename[-3:] == 'ply') or (filename[-3:] == 'obj'):
        points, faces = eptm.triangular_mesh(coords=coords, )
        cells = []
        for f in faces:
            cells.append(("triangle", np.array([f])))
        mesh = meshio.Mesh(points, cells)
        mesh.write(filename)
    elif (filename[-3:] == 'vtk'):
        points, faces = eptm.vertex_mesh(coords=coords, vertex_normals=False)
        cells = []
        for f in faces:
            cells.append(("triangle", np.array([f])))
        mesh = meshio.Mesh(points, cells)
        meshio.vtk.write(filename, mesh)
    else:
        print("This format %s is not taking in charge for now", filename[-3:])

    logger.info("Saved %s as a meshio file", filename)


def import_mesh(filename):

    if (filename[-3:] == 'ply') or (filename[-3:] == 'obj'):
        mesh = meshio.read(filename)
        vert_ = pd.DataFrame(mesh.points, columns=list('xyz'))
        edge_ = pd.DataFrame(columns=['srce', 'trgt', 'face'])
        face_ = pd.DataFrame(columns=list('xyz'))
        cpt = 0
        for c in mesh.cells[0][1]:
            edge_.loc[cpt * 3] = [c[0], c[1], cpt]
            edge_.loc[cpt * 3 + 1] = [c[1], c[2], cpt]
            edge_.loc[cpt * 3 + 2] = [c[2], c[0], cpt]

            face_.loc[cpt] = [0, 0, 0]
            cpt += 1
        data = {'vert': vert_,
                'edge': edge_,
                'face': face_}

        return data
    else:
        print("This format %s is not taking in charge for now", filename[-3:])
