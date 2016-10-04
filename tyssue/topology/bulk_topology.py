
import logging
import itertools
import numpy as np
from .sheet_topology import face_division
from .base_topology import add_vert
from ..geometry.utils import rotation_matrix

logger = logging.getLogger(name=__name__)


def get_division_edges(eptm, mother,
                       plane_normal,
                       plane_center=None):

    plane_normal = np.asarray(plane_normal)
    if plane_center is None:
        plane_center = eptm.cell_df.loc[mother, eptm.coords]

    n_xy = np.linalg.norm(plane_normal[:2])
    theta = -np.arctan2(n_xy, plane_normal[2])
    direction = [plane_normal[1], -plane_normal[0], 0]
    rot = rotation_matrix(theta, direction)
    cell_verts = set(eptm.edge_df[eptm.edge_df['cell'] == mother]['srce'])
    vert_pos = eptm.vert_df.loc[cell_verts, eptm.coords]
    for c in eptm.coords:
        vert_pos[c] -= plane_center[c]
    vert_pos[:] = np.dot(vert_pos, rot)

    mother_edges = eptm.edge_df[eptm.edge_df['cell'] == mother]
    srce_z = vert_pos.loc[mother_edges['srce'], 'z']
    srce_z.index = mother_edges.index
    trgt_z = vert_pos.loc[mother_edges['trgt'], 'z']
    trgt_z.index = mother_edges.index
    division_edges = mother_edges[((srce_z < 0) &
                                   (trgt_z >= 0))]

    # Order the returned edges so that their centers
    # are oriented counterclockwize in the division plane
    # in preparation for septum creation
    srce_pos = vert_pos.loc[division_edges['srce'],
                            eptm.coords].values
    trgt_pos = vert_pos.loc[division_edges['trgt'],
                            eptm.coords].values
    centers = (srce_pos + trgt_pos)/2
    theta = np.arctan2(centers[:, 2], centers[:, 1])
    return division_edges.iloc[np.argsort(theta)]


def get_division_vertices(eptm, mother,
                          plane_normal,
                          plane_center=None):

    division_edges = get_division_edges(eptm, mother,
                                        plane_normal,
                                        plane_center)
    vertices = []
    for edge in division_edges.index:
        new_vert, *new_edges = add_vert(eptm, edge)
        vertices.append(new_vert)
    return vertices


def cell_division(eptm, mother, geom, vertices):

    cell_cols = eptm.cell_df.loc[mother]
    eptm.cell_df = eptm.cell_df.append(cell_cols,
                                       ignore_index=True)
    eptm.cell_df.index.name = 'cell'
    daughter = eptm.cell_df.index[-1]

    pairs = set([frozenset([v1, v2]) for v1, v2
                 in itertools.product(vertices,
                                      vertices) if v1 != v2])
    daughter_faces = []

    # devide existing faces
    for v1, v2 in pairs:
        v1_faces = eptm.edge_df[eptm.edge_df['srce'] == v1]['face']
        v2_faces = eptm.edge_df[eptm.edge_df['srce'] == v2]['face']
        # we should devide a face if both v1 and v2
        # are part of it
        faces = set(v1_faces).intersection(v2_faces)
        for face in faces:
            daughter_faces.append(
                face_division(eptm, face, v1, v2))
    # septum
    face_cols = eptm.face_df.iloc[-2:]
    eptm.face_df = eptm.face_df.append(face_cols, ignore_index=True)
    eptm.face_df.index.name = 'face'
    septum = eptm.face_df.index[-2:]
    daughter_faces.extend(list(septum))

    num_v = len(vertices)
    num_new_edges = num_v*2

    edge_cols = eptm.edge_df.iloc[-num_new_edges:]
    eptm.edge_df = eptm.edge_df.append(edge_cols,
                                       ignore_index=True)
    eptm.edge_df.index.name = 'edge'
    new_edges = eptm.edge_df.index[-num_new_edges:]

    # To keep mother orientation, the first septum face
    # belongs to mother
    for v1, v2, edge, oppo in zip(vertices,
                                  np.roll(vertices, -1),
                                  new_edges[:num_v],
                                  new_edges[num_v:]):
        # Mother septum
        eptm.edge_df.loc[edge,
                         ['srce', 'trgt',
                          'face', 'cell']] = (v1, v2,
                                              septum[0], mother)
        # Daughter septum
        eptm.edge_df.loc[oppo,
                         ['srce', 'trgt',
                          'face', 'cell']] = (v2, v1,
                                              septum[1], daughter)

    eptm.reset_index()
    eptm.reset_topo()
    geom.update_all(eptm)

    m_septum_edges = eptm.edge_df[eptm.edge_df['face'] == septum[0]]
    m_septum_norm = m_septum_edges[eptm.ncoords].mean()
    m_septum_pos = eptm.face_df.loc[septum[0], eptm.coords]

    # splitting the faces between mother and daughter
    # based on the orientation of the vector from septum
    # center to each face center w/r to the septum norm
    mother_faces = set(eptm.edge_df[eptm.edge_df['cell'] == mother]['face'])
    for face in mother_faces:
        if face == septum[0]:
            continue
        dr = eptm.face_df.loc[face, eptm.coords] - m_septum_pos
        proj = (dr.values * m_septum_norm).sum(axis=0)
        f_edges = eptm.edge_df[eptm.edge_df['face'] == face].index
        if proj < 0:
            eptm.edge_df.loc[f_edges, 'cell'] = mother
        else:
            eptm.edge_df.loc[f_edges, 'cell'] = daughter
    eptm.reset_index()
    eptm.reset_topo()
    return daughter
