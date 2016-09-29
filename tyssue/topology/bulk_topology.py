from collections import defaultdict
import logging
import itertools
from .sheet_topology import face_division
from .base_topology import add_vert

from ..geometry.bulk_geometry import BulkGeometry

logger = logging.getLogger(name=__name__)


def cell_division(eptm, mother, vertices):

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
    daughter_faces.extend(list(eptm.face_df.index[-2:]))
