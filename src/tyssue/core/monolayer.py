import numpy as np
import pandas as pd

from scipy.spatial import Delaunay
from .objects import Epithelium
from .generation import extrude, subdivide_faces
from ..geometry.bulk_geometry import BulkGeometry


class Monolayer(Epithelium):
    """
    3D monolayer epithelium
    """
    def __init__(self, name, apical_sheet, specs):

        datasets = extrude(apical_sheet.datasets,
                           method='translation',
                           vector=[0, 0, -1])

        super().__init__(name, datasets, specs)
        self.vert_df['is_active'] = 1
        self.cell_df['is_alive'] = 1
        self.face_df['is_alive'] = 1

        BulkGeometry.update_all(self)

    def segment_index(self, segment, element):
        df = getattr(self, '{}_df'.format(element))
        return df[df['segment'] == segment].index

    @property
    def sagittal_faces(self):
        return self.segment_index('sagittal', 'face')

    @property
    def apical_faces(self):
        return self.segment_index('apical', 'face')

    @property
    def sagittal_edges(self):
        return self.segment_index('sagittal', 'edge')

    @property
    def apical_edges(self):
        return self.segment_index('apical', 'edge')

    @property
    def basal_faces(self):
        return self.segment_index('basal', 'face')

    @property
    def basal_edges(self):
        return self.segment_index('basal', 'edge')


class MonolayerWithLamina(Monolayer):
    """
    3D monolayer epithelium with a lamina meshing
    """
    def __init__(self, name, apical_sheet, specs):

        super().__init__(name, apical_sheet, specs)

        BulkGeometry.update_all(self)
        self.reset_index()

        subdivided = subdivide_faces(self, self.basal_faces)
        for name, df in subdivided.items():
            setattr(self, '{}_df'.format(name), df)
        self.reset_index()
        self.reset_topo()

        subdiv_edges = self.edge_df[self.edge_df['subdiv'] == 1].index
        self.edge_df.loc[subdiv_edges, 'segment'] = 'basal'

        subdiv_verts = self.vert_df[self.vert_df['subdiv'] == 1].index
        self.vert_df.loc[subdiv_verts, 'segment'] = 'basal'
        self.vert_df.loc[subdiv_verts, 'basal_shift'] = 0.
        self.vert_df.loc[subdiv_verts, 'is_active'] = 1.

        subdiv_verts = self.vert_df[self.vert_df['subdiv'] == 1].index
        focal_adhesions = self.vert_df.loc[subdiv_verts]
        lamina_d = Delaunay(focal_adhesions[['x', 'y']],
                            furthest_site=False)
        lamina_edges = pd.DataFrame(
            np.concatenate([lamina_d.simplices[:, :2],
                            lamina_d.simplices[:, 1:],
                            lamina_d.simplices[:, [0, 2]]]),
            columns=['srce', 'trgt'])
        lamina_edges['srce'] = focal_adhesions.index[lamina_edges['srce']]
        lamina_edges['trgt'] = focal_adhesions.index[lamina_edges['trgt']]
        # place holder face and cell
        lamina_face = self.face_df.index.max()+1
        lamina_edges['face'] = lamina_face
        self.face_df.append(self.face_df.ix[0].copy())
        self.face_df.loc[lamina_face, 'is_alive'] = 0

        lamina_cell = self.cell_df.index.max()+1
        lamina_edges['cell'] = lamina_cell
        self.cell_df.append(self.cell_df.ix[0].copy())
        self.cell_df.loc[lamina_cell, 'is_alive'] = 0

        lamina_edges.index += self.edge_df.index.max()+1
        lamina_edges['segment'] = 'lamina'
        self.edge_df = pd.concat([self.edge_df, lamina_edges])
        self.reset_topo()
        BulkGeometry.update_all(self)

    @property
    def lamina_edges(self):
        return self.segment_index('lamina', 'edge')
