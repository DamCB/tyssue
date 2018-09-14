import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from .objects import Epithelium
from .sheet import Sheet
from ..generation import extrude, subdivide_faces
from ..geometry.bulk_geometry import BulkGeometry


class Monolayer(Epithelium):
    """
    3D monolayer epithelium
    """

    def __init__(self, name, datasets, specs, coords=None):

        super().__init__(name, datasets,
                         specs, coords)
        self.vert_df['is_active'] = 1
        self.cell_df['is_alive'] = 1
        self.face_df['is_alive'] = 1
        BulkGeometry.update_all(self)

    @classmethod
    def from_flat_sheet(cls, name, apical_sheet, specs,
                        thickness=1):
        datasets = extrude(apical_sheet.datasets,
                           method='translation',
                           vector=[0, 0, -thickness])

        mono = cls(name, datasets, specs)
        mono.reset_topo()
        return mono

    def segment_index(self, segment, element):
        df = getattr(self, '{}_df'.format(element))
        return df[df['segment'] == segment].index

    @property
    def lateral_faces(self):
        return self.segment_index('lateral', 'face')

    @property
    def apical_faces(self):
        return self.segment_index('apical', 'face')

    @property
    def basal_faces(self):
        return self.segment_index('basal', 'face')

    @property
    def lateral_edges(self):
        return self.segment_index('lateral', 'edge')

    @property
    def apical_edges(self):
        return self.segment_index('apical', 'edge')

    @property
    def basal_edges(self):
        return self.segment_index('basal', 'edge')

    @property
    def apical_verts(self):
        return self.segment_index('apical', 'vert')

    @property
    def basal_verts(self):
        return self.segment_index('basal', 'vert')

    def get_sub_sheet(self, segment):
        """ Returns a :class:`Sheet` object of the corresponding
        segment

        Parameters
        ----------
        segment: str, the corresponding segment, wether 'apical' or 'basal'

        """
        datasets = {element:
                    self.datasets[element].loc[
                        self.segment_index(segment, element)]
                    for element in ['edge', 'face', 'vert']}
        specs = {k: self.specs[k] for k in ['face', 'edge',
                                            'vert', 'settings']}
        return Sheet(self.identifier + 'apical', datasets, specs)


class MonolayerWithLamina(Monolayer):
    """
    3D monolayer epithelium with a lamina meshing
    """

    def __init__(self, name, datasets, specs, coords=None):

        super().__init__(name, datasets,
                         specs, coords)

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

        max_dist = self.edge_df.length.dropna().median() * 1.74
        lamina_tree = cKDTree(focal_adhesions[self.coords].values)
        lamina_edges = pd.DataFrame([[i, j] for i, j in
                                     lamina_tree.query_pairs(max_dist,
                                                             eps=1e-3)],
                                    columns=['srce', 'trgt'])
        lamina_edges.index.name = 'edge'
        lamina_edges['srce'] = focal_adhesions.index[lamina_edges['srce']]
        lamina_edges['trgt'] = focal_adhesions.index[lamina_edges['trgt']]
        # place holder face and cell
        lamina_face = self.face_df.index.max() + 1
        lamina_edges['face'] = lamina_face
        self.face_df.append(self.face_df.ix[0].copy())
        self.face_df.loc[lamina_face, 'is_alive'] = 0

        lamina_cell = self.cell_df.index.max() + 1
        lamina_edges['cell'] = lamina_cell
        self.cell_df.append(self.cell_df.ix[0].copy())
        self.cell_df.loc[lamina_cell, 'is_alive'] = 0

        lamina_edges.index += self.edge_df.index.max() + 1
        lamina_edges['segment'] = 'lamina'
        lamina_edges['subdiv'] = 0
        self.edge_df = pd.concat([self.edge_df, lamina_edges], sort=True)
        self.reset_topo()
        BulkGeometry.update_all(self)

    @property
    def lamina_edges(self):
        return self.segment_index('lamina', 'edge')
