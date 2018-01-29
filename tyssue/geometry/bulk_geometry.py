import numpy as np
from .sheet_geometry import SheetGeometry

from .utils import rotation_matrix
from ..utils import _to_3d


class BulkGeometry(SheetGeometry):
    """Geometry functions for 3D cell arangements
    """
    @classmethod
    def update_all(cls, eptm):
        '''
        Updates the eptm geometry by updating:
        * the edge vector coordinates
        * the edge lengths
        * the face centroids
        * the normals to each edge associated face
        * the face areas
        * the cell areas
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        '''
        cls.update_dcoords(eptm)
        cls.update_length(eptm)
        cls.update_perimeters(eptm)
        cls.update_centroid(eptm)
        cls.update_normals(eptm)
        cls.update_areas(eptm)
        cls.update_vol(eptm)

    @staticmethod
    def update_vol(eptm):
        '''

        '''
        face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
        cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords])

        eptm.edge_df['sub_vol'] = np.sum(
            (face_pos - cell_pos) *
            eptm.edge_df[eptm.ncoords].values, axis=1) / 6

        eptm.cell_df['vol'] = eptm.sum_cell(eptm.edge_df['sub_vol'])

    @staticmethod
    def update_areas(eptm):

        SheetGeometry.update_areas(eptm)
        eptm.cell_df['area'] = eptm.sum_cell(eptm.edge_df['sub_area'])


    @staticmethod
    def update_centroid(eptm):

        upcast_pos = eptm.upcast_srce(eptm.vert_df[eptm.coords])
        upcast_pos = upcast_pos.set_index(eptm.edge_mindex)
        eptm.face_df[eptm.coords] = upcast_pos.mean(level='face')
        eptm.cell_df[eptm.coords] = upcast_pos.mean(level='cell')

    @staticmethod
    def validate_face_norms(eptm):
        face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
        cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords])

        r_cf = (face_pos - cell_pos)
        r_cf['face'] = eptm.edge_df['face']
        r_cf = r_cf.groupby('face').mean()
        face_norm = eptm.edge_df.groupby('face')[eptm.ncoords].mean()

        proj = (face_norm * r_cf.values).sum(axis=1)
        is_outward = proj >= 0
        return is_outward


class RNRGeometry(BulkGeometry):

    @staticmethod
    def update_centroid(eptm):

        srce_pos = eptm.upcast_srce(eptm.vert_df[eptm.coords])
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[eptm.coords])
        mid_pos = (srce_pos + trgt_pos)/2
        weighted_pos =  eptm.sum_face(mid_pos * _to_3d(eptm.edge_df['length']))
        eptm.face_df[eptm.coords] = (
            weighted_pos.values /
            eptm.face_df['perimeter'].values[:, np.newaxis])
        srce_pos['cell'] = eptm.edge_df['cell']
        eptm.cell_df[eptm.coords] = srce_pos.groupby('cell').mean()


class MonoLayerGeometry(RNRGeometry):

    @staticmethod
    def basal_apical_axis(eptm, cell):
        """
        Returns a unit vector allong the apical-basal axis of the cell
        """
        edges = eptm.edge_df[eptm.edge_df['cell'] == cell]
        srce_segments = eptm.vert_df.loc[edges['srce'], 'segment']
        srce_segments.index = edges.index
        trgt_segments = eptm.vert_df.loc[edges['trgt'], 'segment']
        trgt_segments.index = edges.index
        ba_edges = edges[(srce_segments == 'apical') &
                         (trgt_segments == 'basal')]
        return ba_edges[eptm.dcoords].mean()

    @classmethod
    def cell_projected_pos(cls, eptm, cell, psi=0):
        """Returns the positions of the cell vertices
        transformed such that the cell center sits at the
        coordinate system's origin and the basal-apical axis
        is the new `z` axis.
        """
        ab_axis = cls.basal_apical_axis(eptm, cell)
        n_xy = np.linalg.norm(ab_axis[['dx', 'dy']])
        theta = -np.arctan2(n_xy, ab_axis.dz)
        direction = [ab_axis.dy, -ab_axis.dx, 0]
        rot = rotation_matrix(theta, direction)
        cell_verts = set(eptm.edge_df[eptm.edge_df['cell'] == cell]['srce'])
        vert_pos = eptm.vert_df.loc[cell_verts, eptm.coords]
        for c in eptm.coords:
            vert_pos[c] -= eptm.cell_df.loc[cell, c]

        r1 = np.dot(vert_pos, rot)

        if abs(psi) < 1e-6:
            vert_pos[:] = r1
        else:
            vert_pos[:] = np.dot(rotation_matrix(psi, [0, 0, 1]), r1)
        return vert_pos
