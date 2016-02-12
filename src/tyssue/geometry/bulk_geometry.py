import numpy as np
from .sheet_geometry import SheetGeometry


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
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        '''
        cls.update_dcoords(eptm)
        cls.update_length(eptm)
        cls.update_centroid(eptm)
        cls.update_normals(eptm)
        cls.update_areas(eptm)
        cls.update_perimeters(eptm)
        cls.update_vol(eptm)

    @staticmethod
    def update_vol(eptm):
        '''

        '''
        face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
        cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords])

        eptm.je_df['sub_vol'] = np.sum(
            (face_pos - cell_pos) *
            eptm.je_df[eptm.ncoords].values, axis=1) / 6

        eptm.cell_df['vol'] = eptm.sum_cell(eptm.je_df['sub_vol'])

    @staticmethod
    def update_centroid(eptm):

        upcast_pos = eptm.upcast_srce(eptm.jv_df[eptm.coords])
        upcast_pos = upcast_pos.set_index(eptm.je_mindex)
        eptm.face_df[eptm.coords] = upcast_pos.mean(level='face')
        eptm.cell_df[eptm.coords] = upcast_pos.mean(level='cell')
