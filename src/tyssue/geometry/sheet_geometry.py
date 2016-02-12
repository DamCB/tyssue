import numpy as np

from .planar_geometry import PlanarGeometry


class SheetGeometry(PlanarGeometry):
    """Geometry definitions for 2D sheets in 3D
    """

    @classmethod
    def update_all(cls, sheet):
        '''
        Updates the sheet geometry by updating:
        * the edge vector coordinates
        * the edge lengths
        * the face centroids
        * the normals to each edge associated face
        * the face areas
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        '''
        cls.update_dcoords(sheet)
        cls.update_length(sheet)
        cls.update_centroid(sheet)
        cls.update_height(sheet)
        cls.update_normals(sheet)
        cls.update_areas(sheet)
        cls.update_perimeters(sheet)
        cls.update_vol(sheet)

    @staticmethod
    def update_normals(sheet):
        '''
        Updates the face_df `coords` columns as the face's vertices
        center of mass.
        '''
        coords = sheet.coords
        face_pos = sheet.upcast_face(sheet.face_df[coords]).values
        srce_pos = sheet.upcast_srce(sheet.jv_df[coords]).values
        trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords]).values

        normals = np.cross(srce_pos - face_pos, trgt_pos - srce_pos)
        sheet.je_df[sheet.ncoords] = normals

    @staticmethod
    def update_areas(sheet):
        '''
        Updates the normal coordniate of each (srce, trgt, face) face.
        '''
        sheet.je_df['sub_area'] = np.linalg.norm(sheet.je_df[sheet.ncoords],
                                                 axis=1) / 2
        sheet.face_df['area'] = sheet.sum_face(sheet.je_df['sub_area'])

    @staticmethod
    def update_vol(sheet):
        '''
        Note that this is an approximation of the sheet geometry
        module.

        '''
        sheet.je_df['sub_vol'] = (sheet.upcast_srce(sheet.jv_df['height']) *
                                  sheet.je_df['sub_area'])
        sheet.face_df['vol'] = sheet.sum_face(sheet.je_df['sub_vol'])

    @staticmethod
    def update_height(sheet):

        w = sheet.settings['height_axis']
        u, v = (c for c in sheet.coords if c != w)
        if sheet.settings['geometry'] == 'cylindrical':

            sheet.jv_df['rho'] = np.hypot(sheet.jv_df[v],
                                          sheet.jv_df[u])
            sheet.jv_df['height'] = (sheet.jv_df['rho'] -
                                     sheet.jv_df['basal_shift'])

        elif sheet.settings['geometry'] == 'flat':

            sheet.jv_df['rho'] = sheet.jv_df[w]
            sheet.jv_df['height'] = sheet.jv_df[w] - sheet.jv_df['basal_shift']
