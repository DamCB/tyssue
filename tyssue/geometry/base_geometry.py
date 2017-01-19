import numpy as np



class BaseGeometry():
    """
    """
    @staticmethod
    def update_all(sheet):
        raise NotImplementedError

    @staticmethod
    def scale(sheet, delta, coords):
        ''' Scales the coordinates `coords`
        by a factor `delta`
        '''
        sheet.vert_df[coords] = sheet.vert_df[coords] * delta

    @staticmethod
    def update_dcoords(sheet):
        '''
        Update the edge vector coordinates  on the
        `coords` basis (`default_coords` by default).
        Modifies the corresponding
        columns (i.e `['dx', 'dy', 'dz']`) in sheet.edge_df.
        '''
        data = sheet.vert_df[sheet.coords]
        srce_pos = sheet.upcast_srce(data).values
        trgt_pos = sheet.upcast_trgt(data).values

        sheet.edge_df[sheet.dcoords] = (trgt_pos - srce_pos)

    @staticmethod
    def update_length(sheet):
        '''
        Updates the edge_df `length` column on the `coords` basis
        '''
        sheet.edge_df['length'] = np.linalg.norm(sheet.edge_df[sheet.dcoords],
                                                 axis=1)

    @staticmethod
    def update_centroid(sheet):
        '''
        Updates the face_df `coords` columns as the face's vertices
        center of mass.
        '''
        upcast_pos = sheet.upcast_srce(sheet.vert_df[sheet.coords])
        upcast_pos.set_index(sheet.edge_df['face'],
                             append=True, inplace=True)
        sheet.face_df[sheet.coords] = upcast_pos.mean(level='face')

    @classmethod
    def center(cls, eptm):
        """
        Transates the epithelium vertices so that the center
        of mass is at the center of the coordinate system,
        and updates the geometry
        """

        eptm.vert_df[eptm.coords] = (
            eptm.vert_df[eptm.coords].values -
            eptm.vert_df[eptm.coords].mean(axis=0).values[np.newaxis, :])
        cls.update_all(eptm)
