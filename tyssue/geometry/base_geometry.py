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

        Also updates the upcasted coordinates of the source and target
        vertices
        '''
        data = sheet.vert_df[sheet.coords]
        srce_pos = sheet.upcast_srce(data).values
        trgt_pos = sheet.upcast_trgt(data).values
        sheet.edge_df[['s'+c for c in sheet.coords]] = srce_pos
        sheet.edge_df[['t'+c for c in sheet.coords]] = trgt_pos
        sheet.edge_df[sheet.dcoords] = (trgt_pos - srce_pos)

    @staticmethod
    def update_length(sheet):
        '''
        Updates the edge_df `length` column on the `coords` basis
        '''
        sheet.edge_df['length'] = np.linalg.norm(sheet.edge_df[sheet.dcoords],
                                                 axis=1)

    @staticmethod
    def update_perimeters(sheet):
        '''
        Updates the perimeter of each face.
        '''
        sheet.face_df['perimeter'] = sheet.sum_face(sheet.edge_df['length'])

    @staticmethod
    def update_centroid(sheet):
        '''
        Updates the face_df `coords` columns as the face's vertices
        center of mass. Also updates the edge_df fx, fy, fz columns
        with their upcasted values
        '''
        upcast_pos = sheet.upcast_srce(sheet.vert_df[sheet.coords])
        upcast_pos.set_index(sheet.edge_df['face'],
                             append=True, inplace=True)
        sheet.face_df[sheet.coords] = upcast_pos.mean(level='face')
        face_pos = sheet.upcast_face(sheet.face_df[sheet.coords])
        sheet.edge_df[['f'+c for c in sheet.coords]] = face_pos


    @staticmethod
    def center(eptm):
        """
        Transates the epithelium vertices so that the center
        of mass is at the center of the coordinate system,
        and updates the geometry
        """

        eptm.vert_df[eptm.coords] = (
            eptm.vert_df[eptm.coords].values -
            eptm.vert_df[eptm.coords].mean(axis=0).values[np.newaxis, :])

    @staticmethod
    def dist_to_point(vert_df, point, coords):
        """
        Returns the distance of all vertices from point over the
        coordinates

        Parameters
        ----------
        vert_df: a :class:`pandas.DataFrame` with the points coordinates
          in the columns given by the `coords` argument

        point: a doublet (in 2D) or triplet (in 3D) giving the reference point
          coordinates

        coords: list of 2 or 3 strings giving the column names

        Returns
        -------

        distance: a :class:`pandas.Series` with the same length
          as the input `vert_df`
        """
        return sum(((vert_df[c] - u)**2 for
                    c, u in zip(coords, point)))**0.5
