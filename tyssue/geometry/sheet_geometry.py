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
        srce_pos = sheet.upcast_srce(sheet.vert_df[coords]).values
        trgt_pos = sheet.upcast_trgt(sheet.vert_df[coords]).values

        normals = np.cross(srce_pos - face_pos, trgt_pos - srce_pos)
        sheet.edge_df[sheet.ncoords] = normals

    @staticmethod
    def update_areas(sheet):
        '''
        Updates the normal coordniate of each (srce, trgt, face) face.
        '''
        sheet.edge_df['sub_area'] = np.linalg.norm(
            sheet.edge_df[sheet.ncoords],
            axis=1) / 2
        sheet.face_df['area'] = sheet.sum_face(sheet.edge_df['sub_area'])

    @staticmethod
    def update_vol(sheet):
        '''
        Note that this is an approximation of the sheet geometry
        module.

        '''
        sheet.edge_df['sub_vol'] = (
            sheet.upcast_face(sheet.face_df['height']) *
            sheet.edge_df['sub_area'])
        sheet.face_df['vol'] = sheet.sum_face(sheet.edge_df['sub_vol'])

    @staticmethod
    def update_height(sheet):
        """
        Update the height of the sheet vertices, based on the geometry
        specified in the sheet settings:

        `sheet.settings['geometry']` can be set to

          - `cylindrical`: the vertex height is
             measured with respect to the distance to the the axis
             specified in sheet.settings['height_axis'] (e.g `z`)
          - `flat`: the vertex height is
             measured with respect to the position on the axis
             specified in sheet.settings['height_axis']
          - 'spherical': the vertex height is measured with respect to its
             distance to the coordinate system centers

        In all the cases, this distance is shifted by an amount of
        `sheet.vert_df['basal_shift']`
        """
        w = sheet.settings['height_axis']
        u, v = (c for c in sheet.coords if c != w)
        if sheet.settings['geometry'] == 'cylindrical':
            sheet.vert_df['rho'] = np.hypot(sheet.vert_df[v],
                                            sheet.vert_df[u])
            sheet.vert_df['height'] = (sheet.vert_df['rho'] -
                                       sheet.vert_df['basal_shift'])

        elif sheet.settings['geometry'] == 'flat':
            sheet.vert_df['rho'] = sheet.vert_df[w]
            sheet.vert_df['height'] = (sheet.vert_df[w] -
                                       sheet.vert_df['basal_shift'])

        elif sheet.settings['geometry'] == 'spherical':
            sheet.vert_df['rho'] = np.linalg.norm(sheet.vert_df[sheet.coords],
                                                  axis=1)
            sheet.vert_df['height'] = (sheet.vert_df['rho'] -
                                       sheet.vert_df['basal_shift'])

        edge_height = sheet.upcast_srce(sheet.vert_df[['height', 'rho']])

        edge_height.set_index(sheet.edge_df['face'],
                              append=True, inplace=True)
        sheet.face_df[['height', 'rho']] = edge_height.mean(level='face')

    @staticmethod
    def face_rotation(sheet, face, psi=0):
        """Returns a 3D rotation matrix such that the face normal points
        in the z axis

        Parameters
        ----------
        sheet: a :class:Sheet object
        face: int,
          the index of the face on which to rotate the sheet
        psi: float,
          Optional angle giving the rotation along the `z` axis

        Returns
        -------
        rotation: (3, 3) np.ndarray
          The rotation matrix

        """
        normal = sheet.edge_df[
            sheet.edge_df['face'] == face][sheet.ncoords].mean()
        normal = normal / np.linalg.norm(normal)
        cos_psi = np.cos(psi)
        sin_psi = np.sin(psi)

        phi = np.arctan2(normal.ny, normal.nx)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = normal.nz
        sin_theta = (1 - cos_theta**2)**0.5

        rotation = np.array([[cos_psi*cos_phi - sin_psi*cos_theta*sin_phi,
                              -cos_psi*sin_phi - sin_psi*cos_theta*cos_phi,
                              sin_psi*sin_theta],
                             [sin_psi*cos_phi + cos_psi*cos_theta*sin_phi,
                              -sin_psi*sin_phi + cos_psi*cos_theta*cos_phi,
                              -cos_psi*sin_theta],
                             [sin_theta*sin_phi,
                              sin_theta*cos_phi,
                              cos_theta]])

        return rotation

    @classmethod
    def face_projected_pos(cls, sheet, face, psi=0):
        """Returns the position of a face vertices projected on a plane
        perpendicular to the face normal, and translated so that the face
        center is at the center of the coordinate system


        Parameters
        ----------
        sheet: a :class:Sheet object
        face: int,
          the index of the face on which to rotate the sheet
        psi: float,
          Optional angle giving the rotation along the `z` axis

        Returns
        -------
        rot_pos: pd.DataFrame
           The rotated, relative positions of the face's vertices
        """

        face_orbit = sheet.edge_df[sheet.edge_df['face'] == face]['srce']
        n_sides = face_orbit.shape[0]
        face_pos = np.repeat(
            sheet.face_df.loc[face, sheet.coords].values,
            n_sides).reshape(len(sheet.coords), n_sides).T
        rel_pos = sheet.vert_df.loc[face_orbit.values, sheet.coords] - face_pos

        rotation = cls.face_rotation(sheet, face, psi=psi)

        rot_pos = rel_pos.copy()
        rot_pos.loc[:] = np.dot(rotation, rel_pos.T).T
        return rot_pos
