import numpy as np
import pandas as pd

from .planar_geometry import PlanarGeometry
from .utils import rotation_matrix, rotation_matrices


class SheetGeometry(PlanarGeometry):
    """Geometry definitions for 2D sheets in 3D
    """

    @classmethod
    def update_all(cls, sheet):
        """
        Updates the sheet geometry by updating:
        * the edge vector coordinates
        * the edge lengths
        * the face centroids
        * the normals to each edge associated face
        * the face areas
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        """
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
        """
        Updates the face_df `coords` columns as the face's vertices
        center of mass.
        """
        coords = sheet.coords
        face_pos = sheet.edge_df[["f" + c for c in coords]].to_numpy()
        srce_pos = sheet.edge_df[["s" + c for c in coords]].to_numpy()
        r_ij = sheet.edge_df[["d" + c for c in coords]].to_numpy()
        r_ai = srce_pos - face_pos
        normals = np.cross(r_ai, r_ij)
        sheet.edge_df[sheet.ncoords] = normals

    @staticmethod
    def update_areas(sheet):
        """
        Updates the normal coordniate of each (srce, trgt, face) face.
        """
        sheet.edge_df["sub_area"] = (
            np.linalg.norm(sheet.edge_df[sheet.ncoords], axis=1) / 2
        )
        # face_normals = sheet.upcast_face(
        #     sheet.edge_df.groupby("face")[sheet.ncoords].mean()
        # )
        # edge_orient = np.sign(
        #     np.sum(sheet.edge_df[sheet.ncoords] * face_normals, axis=0)
        # )
        # sheet.edge_df["sub_area"] *= edge_orient
        sheet.face_df["area"] = sheet.sum_face(sheet.edge_df["sub_area"])

    @staticmethod
    def update_vol(sheet):
        """
        Note that this is an approximation of the sheet geometry
        module.

        """
        sheet.edge_df["sub_vol"] = (
            sheet.upcast_srce(sheet.vert_df["height"]) * sheet.edge_df["sub_area"]
        )
        sheet.face_df["vol"] = sheet.sum_face(sheet.edge_df["sub_vol"])

    @classmethod
    def update_height(cls, sheet):
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
          - 'rod': the vertex height is measured with respect to its
             distance to the coordinate height axis if between the focii, and
             from the closest focus otherwise. The focii positions are updated
             before the height update.
                 ---------------
               /                 \
               |  *          *   |
               \                 /
                 ---------------
        In all the cases, this distance is shifted by an amount of
        `sheet.vert_df['basal_shift']`
        """
        w = sheet.settings.get("height_axis", "z")
        geometry = sheet.settings.get("geometry", "cylindrical")
        u, v = (c for c in sheet.coords if c != w)
        if geometry == "cylindrical":
            sheet.vert_df["rho"] = np.hypot(sheet.vert_df[v], sheet.vert_df[u])

        elif geometry == "flat":
            sheet.vert_df["rho"] = sheet.vert_df[w]

        elif geometry == "spherical":
            sheet.vert_df["rho"] = np.linalg.norm(sheet.vert_df[sheet.coords], axis=1)
        elif geometry == "rod":

            a, b = sheet.settings["ab"]
            w0 = b - a
            sheet.vert_df["rho"] = np.linalg.norm(sheet.vert_df[[u, v]], axis=1)
            sheet.vert_df["left_tip"] = sheet.vert_df[w] < -w0
            sheet.vert_df["right_tip"] = sheet.vert_df[w] > w0
            l_mask = sheet.vert_df[sheet.vert_df["left_tip"] == 1].index
            r_mask = sheet.vert_df[sheet.vert_df["right_tip"] == 1].index

            sheet.vert_df.loc[l_mask, "rho"] = cls.dist_to_point(
                sheet.vert_df.loc[l_mask], [0, 0, -w0], [u, v, w]
            )
            sheet.vert_df.loc[r_mask, "rho"] = cls.dist_to_point(
                sheet.vert_df.loc[r_mask], [0, 0, w0], [u, v, w]
            )

        elif sheet.settings["geometry"] == "surfacic":
            sheet.vert_df["rho"] = 1.0

        sheet.vert_df["height"] = sheet.vert_df["rho"] - sheet.vert_df["basal_shift"]

        edge_height = sheet.upcast_srce(sheet.vert_df[["height", "rho"]])
        edge_height.set_index(sheet.edge_df["face"], append=True, inplace=True)
        sheet.face_df[["height", "rho"]] = edge_height.mean(level="face")

    @classmethod
    def reset_scafold(cls, sheet):
        """
        Re-centers and (in the case of a rod sheet) resets the
        a-b parameters and tip masks
        """

        w = sheet.settings["height_axis"]
        u, v = (c for c in sheet.coords if c != w)

        cls.center(sheet)
        if sheet.settings["geometry"] == "rod":
            rho = np.linalg.norm(sheet.vert_df[[u, v]], axis=1)
            a = np.percentile(rho, 95)
            b = np.percentile(np.abs(sheet.vert_df[w]), 95)
            sheet.settings["ab"] = [a, b]

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
          Optional angle giving the rotation along the new `z` axis

        Returns
        -------
        rotation: (3, 3) np.ndarray
          The rotation matrix

        """
        normal = sheet.edge_df[sheet.edge_df["face"] == face][sheet.ncoords].mean()
        normal = normal / np.linalg.norm(normal)
        n_xy = np.linalg.norm(normal[["nx", "ny"]])
        theta = -np.arctan2(n_xy, normal.nz)
        direction = [normal.ny, -normal.nx, 0]
        r1 = rotation_matrix(theta, direction)
        if psi == 0:
            return r1
        else:
            return np.dot(rotation_matrix(psi, [0, 0, 1]), r1)

    @staticmethod
    def face_projected_pos(sheet, face, psi=0):
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

        face_orbit = sheet.edge_df[sheet.edge_df["face"] == face]["srce"]
        # n_sides = face_orbit.shape[0]
        # face_pos = np.repeat(
        #     sheet.face_df.loc[face, sheet.coords].values,
        #     n_sides).reshape(len(sheet.coords), n_sides).T
        rel_pos = (
            sheet.vert_df.loc[face_orbit.to_numpy(), sheet.coords].to_numpy()
            - sheet.face_df.loc[face, sheet.coords].to_numpy()
        )
        _, _, rotation = np.linalg.svd(rel_pos.astype(np.float), full_matrices=False)
        # rotation = cls.face_rotation(sheet, face, psi=psi)
        if psi != 0:
            rotation = np.dot(rotation_matrix(psi, [0, 0, 1]), rotation)
        rot_pos = pd.DataFrame(
            np.dot(rel_pos, rotation.T), index=face_orbit, columns=sheet.coords
        )
        return rot_pos

    @staticmethod
    def face_rotations(sheet):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation aligns the coordinate system along face normals

        """
        normals = sheet.edge_df.groupby("face")[sheet.ncoords].mean()
        normals = normals / np.linalg.norm(normals, axis=0)
        normals = sheet.upcast_face(normals)

        n_xy = np.linalg.norm(normals[["nx", "ny"]])
        theta = -np.arctan2(n_xy, normals.nz)

        direction = np.array(
            [normals.ny.to_numpy(), -normals.nx.to_numpy(), np.zeros(sheet.Ne)]
        ).T
        rots = rotation_matrices(theta, direction)

        return rots

    @classmethod
    def get_phis(cls, sheet):
        """Returns the 'latitude' of the vertices in the plane perpendicular
        to each face normal. For not-too-deformed faces, sorting vertices by this
        gives clockwize orientation.

        I think not-too-deformed means starconvex here.

        """
        rots = cls.face_rotations(sheet)

        rel_srce_pos = (
            sheet.edge_df[["sx", "sy", "sz"]]
            - sheet.edge_df[["fx", "fy", "fz"]].to_numpy()
        ).to_numpy()
        rotated = np.einsum("ikj, ik -> ij", rots, rel_srce_pos)
        return np.arctan2(rotated[:, 1], rotated[:, 0])

    @classmethod
    def sort_oriented_edges(cls, sheet):
        phis = cls.get_phis(sheet)
        sheet.edge_df = sheet.edge_df.sort_values(["face", phis]).reset_index()


class ClosedSheetGeometry(SheetGeometry):
    """Geometry for a closed 2.5D sheet.

    Apart from the geometry update from a normal sheet, the enclosed
    volume is also computed. The value is stored in `sheet.settings["lumen_vol"]`
    """

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_lumen_vol(sheet)

    @staticmethod
    def update_lumen_vol(sheet):
        lumen_pos_faces = sheet.edge_df[["f" + c for c in sheet.coords]].to_numpy()
        lumen_sub_vol = (
            np.sum((lumen_pos_faces) * sheet.edge_df[sheet.ncoords].to_numpy(), axis=1)
            / 6
        )
        sheet.settings["lumen_vol"] = sum(lumen_sub_vol)


class EllipsoidGeometry(ClosedSheetGeometry):
    @staticmethod
    def update_height(eptm):

        a, b, c = eptm.settings["abc"]
        eptm.vert_df["theta"] = np.arcsin((eptm.vert_df.z / c).clip(-1, 1))
        eptm.vert_df["vitelline_rho"] = a * np.cos(eptm.vert_df["theta"])
        eptm.vert_df["basal_shift"] = (
            eptm.vert_df["vitelline_rho"] - eptm.specs["vert"]["basal_shift"]
        )
        eptm.vert_df["delta_rho"] = (
            np.linalg.norm(eptm.vert_df[["x", "y"]], axis=1)
            - eptm.vert_df["vitelline_rho"]
        ).clip(lower=0)

        SheetGeometry.update_height(eptm)

    @staticmethod
    def scale(eptm, scale, coords):
        SheetGeometry.scale(eptm, scale, coords)
        eptm.settings["abc"] = [u * scale for u in eptm.settings["abc"]]
