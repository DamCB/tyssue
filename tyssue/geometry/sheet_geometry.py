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
        r_ij = sheet.edge_df[["d" + c for c in coords]].to_numpy()
        r_ai = sheet.edge_df[["r" + c for c in coords]].to_numpy()
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
        rel_pos = (
            sheet.vert_df.loc[face_orbit.to_numpy(), sheet.coords].to_numpy()
            - sheet.face_df.loc[face, sheet.coords].to_numpy()
        )
        _, _, rotation = np.linalg.svd(rel_pos.astype(np.float), full_matrices=False)
        if psi:
            rotation = np.dot(rotation_matrix(psi, [0, 0, 1]), rotation)
        rot_pos = pd.DataFrame(
            np.dot(rel_pos, rotation.T), index=face_orbit, columns=sheet.coords
        )
        return rot_pos

    @classmethod
    def face_rotations(cls, sheet, method="normal"):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation returns a coordinate system (u, v, w) where the face
        vertices are mostly in the u, v plane.

        If method is 'normal', face is oriented with it's normal along w
        if method is 'svd', the u, v, w is determined through singular value decompostion
        of the face vertices relative  positions.

        svd is slower but more effective at reducing face dimensionality.

        """

        if method == "normal":
            return cls.normal_rotations(sheet)
        elif method == "svd":
            return cls.svd_rotations(sheet)
        else:
            raise ValueError("method can be either 'normal' or 'svd' ")

    @staticmethod
    def normal_rotations(sheet):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation aligns the coordinate system along each face normals

        """
        face_normals = sheet.edge_df.groupby("face")[sheet.ncoords].mean()
        rot_angles = face_normals.eval("-arctan2((nx**2 + ny**2), nz)").to_numpy()
        rot_axis = np.vstack([face_normals.ny, -face_normals.nx, np.zeros(sheet.Nf)]).T
        norm = np.linalg.norm(rot_axis, axis=1)
        if abs(norm).max() < 1e-10:
            # No rotation needed
            return
        norm = norm.clip(min=1e-10)
        rot_axis /= norm[:, None]

        r_mats = rotation_matrices(rot_angles, rot_axis)
        # upcast
        rotations = r_mats.take(sheet.edge_df["face"], axis=0)
        return rotations

    @staticmethod
    def svd_rotations(sheet):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation aligns the coordinate system according
        to each face vertex SVD

        """
        svd_rot = sheet.edge_df.groupby("face").apply(face_svd_)
        svd_rot = (
            np.concatenate(svd_rot)
            .reshape((-1, 3, 3))
            .take(sheet.edge_df["face"], axis=0)
        )
        return svd_rot

    @classmethod
    def get_phis(cls, sheet, method="normal"):
        """Returns the angle of the vertices in the plane perpendicular
        to each face normal. For not-too-deformed faces, sorting vertices by this
        gives clockwize orientation.

        I think not-too-deformed means starconvex here.

        The 'method' argument is passed to face_rotations

        """

        rel_srce_pos = sheet.edge_df[["r" + c for c in sheet.coords]]
        rots = cls.face_rotations(sheet, method)
        if rots is None:
            rotated = rel_srce_pos.to_numpy()
        else:
            rotated = np.einsum("ikj, ik -> ij", rots, rel_srce_pos)
        return np.arctan2(rotated[:, 1], rotated[:, 0])


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


def face_svd_(faces):

    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(np.float), full_matrices=False)
    return rotation
