from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .planar_geometry import PlanarGeometry
from .utils import rotation_matrix, rotation_matrices


class SheetGeometry(PlanarGeometry):
    """Geometry definitions for 2D sheets in 3D"""

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
        cls.update_ucoords(sheet)
        cls.update_length(sheet)
        cls.update_centroid(sheet)
        # cls.update_height(sheet)
        cls.update_normals(sheet)
        cls.update_areas(sheet)
        cls.update_perimeters(sheet)
        cls.update_vol(sheet)
        cls.update_boundary_index(sheet)

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

    @staticmethod
    def update_boundary_index(eptm):

        eptm.vert_df['boundary'] = 0

        eptm.get_opposite()

        boundary_verts = eptm.edge_df.loc[eptm.edge_df['opposite'] == -1, 'trgt'].to_numpy()

        eptm.vert_df.loc[boundary_verts, "boundary"] = 1

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
        rel_pos = (sheet.vert_df.loc[face_orbit.to_numpy(), sheet.coords].to_numpy() - sheet.face_df.loc[
            face, sheet.coords].to_numpy())
        _, _, rotation = np.linalg.svd(
            rel_pos, full_matrices=False)
        if psi:
            rotation = np.dot(rotation_matrix(psi, [0, 0, 1]), rotation)
        rot_pos = pd.DataFrame(
            np.dot(rel_pos, rotation.T), index=face_orbit, columns=sheet.coords
        )
        return rot_pos

    @classmethod
    def face_rotations(cls, sheet, method="normal", output_as="edge"):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation returns a coordinate system (u, v, w) where the face
        vertices are mostly in the u, v plane.

        If method is 'normal', face is oriented with it's normal along w
        if method is 'svd', the u, v, w is determined through singular value decompostion
        of the face vertices relative  positions.

        svd is slower but more effective at reducing face dimensionality.

        Parameters
        ----------
        output_as: string, default 'edge' Return the (sheet.Ne, 3, 3),
                            else 'face' Return the (sheet.Nf, 3, 3)

        """

        if method == "normal":
            return cls.normal_rotations(sheet, output_as)
        elif method == "svd":
            return cls.svd_rotations(sheet, output_as)
        else:
            raise ValueError("method can be either 'normal' or 'svd' ")

    @staticmethod
    def normal_rotations(sheet, output_as="edge"):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation aligns the coordinate system along each face normals

        Parameters
        ----------
        output_as: string, default 'edge' Return the (sheet.Ne, 3, 3),
                            else 'face' Return the (sheet.Nf, 3, 3)
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

        rotations = rotation_matrices(rot_angles, rot_axis)
        # upcast
        if output_as == "edge":
            rotations = rotations.take(sheet.edge_df["face"], axis=0)
        return rotations

    @staticmethod
    def svd_rotations(sheet, output_as="edge"):
        """Returns the (sheet.Ne, 3, 3) array of rotation matrices
        such that each rotation aligns the coordinate system according
        to each face vertex SVD

        Parameters
        ----------
        output_as: string, default 'edge' Return the (sheet.Ne, 3, 3),
                            else 'face' Return the (sheet.Nf, 3, 3)

        """
        svd_rot = sheet.edge_df.groupby("face").apply(face_svd_)
        svd_rot = np.concatenate(svd_rot).reshape((-1, 3, 3))
        if output_as == "edge":
            svd_rot = svd_rot.take(sheet.edge_df["face"], axis=0)

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

    @staticmethod
    def update_boundary_index(sheet):
        """Updates the vert_df and edge_df dataframes with a 'boundary' column
        that takes values 0 and 1 with 1 denoting that an edge or vertex lies
        on the tissue boundary, and 0 when it does not.

        """

        sheet.vert_df['boundary'] = 0
        sheet.edge_df['boundary'] = 0

        sheet.get_opposite()

        sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'boundary'] = 1
        boundary_verts = sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'trgt'].to_numpy()

        sheet.vert_df.loc[boundary_verts, "boundary"] = 1


class CylinderGeometryInit(SheetGeometry):

    @staticmethod
    def update_boundary_index(sheet):

        sheet.vert_df['boundary'] = 0
        sheet.edge_df['boundary'] = 0

        sheet.get_opposite()

        sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'boundary'] = 1
        boundary_verts = sheet.edge_df.loc[sheet.edge_df['opposite'] == -1, 'trgt'].to_numpy()

        sheet.vert_df.loc[boundary_verts, "boundary"] = 1

    @staticmethod
    def update_tangents(sheet):

        vert_coords = sheet.vert_df[sheet.coords]
        vert_coords.loc[:, "z"] = 0
        vert_coords = vert_coords.values
        normal = np.column_stack((np.zeros(sheet.Nv), np.zeros(sheet.Nv), np.ones(sheet.Nv)))

        tangent = np.cross(vert_coords, normal)
        tangent = pd.DataFrame(tangent)

        tangent.columns = ["t" + u for u in sheet.coords]

        length = pd.DataFrame(tangent.eval("sqrt(tx**2 + ty**2 +tz**2)"), columns=['length'])
        tangent["length"] = length["length"]

        tangent = tangent[['tx', 'ty', 'tz']].div(length.length, axis=0)

        for u in sheet.coords:
            sheet.vert_df["t" + u] = tangent["t" + u]

    @staticmethod
    def update_face_tangents(sheet):

        face_coords = sheet.face_df[sheet.coords]
        face_coords["z"] = 0
        face_coords = sheet.face_df[sheet.coords].values
        normal = np.column_stack((np.zeros(sheet.Nf), np.zeros(sheet.Nf), np.ones(sheet.Nf)))

        tangent = np.cross(face_coords, normal)
        tangent = pd.DataFrame(tangent)

        tangent.columns = ["t" + u for u in sheet.coords]

        length = pd.DataFrame(tangent.eval("sqrt(tx**2 + ty**2 +tz**2)"), columns=['length'])
        tangent["length"] = length["length"]

        tangent = tangent[['tx', 'ty', 'tz']].div(length.length, axis=0)

        for u in sheet.coords:
            sheet.face_df["t" + u] = tangent["t" + u]

    @staticmethod
    def update_face_distance(sheet):
        sheet.face_df['distance_z_axis'] = sheet.face_df.eval(
            "sqrt(x** 2 + y** 2)"
        )

    @staticmethod
    def update_vert_distance(sheet):
        sheet.vert_df['distance_z_axis'] = sheet.vert_df.eval(
            "sqrt(x** 2 + y** 2)"
        )

    @staticmethod
    def update_vert_deviation(sheet):
        if "dev_length" not in sheet.vert_df.columns:
            sheet.vert_df["dev_length"] = np.nan
        if "dx" not in sheet.vert_df.columns:
            sheet.vert_df["dx"] = np.nan
        if "dy" not in sheet.vert_df.columns:
            sheet.vert_df["dy"] = np.nan
        if "dz" not in sheet.vert_df.columns:
            sheet.vert_df["dz"] = np.nan

        edge_np = sheet.edge_df.to_numpy()
        edge_dict = dict(zip(sheet.edge_df.columns,
                             list(range(0, len(sheet.edge_df.columns)))))
        vert_np = sheet.vert_df.to_numpy()
        vert_dict = dict(zip(sheet.vert_df.columns,
                             list(range(0, len(sheet.vert_df.columns)))))

        grad1 = np.nan
        grad2 = np.nan
        gradt = np.nan
        lenth = []

        for i in sheet.vert_df.index:
            mask = (i == edge_np[:, edge_dict["srce"]])
            neighbor_verts = edge_np[mask, edge_dict["trgt"]].tolist()
            neighbor_verts = vert_np[neighbor_verts][:, [vert_dict["x"], vert_dict["y"], vert_dict["z"]]]
            vert_coords = vert_np[i, [vert_dict["x"], vert_dict["y"], vert_dict["z"]]]

            if len(neighbor_verts) >= 3:
                center = (neighbor_verts[0] + neighbor_verts[1] + neighbor_verts[2]) / 3

                grad = np.array(vert_coords) - np.array(center)

                length = np.linalg.norm(grad)

                grad = grad / length

            else:
                grad = np.array([0, 0, 0])

            if i == 0:
                grad1 = grad

            if i == 1:
                grad2 = grad
                gradt = np.vstack((grad1, grad2))

            if i >= 2:
                gradt = np.vstack((gradt, grad))

            lenth.append(length)

        gradt = pd.DataFrame(gradt, columns=['dx', 'dy', 'dz'])
        sheet.vert_df[['dx', 'dy', 'dz']] = gradt
        sheet.vert_df['dev_length'] = lenth

    @staticmethod
    def update_vert_deviation2(sheet):
        if "dev_length" not in sheet.vert_df.columns:
            sheet.vert_df["dev_length"] = np.nan
        if "dx" not in sheet.vert_df.columns:
            sheet.vert_df["dx"] = np.nan
        if "dy" not in sheet.vert_df.columns:
            sheet.vert_df["dy"] = np.nan
        if "dz" not in sheet.vert_df.columns:
            sheet.vert_df["dz"] = np.nan

        for i in sheet.vert_df.index:
            vert = sheet.vert_df.loc[i, ["x", "y", "z"]].to_numpy()
            neighbors = sheet.edge_df.loc[sheet.edge_df["srce"] == 6, ["tx", "ty", "tz"]].to_numpy()
            center = neighbors.sum(axis=0) / len(neighbors)
            grad = vert - center
            length = np.linalg.norm(grad)
            sheet.vert_df[['dx', 'dy', 'dz']] = grad
            sheet.vert_df['dev_length'] = length

    @staticmethod
    def update_lumen_vol(sheet):
        lumen_pos_faces = sheet.edge_df[["f" + c for c in sheet.coords]].to_numpy()
        lumen_sub_vol = (
                np.sum((lumen_pos_faces) * sheet.edge_df[sheet.ncoords].to_numpy(), axis=1)
                / 6
        )
        lumen_volume_gross = sum(lumen_sub_vol)

        top_verts = sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] > 0)]
        top_radius = top_verts["distance_z_axis"].values.mean()
        top_height = top_verts["z"].values.mean()
        top_volume = (1 / 3) * math.pi * top_radius ** 2 * top_height

        bot_verts = sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] < 0)]
        bot_radius = bot_verts["distance_z_axis"].values.mean()
        bot_height = -bot_verts["z"].values.mean()
        bot_volume = (1 / 3) * math.pi * bot_radius ** 2 * bot_height

        sheet.settings["lumen_vol"] = top_volume + bot_volume + lumen_volume_gross

    @staticmethod
    def update_boundary_radius(sheet):
        sheet.vert_df[["cx", "cy", "cz"]] = np.nan
        sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] <= 0), ["cx", "cy", "cz"]] = (
                    sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] <= 0), ["x", "y", "z"]] -
                    sheet.settings["bot_center"]).to_numpy()
        sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] >= 0), ["cx", "cy", "cz"]] = (
                    sheet.vert_df.loc[(sheet.vert_df["boundary"] == 1) & (sheet.vert_df["z"] >= 0), ["x", "y", "z"]] -
                    sheet.settings["top_center"]).to_numpy()
        sheet.vert_df["bound_rad"] = sheet.vert_df.eval("(cx**2 + cy**2 + cz**2) ** 0.5")
        sheet.vert_df[["cx", "cy", "cz"]] = sheet.vert_df[["cx", "cy", "cz"]].div(sheet.vert_df["bound_rad"], axis=0)
        sheet.vert_df.fillna(0, inplace=True)

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_boundary_index(sheet)
        cls.update_tangents(sheet)
        # cls.update_face_tangents(sheet)
        # cls.update_face_distance(sheet)
        cls.update_vert_distance(sheet)
        cls.update_vert_deviation(sheet)
        cls.update_lumen_vol(sheet)
        # cls.update_boundary_radius(sheet)


class CylinderGeometry(CylinderGeometryInit):

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_preflumen_volume(sheet)

    @staticmethod
    def update_preflumen_volume(sheet):
        sheet.settings["lumen_prefered_vol"] = sheet.settings["vol_cell"] * len(sheet.face_df)


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


class WeightedPerimeterEllipsoidLameGeometry(ClosedSheetGeometry):
    """
    EllipsoidLameGeometry correspond to a super-egg geometry with a calculation
    of perimeter is based on weight of each junction.

    Meaning if all junction of a cell have the same weight, perimeter is
    calculated as a usual perimeter calculation (p = l_ij + l_jk + l_km + l_mn + l_ni)
    Otherwise, weight parameter allowed more or less importance of a junction in the
    perimeter calculation (p = w_ij*l_ij + w_jk*l_jk + w_km*l_km + w_mn*l_mn + w_ni*l_ni)

    In this geometry, a sphere surrounding the tissue, meaning a force is apply only at
    the extremity of the tissue; `eptm.vert_df['delta_rho']` is computed as the
    difference between the vertex radius in a spherical frame of reference
    and `eptm.settings['barrier_radius']`

    """

    @classmethod
    def update_all(cls, eptm):
        cls.center(eptm)
        super().update_all(eptm)

    @staticmethod
    def update_perimeters(eptm):
        """
        Updates the perimeter of each face according to the weight of each junction.
        """
        eptm.edge_df["weighted_length"] = eptm.edge_df.weight * eptm.edge_df.length
        eptm.face_df["perimeter"] = eptm.sum_face(eptm.edge_df["weighted_length"])

    @staticmethod
    def normalize_weights(sheet):
        """
        Normalize weight of each cell.
        Sum of all weights of one cell equals to one.
        """
        sheet.edge_df["num_sides"] = sheet.upcast_face("num_sides")
        sheet.edge_df["weight"] = (
            sheet.edge_df.groupby("face")
            .apply(lambda df: (df["num_sides"] * df["weight"] / df["weight"].sum()))
            .sort_index(level=1)
            .to_numpy()
        )

    @staticmethod
    def update_height(eptm):
        eptm.vert_df["rho"] = np.linalg.norm(eptm.vert_df[eptm.coords], axis=1)
        r = eptm.settings["barrier_radius"]
        eptm.vert_df["delta_rho"] = (eptm.vert_df["rho"] - r).clip(0)
        eptm.vert_df["height"] = eptm.vert_df["rho"]


def face_svd_(faces):
    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(float), full_matrices=False)
    return rotation