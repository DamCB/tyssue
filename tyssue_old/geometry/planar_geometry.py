import numpy as np
from .base_geometry import BaseGeometry


class PlanarGeometry(BaseGeometry):
    """Geomtetry methods for 2D planar cell arangements
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
        """

        cls.update_dcoords(sheet)
        cls.update_ucoords(sheet)
        cls.update_length(sheet)
        cls.update_centroid(sheet)
        cls.update_normals(sheet)
        cls.update_areas(sheet)
        cls.update_perimeters(sheet)

    @staticmethod
    def update_normals(sheet):

        rcoords = ["r" + c for c in sheet.coords]
        dcoords = ["d" + c for c in sheet.coords]

        normals = np.cross(sheet.edge_df[rcoords], sheet.edge_df[dcoords])
        sheet.edge_df["nz"] = normals

    @staticmethod
    def update_areas(sheet):
        """
        Updates the normal coordinate of each (srce, trgt, face) face.
        """
        sheet.edge_df["sub_area"] = sheet.edge_df["nz"] / 2
        sheet.face_df["area"] = sheet.sum_face(sheet.edge_df["sub_area"])

    @staticmethod
    def face_projected_pos(sheet, face, psi):
        """
        returns the sheet vertices position translated to center the face
        `face` at (0, 0) and rotated in the (x, y) plane
        by and angle `psi` radians

        """
        rot_pos = sheet.vert_df[sheet.coords].copy()
        face_x, face_y = sheet.face_df.loc[face, ["x", "y"]]
        rot_pos.x = (sheet.vert_df.x - face_x) * np.cos(psi) - (
            sheet.vert_df.y - face_y
        ) * np.sin(psi)
        rot_pos.y = (sheet.vert_df.x - face_x) * np.sin(psi) + (
            sheet.vert_df.y - face_y
        ) * np.cos(psi)

        return rot_pos

    @classmethod
    def get_phis(cls, sheet):
        if not "rx" in sheet.edge_df:
            cls.update_dcoords(sheet)
            cls.update_centroid(sheet)

        return np.arctan2(sheet.edge_df["ry"], sheet.edge_df["rx"])


# The following classes will probably be included in tyssue at some point
class AnnularGeometry(PlanarGeometry):
    """
    """

    @classmethod
    def update_all(cls, eptm):
        PlanarGeometry.update_all(eptm)
        cls.update_lumen_volume(eptm)

    @staticmethod
    def update_lumen_volume(eptm):
        srce_pos = eptm.upcast_srce(eptm.vert_df[["x", "y"]]).loc[
            eptm.apical_edges]
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[["x", "y"]]).loc[
            eptm.apical_edges]
        apical_edge_pos = (srce_pos + trgt_pos) / 2
        apical_edge_coords = eptm.edge_df.loc[eptm.apical_edges, ["dx", "dy"]]
        eptm.settings["lumen_volume"] = (
            -apical_edge_pos["x"] * apical_edge_coords["dy"]
            + apical_edge_pos["y"] * apical_edge_coords["dx"]
        ).values.sum()


class WeightedPerimeterPlanarGeometry(PlanarGeometry):
    """
    Geometry methods for 2D planar cell arangements with a calculation
    of perimeter is based on weight of each junction.

    Meaning if all junction of a cell have the same weight, perimeter is
    calculated as a usual perimeter calculation (p = l_ij + l_jk + l_km + l_mn + l_ni)
    Otherwise, weight parameter allowed more or less importance of a junction in the
    perimeter calculation (p = w_ij*l_ij + w_jk*l_jk + w_km*l_km + w_mn*l_mn + w_ni*l_ni)

    """

    @classmethod
    def update_all(cls, eptm):
        cls.center(eptm)
        cls.normalize_weights(eptm)
        super().update_all(eptm)

    @staticmethod
    def update_perimeters(eptm):
        """
        Updates the perimeter of each face according to the weight of each junction.
        """
        eptm.edge_df["weighted_length"] = eptm.edge_df.weight * \
            eptm.edge_df.length
        eptm.face_df["perimeter"] = eptm.sum_face(
            eptm.edge_df["weighted_length"])

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
