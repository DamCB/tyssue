import warnings
import numpy as np

from .sheet_geometry import SheetGeometry

from .utils import rotation_matrix
from ..utils import _to_3d
from ..core.sheet import Sheet


class BulkGeometry(SheetGeometry):
    """Geometry functions for 3D cell arangements
    """

    @classmethod
    def update_all(cls, eptm):
        """
        Updates the eptm geometry by updating:
        * the edge vector coordinates
        * the edge lengths
        * the face centroids
        * the normals to each edge associated face
        * the face areas
        * the cell areas
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        """
        cls.update_dcoords(eptm)
        cls.update_length(eptm)
        cls.update_perimeters(eptm)
        cls.update_centroid(eptm)
        cls.update_normals(eptm)
        cls.update_vol(eptm)
        cls.update_areas(eptm)

    @staticmethod
    def update_dcoords(eptm):
        SheetGeometry.update_dcoords(eptm)

    @staticmethod
    def update_vol(eptm):
        """

        """
        face_pos = eptm.edge_df[["f" + c for c in eptm.coords]].values
        cell_pos = eptm.edge_df[["c" + c for c in eptm.coords]].values

        eptm.edge_df["sub_vol"] = (
            np.sum((face_pos - cell_pos) * eptm.edge_df[eptm.ncoords].values, axis=1)
            / 6
        )
        eptm.cell_df["vol"] = eptm.sum_cell(eptm.edge_df["sub_vol"])

    @staticmethod
    def update_areas(eptm):

        # SheetGeometry.update_areas(eptm)
        eptm.edge_df["sub_area"] = (
            np.linalg.norm(eptm.edge_df[eptm.ncoords], axis=1) / 2
        )
        eptm.face_df["area"] = eptm.sum_face(eptm.edge_df["sub_area"])
        eptm.cell_df["area"] = eptm.sum_cell(eptm.edge_df["sub_area"])

    @staticmethod
    def update_centroid(eptm):
        scoords = ["s" + c for c in eptm.coords]
        eptm.face_df[eptm.coords] = eptm.edge_df.groupby("face")[scoords].mean()
        face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
        for c in eptm.coords:
            eptm.edge_df["f" + c] = face_pos[c]
            eptm.edge_df["r" + c] = eptm.edge_df["s" + c] - eptm.edge_df["f" + c]

        eptm.cell_df[eptm.coords] = eptm.edge_df.groupby("cell")[scoords].mean()
        cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords]).values
        eptm.edge_df[["c" + c for c in eptm.coords]] = cell_pos

    @staticmethod
    def validate_face_norms(eptm):

        fcoords = ["f" + c for c in eptm.coords]
        ccoords = ["c" + c for c in eptm.coords]

        face_pos = eptm.edge_df[fcoords].values
        cell_pos = eptm.edge_df[ccoords].values

        r_cf = face_pos - cell_pos
        r_cf["face"] = eptm.edge_df["face"]
        r_cf = r_cf.groupby("face").mean()
        face_norm = eptm.edge_df.groupby("face")[eptm.ncoords].mean()

        proj = (face_norm * r_cf.values).sum(axis=1)
        is_outward = proj >= 0
        return is_outward


class RNRGeometry(BulkGeometry):
    @staticmethod
    def update_centroid(eptm):
        scoords = ["s" + c for c in eptm.coords]
        tcoords = ["t" + c for c in eptm.coords]

        srce_pos = eptm.edge_df[scoords].values
        trgt_pos = eptm.edge_df[tcoords].values
        mid_pos = (srce_pos + trgt_pos) / 2
        weighted_pos = eptm.sum_face(mid_pos * _to_3d(eptm.edge_df["length"]))
        eptm.face_df[eptm.coords] = (
            weighted_pos.values / eptm.face_df["perimeter"].values[:, np.newaxis]
        )

        face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
        for c in eptm.coords:
            eptm.edge_df["f" + c] = face_pos[c]
            eptm.edge_df["r" + c] = eptm.edge_df["s" + c] - eptm.edge_df["f" + c]

        eptm.cell_df[eptm.coords] = eptm.edge_df.groupby("cell")[scoords].mean()
        cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords]).values
        eptm.edge_df[["c" + c for c in eptm.coords]] = cell_pos


class MonolayerGeometry(RNRGeometry):
    @staticmethod
    def basal_apical_axis(eptm, cell):
        """
        Returns a unit vector allong the apical-basal axis of the cell
        """
        edges = eptm.edge_df[eptm.edge_df["cell"] == cell]
        srce_segments = eptm.vert_df.loc[edges["srce"], "segment"]
        srce_segments.index = edges.index
        trgt_segments = eptm.vert_df.loc[edges["trgt"], "segment"]
        trgt_segments.index = edges.index
        ba_edges = edges[(srce_segments == "apical") & (trgt_segments == "basal")]
        return ba_edges[eptm.dcoords].mean()

    @classmethod
    def cell_projected_pos(cls, eptm, cell, psi=0):
        """Returns the positions of the cell vertices
        transformed such that the cell center sits at the
        coordinate system's origin and the basal-apical axis
        is the new `z` axis.
        """
        ab_axis = cls.basal_apical_axis(eptm, cell)
        n_xy = np.linalg.norm(ab_axis[["dx", "dy"]])
        theta = -np.arctan2(n_xy, ab_axis.dz)
        direction = [ab_axis.dy, -ab_axis.dx, 0]
        rot = rotation_matrix(theta, direction)
        cell_verts = set(eptm.edge_df[eptm.edge_df["cell"] == cell]["srce"])
        vert_pos = eptm.vert_df.loc[cell_verts, eptm.coords]
        for c in eptm.coords:
            vert_pos[c] -= eptm.cell_df.loc[cell, c]

        r1 = np.dot(vert_pos, rot)

        if abs(psi) < 1e-6:
            vert_pos[:] = r1
        else:
            vert_pos[:] = np.dot(rotation_matrix(psi, [0, 0, 1]), r1)
        return vert_pos


class MonoLayerGeometry(MonolayerGeometry):
    pass


class ClosedMonolayerGeometry(MonolayerGeometry):
    @classmethod
    def update_all(cls, eptm):
        """
        Updates the eptm geometry by updating:
        * the edge vector coordinates
        * the edge lengths
        * the face centroids
        * the normals to each edge associated face
        * the face areas
        * the cell areas
        * the vertices heights (depends on geometry)
        * the face volumes (depends on geometry)

        """
        super().update_all(eptm)
        cls.update_lumen_vol(eptm)

    @staticmethod
    def update_lumen_vol(eptm):
        """

        """
        lumen_edges = eptm.edge_df[
            eptm.edge_df.segment == eptm.settings.get("lumen_side", "basal")
        ].copy()
        lumen_pos_faces = lumen_edges[["f" + c for c in eptm.coords]].values
        lumen_sub_vol = (
            np.sum((lumen_pos_faces) * lumen_edges[eptm.ncoords].values, axis=1) / 6
        )
        eptm.settings["lumen_vol"] = -lumen_sub_vol.sum()
