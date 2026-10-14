from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .planar_geometry import update_boundary_index
from .sheet_geometry import SheetGeometry

class CylinderGeometryInit(SheetGeometry):

    @staticmethod
    def update_boundary_index(sheet):
        # face_df 'boundary' is left unchanged
        update_boundary_index(sheet, with_faces=False)

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
        face_coords["z"] = 0.0
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
        """Writes in `vert_df` the deviation of each vertex from the mean
        position of its neighbours: its length in `dev_length` and the unit
        vector from that mean to the vertex in `dx, dy, dz`, as read by
        `SurfaceElasticity`.

        Neighbours linked by an edge in either direction are counted. A vertex
        without neighbours gets a zero deviation.
        """
        vert_df, edge_df = sheet.vert_df, sheet.edge_df
        pos = vert_df[["x", "y", "z"]].to_numpy(dtype=float)
        n_verts = len(pos)
        srce = vert_df.index.get_indexer(edge_df["srce"].to_numpy())
        trgt = vert_df.index.get_indexer(edge_df["trgt"].to_numpy())

        # each (vertex, neighbour) pair once, whichever way the edge runs
        linked = (srce >= 0) & (trgt >= 0) & (srce != trgt)
        s, t = srce[linked].astype(np.int64), trgt[linked].astype(np.int64)
        pairs = np.unique(np.concatenate([s * n_verts + t, t * n_verts + s]))
        vert, neighbor = pairs // n_verts, pairs % n_verts

        neighbor_sum = np.zeros_like(pos)
        np.add.at(neighbor_sum, vert, pos[neighbor])
        degree = np.bincount(vert, minlength=n_verts)
        has_neighbors = degree > 0

        deviation = np.zeros_like(pos)
        deviation[has_neighbors] = (
            pos[has_neighbors] - neighbor_sum[has_neighbors] / degree[has_neighbors, None]
        )
        dev_length = np.linalg.norm(deviation, axis=1)
        grad = np.zeros_like(pos)
        nonzero = dev_length > 0
        grad[nonzero] = deviation[nonzero] / dev_length[nonzero, None]

        vert_df[["dx", "dy", "dz"]] = grad
        vert_df["dev_length"] = dev_length

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
    def update_vol_cell(sheet):
        sheet.settings["vol_cell"] = sheet.settings["lumen_vol"]/len(sheet.face_df)

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
        cls.update_tangents(sheet)
        # cls.update_face_tangents(sheet)
        # cls.update_face_distance(sheet)
        cls.update_vert_distance(sheet)
        cls.update_vert_deviation(sheet)
        cls.update_lumen_vol(sheet)
        # cls.update_vol_cell(sheet)
        # cls.update_boundary_radius(sheet)


class CylinderGeometry(CylinderGeometryInit):

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        # cls.update_preflumen_volume(sheet)

    @staticmethod
    def update_preflumen_volume(sheet):
        sheet.settings["lumen_prefered_vol"] = sheet.settings["vol_cell"] * len(sheet.face_df)