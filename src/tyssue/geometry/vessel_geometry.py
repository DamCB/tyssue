from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .sheet_geometry import SheetGeometry
from .utils import rotation_matrix, rotation_matrices


class VesselGeometry(SheetGeometry):

    @staticmethod
    def update_tangents(sheet):
        x = sheet.vert_df['x'].to_numpy()
        y = sheet.vert_df['y'].to_numpy()

        # tangent: (x, y, 0) x z
        tx = y
        ty = -x

        length = np.hypot(tx, ty)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_length = 1.0 / length

        sheet.vert_df['tx'] = tx * inv_length
        sheet.vert_df['ty'] = ty * inv_length
        sheet.vert_df['tz'] = 0.0

    @staticmethod
    def update_vert_distance(sheet):
        coords = ["x", "y", "z"]
        if "axis" in sheet.settings.keys():
            axis = sheet.settings["axis"]
        else:
            axis = "z"
        coords.remove(axis)
        distances = np.sqrt(sheet.vert_df[coords[0]].to_numpy()**2 + sheet.vert_df[coords[1]].to_numpy()**2)
        sheet.vert_df['distance_origin'] = distances
        sheet.vert_df["o"+f"{coords[0]}"] = sheet.vert_df[f"{coords[0]}"]/distances
        sheet.vert_df["o"+f"{coords[1]}"] = sheet.vert_df[f"{coords[1]}"]/distances

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_tangents(sheet)
        cls.update_vert_distance(sheet)

def face_svd_(faces):

    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(float), full_matrices=False)
    return rotation
