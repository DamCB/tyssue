from cmath import sqrt
import numpy as np
import pandas as pd
import math

from .sheet_geometry import SheetGeometry
from .utils import rotation_matrix, rotation_matrices


class VesselGeometry(SheetGeometry):
    
    @staticmethod     
    def update_boundary_index(sheet):
    # Reset boundary flags
        sheet.vert_df['boundary'] = 0
        sheet.edge_df['boundary'] = 0
        sheet.face_df['boundary'] = 0

        # Update opposite edges
        sheet.get_opposite()

        # Identify boundary edges
        boundary_edges = sheet.edge_df['opposite'] == -1
        sheet.edge_df.loc[boundary_edges, 'boundary'] = 1

        # Set boundary vertices
        boundary_verts = sheet.edge_df.loc[boundary_edges, 'trgt']
        sheet.vert_df.loc[boundary_verts.unique(), 'boundary'] = 1

        # Set boundary faces
        boundary_faces = sheet.edge_df.loc[boundary_edges, 'face']
        sheet.face_df.loc[boundary_faces.dropna().unique().astype(int), 'boundary'] = 1

    @staticmethod
    def update_tangents(sheet): 
        
        vert_coords = sheet.vert_df[sheet.coords]
        vert_coords.loc[:, "z"] = 0
        vert_coords = vert_coords.values 
        normal = np.column_stack((np.zeros(sheet.Nv), np.zeros(sheet.Nv), np.ones(sheet.Nv)))
        
        tangent = np.cross(vert_coords, normal)
        tangent = pd.DataFrame(tangent)
        
        tangent.columns = ["t" + u for u in sheet.coords]
        
        
        length = pd.DataFrame(tangent.eval("sqrt(tx**2 + ty**2 +tz**2)"), columns = ['length'])
        tangent["length"] = length["length"]
        
        
        tangent = tangent[['tx','ty','tz']].div(length.length, axis=0)
        
        for u in sheet.coords:
            sheet.vert_df["t" + u] = tangent["t" + u]

    @classmethod
    def update_all(cls, sheet):
        super().update_all(sheet)
        cls.update_tangents(sheet)
        cls.update_boundary_index(sheet)         

def face_svd_(faces):

    rel_pos = faces[["rx", "ry", "rz"]]
    _, _, rotation = np.linalg.svd(rel_pos.astype(float), full_matrices=False)
    return rotation
