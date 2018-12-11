import numpy as np
from .cpp import c_collisions


def self_intersections(sheet):
    faces, vertices = sheet.triangular_mesh(sheet.coords, return_mask=False)
    mesh = c_collisions.sheet_to_surface_mesh(faces, vertices)
    if not c_collisions.does_self_intersect(mesh):
        return np.empty((0, 2), dtype=np.int)
    return np.array(c_collisions.self_intersections(mesh), dtype=np.int)
