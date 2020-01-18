import numpy as np

try:
    from .cpp import c_collisions
except ImportError:
    print(
        "collision solver could not be imported "
        "You may need to install CGAL and re-install tyssue"
    )
    c_collisions = None


def self_intersections(sheet):
    """Checks for self collisions for the sheet

    Parameters
    ----------
    sheet : a :class:`Sheet` object
        This object must have a `triangular_mesh` method returning a
        valid triangular mesh.

    Returns
    -------
    edge_pairs: np.ndarray of indices
         Array of shape (n_intersections, 2) with the indices of the
         pairs of intersecting edges
    """
    faces, vertices = sheet.triangular_mesh(sheet.coords, return_mask=False)
    mesh = c_collisions.sheet_to_surface_mesh(faces, vertices)
    if not c_collisions.does_self_intersect(mesh):
        return np.empty((0, 2), dtype=np.int)
    return np.array(c_collisions.self_intersections(mesh), dtype=np.int)
