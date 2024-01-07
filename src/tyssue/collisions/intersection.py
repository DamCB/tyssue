import numpy as np

try:
    from tyssue_cpp import c_collisions
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
    vertices, faces = sheet.triangular_mesh(sheet.coords, return_mask=False)
    if vertices[0].shape[0] == 2:
        vertices = np.array([list(np.append(vert, 0)) for vert in vertices])
    mesh = c_collisions.sheet_to_surface_mesh(vertices, faces)
    if not c_collisions.does_self_intersect(mesh):
        return np.empty((0, 2), dtype=int)
    return np.array(c_collisions.self_intersections(mesh), dtype=int)
