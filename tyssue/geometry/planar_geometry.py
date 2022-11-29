import numpy as np

from .base_geometry import BaseGeometry


class PlanarGeometry(BaseGeometry):
    """Geomtetry methods for 2D planar cell arangements"""

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
        cls.update_repulsion(sheet)

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
    def update_repulsion(sheet):
        # Create globale grid
        grid = np.mgrid[np.min(sheet.vert_df['x'])-0.1:np.max(sheet.vert_df['x']) + 0.1:0.1,
               np.min(sheet.vert_df['y'])-0.1:np.max(sheet.vert_df['y']) + 0.1:0.1]

        shape = list(grid[0].shape)
        shape.append(sheet.Nf)
        face_repulsion = np.zeros(shape)

        for f in range(sheet.Nf):
            Z_gauss = gaussian_repulsion(grid,
                                         sheet,
                                         f,
                                         1)
            face_repulsion[:, :, f] = Z_gauss

        sheet.vert_df['repulse_u'] = 0.
        sheet.vert_df['repulse_v'] = 0.
        for v in range(sheet.Nv):
            faces = sheet.edge_df[sheet.edge_df["srce"] == v]['face'].to_numpy()
            v_repulsion = np.sum(face_repulsion[:, :, np.delete(np.arange(sheet.Nf), (faces))], axis=2)
            U, V = calculate_vector_field(v_repulsion)
            sheet.vert_df.loc[v, 'repulse_u'] = U[np.where(np.isclose(grid[0], sheet.vert_df.loc[v, 'x'], rtol=0.01, atol=0.1))[0][0],
                                                  np.where(np.isclose(grid[1], sheet.vert_df.loc[v, 'y'], rtol=0.01, atol=0.1))[1][0]]
            sheet.vert_df.loc[v, 'repulse_v'] = V[np.where(np.isclose(grid[0], sheet.vert_df.loc[v, 'x'], rtol=0.01, atol=0.1))[0][0],
                                                  np.where(np.isclose(grid[1], sheet.vert_df.loc[v, 'y'], rtol=0.01, atol=0.1))[1][0]]


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
        if "rx" not in sheet.edge_df:
            cls.update_dcoords(sheet)
            cls.update_centroid(sheet)

        return np.arctan2(sheet.edge_df["ry"], sheet.edge_df["rx"])


class AnnularGeometry(PlanarGeometry):
    """ """

    @classmethod
    def update_all(cls, eptm):
        PlanarGeometry.update_all(eptm)
        cls.update_lumen_volume(eptm)

    @staticmethod
    def update_lumen_volume(eptm):
        srce_pos = eptm.upcast_srce(eptm.vert_df[["x", "y"]]).loc[eptm.apical_edges]
        trgt_pos = eptm.upcast_trgt(eptm.vert_df[["x", "y"]]).loc[eptm.apical_edges]
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
    calculated as a usual perimeter calculation:
    .. math::
        p = \\sum_{ij} l_{ij}

    Otherwise, weight parameter allowed more or less importance of a junction in the
    perimeter calculation
    .. math::
        p = \\sum_{ij} w_{ij} \\, l_{ij}

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

# need to find solution to this duplication to avoid circular import
def _ordered_edges(face_edges):
    """Returns "srce", "trgt" and "face" indices
    organized clockwise for each face.

    Parameters
    ----------
    face_edges: `pd.DataFrame`
        exerpt of an edge_df for a single face

    Returns
    -------
    edges: list of 3 ints
        srce, trgt, face indices, ordered
    """
    face_edges = face_edges.copy()
    face_edges["edge"] = face_edges.index
    srces, trgts, faces, edge = face_edges[["srce", "trgt", "face", "edge"]].values.T
    srce, trgt, face_, edge_ = srces[0], trgts[0], faces[0], edge[0]
    edges = [[srce, trgt, face_, edge_]]
    for face_ in faces[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        edge_ = face_edges[(face_edges["srce"] == srce)]["edge"].to_numpy()[0]
        edges.append([srce, trgt, face_, edge_])
    return edges

from skimage.draw import line_aa
from scipy import ndimage
def gaussian_repulsion(grid, sheet, face, sigma):
    """
    Créer un profil de repulsion générique qui va être utilisé pour toutes les cellules.
    Parameters
    ----------
    width : prendre plus grand qu'une largeur moyenne de cellule

    Returns
    -------
    X : np.array of x position
    Y : np.array of y position
    Z : np.array of field repulsion value
    """
    X, Y = grid
    x_center = sheet.face_df.loc[face]['x']
    y_center = sheet.face_df.loc[face]['y']
    Z = np.zeros(X.shape)
    Z[:] = Z[:] + np.exp(- ((X - x_center) / sigma) ** 2 - ((Y - y_center) / sigma) ** 2)

    # apply cell mask
    Z_MASK = np.zeros(Z.shape)
    pos_v = list(
        np.array(_ordered_edges(sheet.edge_df[sheet.edge_df['face'] == face][['srce', 'trgt', 'face']])).flatten()[
        1::4])
    pos_v.append(pos_v[0])
    for p in range(len(pos_v) - 1):
        # rr, cc, val = line_aa(np.where(Y == sheet.vert_df.loc[pos_v[p], 'y'])[1][0],
        #                       np.where(X == sheet.vert_df.loc[pos_v[p], 'x'])[0][0],
        #                       np.where(Y == sheet.vert_df.loc[pos_v[p + 1], 'y'])[1][0],
        #                       np.where(X == sheet.vert_df.loc[pos_v[p + 1], 'x'])[0][0])
        rr, cc, val = line_aa(np.where(np.isclose(Y, sheet.vert_df.loc[pos_v[p], 'y'], rtol=0.01, atol=0.1))[1][0],
                              np.where(np.isclose(X, sheet.vert_df.loc[pos_v[p], 'x'], rtol=0.01, atol=0.1))[0][0],
                              np.where(np.isclose(Y, sheet.vert_df.loc[pos_v[p+1], 'y'], rtol=0.01, atol=0.1))[1][0],
                              np.where(np.isclose(X, sheet.vert_df.loc[pos_v[p+1], 'x'], rtol=0.01, atol=0.1))[0][0])
        Z_MASK[cc, rr] = 1

    Z_MASK = ndimage.binary_fill_holes(Z_MASK).astype(int)

    Z = Z * Z_MASK

    return Z


def calculate_vector_field(Z):
    U, V = np.gradient(Z, 1, 1)
    U = U
    V = V

    return U, V