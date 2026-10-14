import zlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base_geometry import BaseGeometry
from skimage.draw import polygon


@dataclass
class BoundaryIndex:
    """Positional arrays describing the free border of an epithelium.

    Stored on ``eptm._boundary`` by :meth:`PlanarGeometry.update_boundary_index`
    and reused until the topology changes.

    free_edges  (k,) int  row positions in edge_df of half-edges with no opposite
    free_srce   (k,) int  row positions in vert_df of their source vertices
    free_trgt   (k,) int  row positions in vert_df of their target vertices
    free_face   (k,) int  row positions in face_df of their faces
    verts       (m,) int  row positions in vert_df of the vertices on the border
    with_faces  bool      whether face_df['boundary'] was written too
    """

    signature: tuple
    free_edges: np.ndarray
    free_srce: np.ndarray
    free_trgt: np.ndarray
    free_face: np.ndarray
    verts: np.ndarray
    with_faces: bool = True


def topology_signature(eptm):
    """A cheap fingerprint of the epithelium's connectivity.

    Combines the element counts, a checksum of the edge connectivity and
    indices (which changes on a type 1 transition), and the dataframes' ids.
    """
    edge_df, vert_df, face_df = eptm.edge_df, eptm.vert_df, eptm.face_df
    crc = 0
    for arr in (
        edge_df["srce"].to_numpy(),
        edge_df["trgt"].to_numpy(),
        edge_df["face"].to_numpy(),
        edge_df.index.to_numpy(),
        vert_df.index.to_numpy(),
        face_df.index.to_numpy(),
    ):
        if arr.dtype.hasobject:  # no buffer to checksum, hash the values
            arr = pd.util.hash_array(arr)
        crc = zlib.crc32(np.ascontiguousarray(arr), crc)
    return (
        len(vert_df),
        len(edge_df),
        len(face_df),
        crc,
        id(vert_df),
        id(edge_df),
        id(face_df),
    )


def update_boundary_index(sheet, with_faces=True):
    """Implementation of :meth:`PlanarGeometry.update_boundary_index`.

    With ``with_faces=False`` the face_df 'boundary' column is left untouched.
    """
    sig = topology_signature(sheet)
    cached = getattr(sheet, "_boundary", None)
    if (
        cached is not None
        and cached.signature == sig
        and "boundary" in sheet.vert_df.columns
        and "boundary" in sheet.edge_df.columns
        and "opposite" in sheet.edge_df.columns
        and (not with_faces or (cached.with_faces and "boundary" in sheet.face_df.columns))
    ):
        return

    # avoids a circular import
    from ..core.sheet import get_opposite

    edge_df = sheet.edge_df
    opposite = get_opposite(edge_df)
    edge_df["opposite"] = opposite
    free = opposite == -1
    free_edges = np.flatnonzero(free)

    free_srce = sheet.vert_df.index.get_indexer(edge_df["srce"].to_numpy()[free])
    free_trgt = sheet.vert_df.index.get_indexer(edge_df["trgt"].to_numpy()[free])
    free_face = sheet.face_df.index.get_indexer(edge_df["face"].to_numpy()[free])

    # border vertices are the targets of free edges
    verts = np.unique(free_trgt[free_trgt >= 0])

    vert_flag = np.zeros(len(sheet.vert_df), dtype=np.int64)
    vert_flag[verts] = 1

    edge_df["boundary"] = free.astype(np.int64)
    sheet.vert_df["boundary"] = vert_flag
    if with_faces:
        face_flag = np.zeros(len(sheet.face_df), dtype=np.int64)
        face_flag[free_face[free_face >= 0]] = 1
        sheet.face_df["boundary"] = face_flag

    sheet._boundary = BoundaryIndex(
        signature=sig,
        free_edges=free_edges,
        free_srce=free_srce,
        free_trgt=free_trgt,
        free_face=free_face,
        verts=verts,
        with_faces=with_faces,
    )


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
        * the boundary index (only recomputed when the topology changed)
        """

        cls.update_dcoords(sheet)
        cls.update_length(sheet)
        cls.update_ucoords(sheet)  # needs the current lengths
        cls.update_centroid(sheet)
        cls.update_normals(sheet)
        cls.update_areas(sheet)
        cls.update_perimeters(sheet)
        cls.update_boundary_index(sheet)
        # cls.update_repulsion(sheet)

    @staticmethod
    def update_boundary_index(sheet):
        """Updates the vert_df, edge_df and face_df dataframes with a 'boundary'
        column that takes values 0 and 1, with 1 denoting that a vertex, edge or
        face lies on the tissue boundary, and 0 when it does not.

        Also updates ``edge_df["opposite"]`` and stores a :class:`BoundaryIndex`
        on ``sheet._boundary``. Does nothing if the topology has not changed.
        """
        update_boundary_index(sheet, with_faces=True)

    @staticmethod
    def update_normals(sheet):

        rcoords = ["r" + c for c in sheet.coords]
        dcoords = ["d" + c for c in sheet.coords]

        # z component of the 2D cross product
        rx, ry = (sheet.edge_df[c] for c in rcoords)
        dx, dy = (sheet.edge_df[c] for c in dcoords)
        sheet.edge_df["nz"] = rx * dy - ry * dx

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
        grid = np.mgrid[np.min(sheet.vert_df['x']) - 0.1:np.max(sheet.vert_df['x']) + 0.1:0.1,
               np.min(sheet.vert_df['y']) - 0.1:np.max(sheet.vert_df['y']) + 0.1:0.1]
        face_repulsion = gaussian_repulsion(grid, sheet)

        sheet.vert_df['v_repulsion'] = 0
        sheet.vert_df['grid'] = 0
        sheet.vert_df['v_repulsion'] = sheet.vert_df['v_repulsion'].astype(object)
        sheet.vert_df['grid'] = sheet.vert_df['grid'].astype(object)
        for v in range(sheet.Nv):
            faces = sheet.edge_df[sheet.edge_df["srce"] == v]['face'].to_numpy()
            sum_ = np.sum(face_repulsion, axis=2)
            sub_ = np.sum(face_repulsion[:, :, faces], axis=2)
            v_repulsion = sum_ - sub_
            sheet.vert_df.loc[v, 'v_repulsion'] = [v_repulsion]
            sheet.vert_df.loc[v, 'grid'] = [grid]
            v_repulsion = None
            del v_repulsion

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

    # update_all re-centers the vertices, which moves them by rounding errors
    idempotent_update = False

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
            sheet.edge_df.groupby("face")[["num_sides", "weight"]]
            .apply(lambda df: (df["num_sides"] * df["weight"] / df["weight"].sum()))
            .sort_index(level=1)
            .to_numpy()
        )


def gaussian_repulsion(grid, sheet):
    """
    Parameters
    ----------

    Returns
    -------
    X : np.array of x position
    Y : np.array of y position
    Z : np.array of field repulsion value
    """
    shape = list(grid[0].shape)
    shape.append(sheet.Nf)
    face_repulsion = np.zeros(shape)
    face_ord_edges = sheet.ordered_edges()

    for face in range(sheet.Nf):
        # apply cell mask
        pos_v = list(np.array(face_ord_edges[face]).flatten()[1::4])
        pos_v.append(pos_v[0])

        xx = np.argmin(
            np.abs([grid[0][:, 0] - x for x in sheet.vert_df.loc[pos_v, "x"]]), axis=1
        )
        yy = np.argmin(
            np.abs([grid[1][0, :] - y for y in sheet.vert_df.loc[pos_v, "y"]]), axis=1
        )

        rr, cc = polygon(xx, yy)
        face_repulsion[rr, cc, face] = 1
    return face_repulsion
