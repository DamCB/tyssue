"""
An epithelial sheet, i.e a 2D mesh in a 2D or 3D space,
akin to a HalfEdge data structure in CGAL.

For purely 2D the geometric properties are defined in
 `tyssue.geometry.planar_geometry`
A dynamical model derived from Fahradifar et al. 2007 is provided in
`tyssue.dynamics.planar_vertex_model`


For 2D in 3D, the geometric properties are defined in
 `tyssue.geometry.sheet_geometry`
A dynamical model derived from Monier, Gettings et al. 2015 is provided in
`tyssue.dynamics.sheet_vertex_model`


"""

import warnings
import numpy as np
import pandas as pd

from .objects import Epithelium
from ..config.geometry import flat_sheet


class Sheet(Epithelium):
    """
    An epithelial sheet, i.e a 2D mesh in a 2D or 3D space,
    akin to a HalfEdge data structure in CGAL.

    The geometric properties are defined in `tyssue.geometry.sheet_geometry`
    A dynamical model derived from Fahradifar et al. 2007 is provided in
    `tyssue.dynamics.sheet_vertex_model`


    """

    def __init__(self, identifier, datasets, specs=None, coords=None):
        """
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        face_df: `pandas.DataFrame` indexed by the faces indexes
            this df holds the vertices associated with

        """
        if specs is None:
            specs = flat_sheet()
        super().__init__(identifier, datasets, specs, coords)

    def reset_topo(self):
        super().reset_topo()
        if "opposite" in self.edge_df.columns:
            self.edge_df["opposite"] = get_opposite(self.edge_df)

    def get_opposite(self):
        self.edge_df["opposite"] = get_opposite(self.edge_df)

    def get_neighbors(self, face):
        """Returns the faces adjacent to `face`
        """
        return super().get_neighbors(face, elem="face")

    def get_neighborhood(self, face, order):
        """Returns `face` neighborhood up to a degree of `order`

        For example, if `order` is 2, it wil return the adjacent, faces
        and theses faces neighbors.

        Returns
        -------
        neighbors : pd.DataFrame with two colums, the index
            of the neighboring face, and it's neighboring order

        """
        # Start with the face so that it's not gathered later
        return super().get_neighborhood(face, order, elem="face")

    def get_extra_indices(self):
        """Computes extra indices:

        - `self.free_edges`: half-edges at the epithelium boundary
        - `self.dble_edges`: half-edges inside the epithelium,
          with an opposite
        - `self.east_edges`: half of the `dble_edges`, pointing east
          (figuratively)
        - `self.west_edges`: half of the `dble_edges`, pointing west
           (the order of the east and west edges is conserved, so that
           the ith west half-edge is the opposite of the ith east half-edge)
        - `self.sgle_edges`: joint index over free and east edges, spanning
           the entire graph without double edges
        - `self.wrpd_edges`: joint index over free edges followed by the
           east edges twice, such that a vector over the whole half-edge
            dataframe is wrapped over the single edges
        - `self.srtd_edges`: index over the whole half-edge sorted such that
           the free edges come first, then the east, then the west

        Also computes:
        - `self.Ni`: the number of inside full edges
          (i.e. `len(self.east_edges)`)
        - `self.No`: the number of outside full edges
          (i.e. `len(self.free_edges)`)
        - `self.Nd`: the number of double half edges
          (i.e. `len(self.dble_edges)`)
        - `self.anti_sym`: `pd.Series` with shape `(self.Ne,)`
          with 1 at the free and east half-edges and -1
          at the opposite half-edges.

        Notes
        -----

        - East and west is resepctive to some orientation at the
          moment the indices are computed the partition stays valid as
          long as there are no changes in the topology, so due to vertex
          displacement, 'east' and 'west' might not stay valid. This is
          just a practical naming convention.

        - As the name suggest, this method is not working for edges in
          3D pointing *exactly* north or south, ie iff `edge['dx'] ==
          edge['dy'] == 0`. Until we need or find a better solution,
          we'll just assert it worked.
        """

        if "opposite" not in self.edge_df.columns:
            self.edge_df["opposite"] = get_opposite(self.edge_df)

        self.dble_edges = self.edge_df[self.edge_df["opposite"] >= 0].index
        theta = np.arctan2(
            self.edge_df.loc[self.dble_edges, "dy"],
            self.edge_df.loc[self.dble_edges, "dx"],
        )

        self.east_edges = self.edge_df.loc[self.dble_edges][
            (theta > 0) & (theta <= np.pi)
        ].index
        self.west_edges = pd.Index(
            self.edge_df.loc[self.east_edges, "opposite"].astype(np.int), name="edge"
        )

        self.free_edges = self.edge_df[self.edge_df["opposite"] == -1].index
        self.sgle_edges = self.free_edges.append(self.east_edges)
        self.srtd_edges = self.sgle_edges.append(self.west_edges)

        # Index over the east and free edges, then the opposite indexed
        # by their east counterpart
        self.wrpd_edges = self.sgle_edges.append(self.east_edges)

        self.Ni = self.east_edges.size  # number of inside (east) edges
        self.Nd = self.dble_edges.size  # number of non free half edges
        self.No = self.free_edges.size  # number of free halfedges
        try:
            assert (2 * self.Ni + self.No) == self.Ne
            assert self.west_edges.size == self.Ni
            assert self.Nd == 2 * self.Ni
        # - BC -#
        # Not sure how to build
        # input data so the partition
        # fails (so we can see
        # if the exception is
        # correctly raised).
        # Leaving it in the coverage
        # anyway.
        except AssertionError:
            raise AssertionError(
                """
            Inconsistent partition:
            total half-edges: %s
            number of free: %s
            number of east: %s
            number of west: %s"""
                % (self.Ne, self.No, self.Ni, self.west_edges.size)
            )

        # Anti symetric vector (1 at east and free edges, -1 at opposite)
        self.anti_sym = pd.Series(np.ones(self.Ne), index=self.edge_df.index)
        self.anti_sym.loc[self.west_edges] = -1

    def sort_edges_eastwest(self):
        """reorder edges such the free edges are first,
        then the first half of the double edges, then the other half of
        the double edges, this way, each subset of the edges dataframe
        are contiguous.
        """
        self.get_extra_indices()
        self.edge_df = self.edge_df.loc[self.srtd_edges]
        self.reset_index(order=False)
        self.reset_topo()
        self.get_extra_indices()

    def extract(self, face_mask, coords=["x", "y", "z"]):
        """ Extract a new sheet from the sheet
        that correspond to a key word that define a face.

        Parameters
        ----------

        face_mask : column name in face composed by boolean value
        coords :

        Returns
        -------
        sheet_fold_patch_extract :
            subsheet corresponding to the fold patch area.

        """

        datasets = {}
        mask = self.face_df[face_mask].astype(bool)
        datasets["face"] = self.face_df[mask].copy()
        datasets["edge"] = self.edge_df[
            self.edge_df["face"].isin(datasets["face"].index)
        ].copy()
        datasets["vert"] = self.vert_df.loc[self.edge_df["srce"].unique()].copy()

        subsheet = Sheet("subsheet", datasets, self.specs)
        subsheet.reset_index()
        subsheet.reset_topo()
        return subsheet

    def extract_bounding_box(
        self, x_boundary=None, y_boundary=None, z_boundary=None, coords=["x", "y", "z"]
    ):
        """Extracts a new sheet from the embryo sheet

        that correspond to boundary coordinate define by the user.

        Parameters
        ----------
        x_boundary : pair of floats
        y_boundary : pair of floats
        z_boundary : pair of floats
        coords : list of strings, default ['x', 'y', 'z']
          coordinates over which to crop the sheet

        Returns
        -------
        subsheet : a new :class:`Sheet` object

        """
        x, y, z = coords
        datasets = {}
        datasets["face"] = self.face_df.copy()

        if x_boundary is not None:
            xmin, xmax = x_boundary
            datasets["face"] = datasets["face"][
                (datasets["face"][x] > xmin) & (datasets["face"][x] < xmax)
            ].copy()

        if y_boundary is not None:
            ymin, ymax = y_boundary
            datasets["face"] = datasets["face"][
                (datasets["face"][y] > ymin) & (datasets["face"][y] < ymax)
            ].copy()

        if z_boundary is not None:
            zmin, zmax = z_boundary
            datasets["face"] = datasets["face"][
                (datasets["face"][z] > zmin) & (datasets["face"][z] < zmax)
            ].copy()

        datasets["edge"] = self.edge_df[
            self.edge_df["face"].isin(datasets["face"].index)
        ].copy()

        datasets["vert"] = self.vert_df.loc[self.edge_df["srce"].unique()].copy()

        subsheet = Sheet("subsheet", datasets, self.specs)
        subsheet.reset_index()
        subsheet.reset_topo()
        return subsheet

    @classmethod
    def planar_sheet_2d(cls, identifier, nx, ny, distx, disty, noise=None):
        """Creates a planar sheet from an hexagonal grid of cells.

        Parameters
        ----------
        identifier : string
        nx, ny : int
          number of cells in the x and y axes
        distx, disty : float,
          the distances in x and y between the cells
        noise : float, default None
          position noise on the hexagonal grid

        Returns
        -------
        planar_sheet: a 2D :class:`Sheet` instance
          in the (x, y) plane

        """
        from scipy.spatial import Voronoi
        from ..config.geometry import planar_spec
        from ..generation import hexa_grid2d, from_2d_voronoi

        grid = hexa_grid2d(nx, ny, distx, disty, noise)
        datasets = from_2d_voronoi(Voronoi(grid))
        return cls(identifier, datasets, specs=planar_spec(), coords=["x", "y"])

    @classmethod
    def planar_sheet_3d(cls, identifier, nx, ny, distx, disty, noise=None):
        """Creates a planar sheet from an hexagonal grid of cells.

        Parameters
        ----------
        identifier : string
        nx, ny : int
          number of cells in the x and y axes
        distx, disty : float,
          the distances in x and y between the cells
        noise : float, default None
          position noise on the hexagonal grid

        Returns
        -------
        flat_sheet: a 2.5D :class:`Sheet` instance
        """

        from scipy.spatial import Voronoi
        from ..config.geometry import flat_sheet
        from ..generation import hexa_grid2d, from_2d_voronoi

        grid = hexa_grid2d(nx, ny, distx, disty, noise)
        datasets = from_2d_voronoi(Voronoi(grid))
        datasets["vert"]["z"] = 0
        datasets["face"]["z"] = 0

        return cls(identifier, datasets, specs=flat_sheet(), coords=["x", "y", "z"])


def get_opposite(edge_df):
    """
    Returns the indices opposite to the edges in `edge_df`
    """

    st_indexed = (
        edge_df[["srce", "trgt"]].reset_index().set_index(["srce", "trgt"], drop=False)
    )
    flipped = st_indexed.index.swaplevel(0, 1)
    flipped.names = ["srce", "trgt"]
    try:
        opposite = st_indexed.reindex(flipped)["edge"].values
    except ValueError:
        dup = flipped.duplicated()
        warnings.warn(
            "Duplicated (`srce`, `trgt`) values in edge_df, maybe sanitize your input"
        )
        opposite = st_indexed[~dup].reindex(flipped)["edge"].values
    opposite[np.isnan(opposite)] = -1
    return opposite.astype(np.int)


def get_outer_sheet(eptm):
    """Return a Sheet object formed by all the faces w/o an opposite
    face.
    """
    eptm.get_opposite_faces()
    is_free_face = eptm.face_df["opposite"] == -1
    is_free_edge = eptm.upcast_face(is_free_face)
    edge_df = eptm.edge_df[is_free_edge].copy()
    face_df = eptm.face_df[is_free_face].copy()
    vert_df = eptm.vert_df.loc[edge_df["srce"].unique()].copy()

    datasets = {"edge": edge_df, "face": face_df, "vert": vert_df}
    specs = {k: eptm.specs.get(k, {}) for k in ["face", "edge", "vert", "settings"]}

    return Sheet(eptm.identifier + "outer", datasets, specs)
