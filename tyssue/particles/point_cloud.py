"""Utilities to generate point clouds

The positions of the points are generated along the architecture
of the epithelium.
"""
from collections import abc

import numpy as np
import pandas as pd
from ..config.subdiv import bulk_spec


class EdgeSubdiv:
    """
    Container class to ease discretisation along the edges

    """

    def __init__(self, edge_df, **kwargs):
        """Creates an indexer and an offset array  to ease
        discretisation along the edges.

        Parameters
        ----------
        edge_df: pd.DataFrame,

        Keyword parameters
        ------------------
        density: number of points per edge


        Attributes
        ----------

        upcaster: np.ndarray, shape (Np,)
          edge indices repeated to match the lookup table
        offset: np.ndarray, shape (Np,)
          piecewise linear offset along the edges, such that
          ::math:M_{ij}^n = offset[n]*r_{ij}:

        """

        self.edge_df = edge_df.copy()
        self.n_edges = self.edge_df.shape[0]
        self.specs = bulk_spec()
        self.specs.update(**kwargs)
        self.unit_length = edge_df.length.mean()

        if "density" not in edge_df:
            self.edge_df["density"] = self.specs["density"]
        self.n_points = 0
        self.points = None
        self.offset_lut = None
        self.update_all()

    def update_all(self):
        self.update_offset_lut()
        self.update_particles()
        self.update_upcaster()
        self.update_offset()

    @staticmethod
    def from_eptm_edges(eptm, **kwargs):
        """Creates an EdgeSubdiv instance and computes the point
        grid allong the edges from the source vertex to its target.
        Returns
        -------
        subdiv: a :class:`EdgeSubdiv` instance

        """

        subdiv = EdgeSubdiv(eptm.edge_df[["length", "density"]], **kwargs)
        srce_pos = eptm.upcast_srce(eptm.vert_df[eptm.coords])
        r_ij = eptm.edge_df[eptm.dcoords]
        subdiv.edge_point_cloud(srce_pos, r_ij)
        return subdiv

    @staticmethod
    def _offset_lut_(num):
        return np.arange(0.5, num + 0.5) / num

    def update_offset_lut(self, offset_lut=None):
        """
        Updates the density lookup table function.

        The `offset_lut` can be any function
        with a single `num` argument


        Parameters
        ----------
        offset_lut: function, default None,
          edge-wise function of the number of points
          giving the offset positions
        Default is a shifted regular grid:
        `np.arange(0.5, num+0.5) / num`

        """
        if offset_lut is None:
            self.offset_lut = self._offset_lut_
        else:
            self.offset_lut = offset_lut

    def update_particles(self):
        """
        * Updates the number of particles per edge from edges length
        and density values:
        `num_particles = length * density`
        * Also updates the self.points df
        """
        self.edge_df["norm_length"] = self.edge_df["length"] / self.unit_length
        points_per_edges = np.round(self.edge_df.eval("norm_length * density")).astype(
            np.int
        )
        self.edge_df["num_particles"] = points_per_edges
        self.n_points = points_per_edges.sum()
        self.points = pd.DataFrame(
            np.zeros((self.n_points, 2)), columns=["upcaster", "offset"]
        )

    def update_upcaster(self):
        """
        resets the 'upcaster' column of self.points,

        'upcaster' indexes over self.edge_df repeated to
        upcast data from the edge df to the points df
        """
        self.points["upcaster"] = np.repeat(
            np.arange(self.edge_df.shape[0]), self.edge_df["num_particles"]
        )

    def update_offset(self):
        self.points["offset"] = np.concatenate(
            [self.offset_lut(num=ns) for ns in self.edge_df["num_particles"]]
        )

    def validate(self):

        if not self.points["upcaster"].max() + 1 == self.n_edges:
            return False
        if not self.points["upcaster"].shape[0] == self.n_points:
            return False
        if not self.points["offset"].shape()[0] == self.n_points:
            return False
        return True

    def upcast(self, df):
        if isinstance(df, str) and df in self.edge_df:
            return self.edge_df.loc[self.points["upcaster"], df]
        elif (
            isinstance(df, abc.Iterable)
            and isinstance(df[0], str)
            and set(df).issubset(self.edge_df.columns)
        ):
            return self.edge_df.loc[self.points["upcaster"], df]
        elif hasattr(df, "loc"):
            return df.loc[self.points["upcaster"]]
        else:
            raise ValueError(
                """
Argument df should be a column name or a sequence of column names
or a Series or Dataframe indexed like self.edge_df
                """
            )

    def edge_point_cloud(
        self,
        srce_pos,
        r_ij,
        offset_modulation=None,
        modulation_kwargs=None,
        coords=["x", "y", "z"],
        dcoords=["dx", "dy", "dz"],
    ):
        """Generates a point cloud along the edges of the epithelium.

        if a offset_modulation function is provided, it is used to
        transform the offsets

        Parameters
        ----------
        srce_pos: DataFrame of shape (self.Ne, ndim)
          with the origins of the points for each edge
          (usually the edge upcasted source vertex)

        r_ij: DataFrame of shape (self.Ne, ndim)
          the edge vector coordiantes

        offset_modulation: function of self returning
          an array with shape (self.Np,) containing
          the modified offsets. self.points['offset']
          is used by default.
        modulation_kwargs: keyword arguments to the modulation
          function

        Returns
        -------
        points: (Np, 3) pd.DataFrame with the points positions
        """
        for u, du in zip(coords, dcoords):
            self.edge_df[u] = srce_pos[u]
            self.edge_df["d" + u] = r_ij[du]
        cols = coords + dcoords
        upcast = self.edge_df.loc[self.points["upcaster"], cols]
        if offset_modulation is None:
            upcast["offset"] = self.points["offset"].values
        else:
            upcast["offset"] = offset_modulation(self, **modulation_kwargs)
        for c in coords:
            self.points[c] = upcast.eval("{} + offset * {}".format(c, "d" + c)).values
        if self.specs["noise"] > 0.0:
            self.points[coords] += np.random.normal(
                scale=self.specs["noise"], size=(self.n_points, 3)
            )
        return self.points[coords]


def get_edge_bases(eptm, base=("face", "srce", "trgt")):

    edge_upcast_pos = {
        element: eptm.upcast_cols(element, eptm.coords) for element in base
    }
    origin = base[0]
    edge_bases = {}
    for vertex in base[1:]:
        df = pd.DataFrame(
            0, columns=eptm.coords + eptm.dcoords + ["length"], index=eptm.edge_df.index
        )
        df[eptm.dcoords] = (
            edge_upcast_pos[vertex].values - edge_upcast_pos[origin].values
        )

        df["length"] = np.linalg.norm(df[eptm.dcoords].values, axis=1)
        df[eptm.coords] = edge_upcast_pos[origin].values
        edge_bases["{}_{}".format(origin, vertex)] = df.copy()
    return edge_bases


class FaceGrid:
    def __init__(self, edges_df, base, **kwargs):

        self.origin = base[0]
        self.base = ["{}_{}".format(base[0], other) for other in base[1:]]

        self.specs = bulk_spec()
        self.specs.update(kwargs)
        e_specs = kwargs
        self.subdivs = {key: EdgeSubdiv(edges_df[key], **e_specs) for key in self.base}
        self.n_points = np.product(
            [
                subdiv.edge_df["num_particles"].values
                for subdiv in self.subdivs.values()
            ],
            axis=0,
        ).sum()
        self.dim = len(self.subdivs)
        self.up_cols = ["up_{}".format(key) for key in self.base]
        self.of_cols = ["of_{}".format(key) for key in self.base]
        self.points = None

    def update_grid(self):
        upcasters = {}
        for key, subdiv in self.subdivs.items():
            upcasters["up_" + key] = subdiv.points["upcaster"]
            upcasters["of_" + key] = subdiv.points["offset"]
        upcasters = pd.DataFrame.from_dict(upcasters)
        points = {}
        u_axis = "up_{}".format(self.base[0])
        upcasters.set_index(u_axis, drop=False, inplace=True)
        for cols in (self.of_cols, self.up_cols):
            df = upcasters.groupby(level=u_axis).apply(_local_grid, *cols)
            points.update({col: df[col].values for col in cols})
        self.points = pd.DataFrame.from_dict(points)

    def face_point_cloud(self, coords=["x", "y", "z"], dcoords=["dx", "dy", "dz"]):
        upcast = {}
        offsets = self.points[self.of_cols]
        for key, subdiv in self.subdivs.items():
            upcast[key] = subdiv.edge_df.loc[
                self.points["up_{}".format(key)], coords + dcoords + ["length"]
            ].copy()
            upcast[key].reset_index(inplace=True)
            upcast[key]["offset"] = offsets["of_{}".format(key)].values

        for u, du in zip(coords, dcoords):
            self.points[u] = upcast[self.base[0]].eval(
                "{} + offset * {}".format(u, du)
            ).values + np.sum(
                [
                    upcast[other].eval("offset * {}".format(du)).values
                    for other in self.base[1:]
                ],
                axis=0,
            )

        in_out = np.zeros(self.points.shape[0], dtype=bool)
        for other in self.base[1:]:
            xx = upcast[self.base[0]]["offset"].values
            yy = upcast[other]["offset"].values
            in_out += (xx + yy) < 1.0

        self.points = self.points[in_out]
        return self.points[coords]


def _local_grid(df, *cols):

    grid = np.meshgrid(*(df[col] for col in cols))
    out = pd.DataFrame.from_dict({col: mm.ravel() for col, mm in zip(cols, grid)})
    return out
