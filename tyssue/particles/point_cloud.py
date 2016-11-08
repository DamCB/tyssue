"""Utilities to generate point clouds

The positions of the points are generated along the architecture
of the epithelium.
"""

import numpy as np
import pandas as pd
import collections

from ..utils.utils import spec_updater
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
        density: dictionnary,
          the keywords arguments of density_func, defaults to {}


        Attributes
        ----------

        upcaster: np.ndarray, shape (Np,)
          edge indices repeated to match the lookup table
        density_lookup: np.ndarray, shape (Np,)
          piecewise continuous density lookup

        """

        self.edge_df = edge_df.copy()
        self.n_edges = self.edge_df.shape[0]
        self.specs = bulk_spec()
        self.specs.update(**kwargs)

        if not 'density' in edge_df:
            self.edge_df['density'] = self.specs['density']
        self.specs.update(**kwargs)
        self.n_points = 0
        self.points = None
        self.density_lut = None
        self.update_all()

    def update_all(self):
        self.update_density_lut()
        self.update_particles()
        self.update_upcaster()
        self.update_offset()

    def update_density_lut(self, density_lut=None):
        """
        Updates the density lookup table function.

        The `density_lut` can be any function
        with a single `num` argument


        Parameters
        ----------
        density_lut: function, default None,
          edge-wise function of the number of points
        """
        if density_lut is not None:
            self.density_lut = density_lut
        if np.isclose(self.specs["gamma"], 1.0):
            self.density_lut = lambda num: np.arange(0., num)/num
        else:
            gamma = self.specs["gamma"]
            self.density_lut = lambda num: (np.arange(0., num)/num)**gamma

    def update_particles(self):
        """
        * Updates the number of particles per edge from edges length
        and density values:
        `num_particles = length * density`
        * Also updates the self.points df
        """
        points_per_edges = self.edge_df.eval('length * density').astype(np.int)
        self.edge_df['num_particles'] = points_per_edges
        self.n_points = points_per_edges.sum()
        self.points = pd.DataFrame(np.zeros((self.n_points, 2)),
                                            columns=['upcaster',
                                                     'offset'])
    def update_upcaster(self):
        """
        resets the 'upcaster' column of self.points,

        'upcaster' indexes over self.edge_df repeated to
        upcast data from the edge df to the points df
        """
        self.points['upcaster'] = np.repeat(
            np.arange(self.edge_df.shape[0]),
            self.edge_df['num_particles'])

    def update_offset(self):
        self.points['offset'] = np.concatenate(
            [self.density_lut(num=ns)
             for ns in self.edge_df['num_particles']])

    def validate(self):

        if not self.points['upcaster'].max() + 1 == self.n_edges:
            return False
        if not self.points['upcaster'].shape[0] == self.n_points:
            return False
        if not self.points['offset'].shape()[0] == self.n_points:
            return False
        return True

    def edge_point_cloud(self, srce_pos, r_ij,
                         coords=['x', 'y', 'z'],
                         dcoords=['dx', 'dy', 'dz']):
        """Generates a point cloud along the edges of the epithelium.

        If eptm.edge_df has a `"density"` column, it is used
        to modulate the number of points per edges on a edge basis.

        Returns
        -------
        points: (Np, 3) pd.DataFrame with the points positions
        upcaster: indexer of shape Np with the repeated
          edge index for each point
        """
        for u, du in zip(coords, dcoords):
            self.edge_df[u] = srce_pos[u]
            self.edge_df['d'+u] = r_ij[du]
        cols = coords + dcoords
        upcast = self.edge_df.loc[self.points['upcaster'],
                                  cols]
        upcast['offset'] = self.points['offset'].values
        for c in coords:
            self.points[c] = upcast.eval(
                '{} + offset * {}'.format(c, 'd'+c)).values
        if self.specs['noise'] > 0.0:
            self.points[coords] += np.random.normal(scale=self.specs['noise'],
                                                    size=(self.n_points, 3))