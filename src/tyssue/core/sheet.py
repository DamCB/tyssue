'''
An epithelial sheet, i.e a 2D mesh in a 2D or 3D space,
akin to a HalfEdge data structure in CGAL.

For purely 2D the geometric properties are defined in `tyssue.geometry.planar_geometry`
A dynamical model derived from Fahradifar et al. 2007 is provided in
`tyssue.dynamics.planar_vertex_model`


For 2D in 3D, the geometric properties are defined in `tyssue.geometry.sheet_geometry`
A dynamical model derived from Fahradifar et al. 2007 is provided in
`tyssue.dynamics.sheet_vertex_model`


'''



import numpy as np
from .objects import Epithelium

class Sheet(Epithelium):
    '''
    An epithelial sheet, i.e a 2D mesh in a 2D or 3D space,
    akin to a HalfEdge data structure in CGAL.

    The geometric properties are defined in `tyssue.geometry.sheet_geometry`
    A dynamical model derived from Fahradifar et al. 2007 is provided in
    `tyssue.dynamics.sheet_vertex_model`


    '''

    def __init__(self, identifier, datasets):
        '''
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        cell_df: `pandas.DataFrame` indexed by the cells indexes
            this df holds the vertices associated with

        '''
        super().__init__(identifier, datasets)

    def triangular_mesh(self, coords):
        '''
        Return a triangulation of an epithelial sheet (2D in a 3D space),
        with added edges between cell barycenters and junction vertices.

        Parameters
        ----------
        coords: list of str:
          pair of coordinates corresponding to column names
          for self.cell_df and self.jv_df

        Returns
        -------
        vertices: (self.Nc+self.Nv, 3) ndarray
           all the vertices' coordinates
        faces: (self.Nf, 3) ndarray of ints
           triple of the vertices' indexes forming
           the triangular faces. For each junction edge, this is simply
           the index (srce, trgt, cell). This is correctly oriented.
        cell_mask: (self.Nc + self.Nv,) mask with 1 iff the vertex corresponds
           to a cell center
        '''

        vertices = np.concatenate((self.cell_df[coords],
                                   self.jv_df[coords]), axis=0)

        # edge indices as (Nc + Nv) * 3 array
        faces = np.asarray(self.je_idx.labels).T
        # The src, trgt, cell triangle is correctly oriented
        # both jv_idx cols are shifted by Nc
        faces[:, :2] += self.Nc

        cell_mask = np.arange(self.Nc + self.Nv) < self.Nc
        return vertices, faces, cell_mask
