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

    def __init__(self, identifier, datasets,
                 datadicts=None, coords=None):
        '''
        Creates an epithelium sheet, such as the apical junction network.

        Parameters
        ----------
        identifier: `str`, the tissue name
        face_df: `pandas.DataFrame` indexed by the faces indexes
            this df holds the vertices associated with

        '''
        super().__init__(identifier, datasets,
                         datadicts, coords)
