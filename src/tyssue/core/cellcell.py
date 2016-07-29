import pandas as pd
from scipy.spatial import ConvexHull
from .objects import Epithelium


class CellCellMesh(Epithelium):
    """
    Class to manipulate cell centric models
    """

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
        '''

        Parameters:
        -----------
        identifier: string
        datasets: dictionary of dataframes
        the datasets dict specifies the names, data columns
        and value types of the modeled tyssue

        '''
        super().__init__(identifier, datasets,
                         specs, coords)

    def vertex_mesh(self, coords, vertex_normals=False):
        '''
        Subclassed version to acccount for the CellCellMesh specificity

        For now uses the Convex Hull as triangulation. a Better solution should
        be implemented.
        '''

        vertices = self.vert_df[coords].values
        qhull = ConvexHull(vertices)
        faces = qhull.simplices
        return vertices, faces, None
