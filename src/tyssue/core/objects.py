import libtyssue_core as core


def test_import():
    planet = core.World()
    planet.set('howdy')
    return planet.greet()


class Epithelium(core.Epithelium):

    def __init__(self, eptm=None):
        if eptm is None:
            self.__eptm = core.Epithelium()
        else:
            self.__eptm = eptm.__eptm


class LinearCellComplex:
    '''
    Just a stand up for the actual CGAL class
    '''
    def __init__(self, dim, space_dim):
        '''
        Parameters
        ----------

        dim: int
          The dimension of the LCC (0 for a vertex, 1 for an edge, and so on)
        space_dim: int
          The surrounding space dimension (2 or 3, usually)

        '''
        self.i = dim  # as in i-cell in the doc.


class Vertex(LinearCellComplex):

    def __init__(self):
        LinearCellComplex.__init__(self, 0)


class Edge(LinearCellComplex):

    def __init__(self):
        LinearCellComplex.__init__(self, 1)


class Face(LinearCellComplex):

    def __init__(self):
        LinearCellComplex.__init__(self, 2)


class Volume(LinearCellComplex):

    def __init__(self):
        LinearCellComplex.__init__(self, 3)


class Cell(LinearCellComplex):

    def __init__(self, dim):
        LinearCellComplex.__init__(self, dim)

    @property
    def j_edges(self):
        '''
        Iterate over the junction edges
        '''
        for je in self._jnct_edges:
            yield je

    @property
    def faces(self):
        '''
        Iterate over the faces
        '''
        for face in self._faces:
            yield face
