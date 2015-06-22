
from ..dl_import import dl_import

libcore = None ## Avoids code check complains od libcore being undefined

dl_import("from .. import libtyssue_core as libcore")
dl_import("from ..libtyssue_core import Point")

#dl_import("from ..libtyssue_core import Epithelium")



def test_import():
    planet = libcore.World()
    planet.set('howdy')
    print(planet.greet())



class Epithelium():

    def __init__(self, eptm=None):
        if eptm is None:
            self._eptm = libcore.Epithelium()
        else:
            self._eptm = eptm._eptm

    def cells(self):
        for cell in self._eptm.iter_cells():
            yield cell

    def junction_vertices(self):
        for vertex in self._eptm.iter_junction_vertex():
            yield vertex

    def junction_edges(self):
        for edge in self._eptm.iter_junction_edges():
            yield edge


class Cell():

    def __init__(self, dim):
        self.dim = dim

    def j_edges(self):
        '''
        Iterate over the junction edges
        '''
        for je in self._jnct_edges:
            yield je

    def j_vertices(self):
        '''
        Iterate over the junction edges
        '''
        for jv in self._jnct_vertices:
            yield jv

    def faces(self):
        '''
        Iterate over the faces
        '''
        for face in self._faces:
            yield face
