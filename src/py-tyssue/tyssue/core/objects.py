import numpy as np
import pandas as pd


from  .. import libtyssue_core as libcore
from . import generation
from .generation import make_df

# from ..dl_import import dl_import

import logging
log = logging.getLogger(name=__name__)

def test_import():
    planet = libcore.World()
    planet.set('howdy')
    return planet.greet()


class Epithelium:
    '''
    The whole tissue.

    '''

    def __init__(self, identifier, cell_df, jv_df, je_df):
        '''
        Creates an epithelium

        '''

        self.identifier = identifier
        self.cell_df = cell_df
        self.jv_df = jv_df
        self.je_df = je_df

        self.cc_idx = self._build_cell_cell_indexes()


    @classmethod
    def from_points(cls, identifier, points,
                    cell_idx, jv_idx, je_idx,
                    cell_data=None, jv_data=None, je_data=None):
        '''

        '''

        if points.shape[1] == 2:
            coords = ['x', 'y']
        elif points.shape[1] == 3:
            coords = ['x', 'y', 'z']
        else:
            raise ValueError('the `points` argument must be'
                             ' a (Nv, 2) or (Nv, 3) array')
        if cell_data is None:
            cell_data = generation.cell_data
            jv_data = generation.jv_data
            je_data = generation.je_data
        else:
            cell_data = generation.cell_data.update(cell_data)
            jv_data = generation.jv_data.update(jv_data)
            je_data = generation.je_data.update(je_data)

        ### Cells DataFrame
        self.cell_df = make_df(index=cell_idx, data_dict=cell_data)

        ### Junction vertices and edges DataFrames
        self.jv_df = make_df(index=jv_idx, data_dict=jv_data)
        self.je_df = make_df(index=je_idx, data_dict=je_data)

        self.jv_df[coords] = points

        return cls.__init__(identifier, cell_df, jv_df, je_df)


    @classmethod
    def from_file(cls, input_file, identifier):
        '''
        Creates an `Epithelium` instance from parsing an input file

        '''
        with open(input_file, 'r') as source:
            input_data = parse(source)
            return cls.__init__(identifier, *input_data)


    @property
    def cell_idx(self):
        return self.cell_df.index

    @property
    def jv_idx(self):
        return self.jv_df.index

    @property
    def je_idx(self):
        return self.je_df.index

    @property
    def Nc(self):
        return self.cell_df.shape[0]

    @property
    def Nv(self):
        return self.jv_df.shape[0]

    @property
    def Nf(self):
        return self.je_df.shape[0]

    @property
    def e_srce_idx(self):
        return self.je_idx.get_level_values('srce')

    @property
    def e_trgt_idx(self):
        return self.je_idx.get_level_values('trgt')

    @property
    def e_cell_idx(self):
        return self.je_idx.get_level_values('cell')

    @property
    def je_idx_array(self):
        return np.vstack((self.e_srce_idx,
                          self.e_trgt_idx,
                          self.e_cell_idx)).T

    def triangular_mesh(self, coords):
        '''
        Return a triangulation of an epithelial sheet (2D in a 3D space),
        with added edges between cell barycenters and junction vertices.

        Parameters
        ----------
        coords: list of str:
          pair of coordinates corresponding to column names
          for eptm.cell_df and eptm.jv_df

        Returns
        -------
        vertices: (eptm.Nc+eptm.Nv, 3) ndarray
           all the vertices' coordinates
        faces: (eptm.Nf, 3) ndarray of ints
           triple of the vertices' indexes forming
           the triangular faces. For each junction edge, this is simply
           the index (srce, trgt, cell). This is correctly oriented.
        cell_mask: (eptm.Nc + eptm.Nv,) mask with 1 iff the vertex corresponds
           to a cell center
        '''

        vertices = np.concatenate((eptm.cell_df[coords],
                                   eptm.jv_df[coords]), axis=0)

        ## edge indices as (Nc + Nv) * 3 array
        faces = np.asarray(eptm.je_idx.labels).T
        ## The src, trgt, cell triangle is correctly oriented
        ## both jv_idx cols are shifted by Nc
        faces[:, :2] += eptm.Nc

        cell_mask = np.arange(eptm.Nc + eptm.Nv) < eptm.Nc
        return vertices, faces, cell_mask


    def _build_cell_cell_indexes(self):
        '''
        This is hackish and not optimized,
        should be provided by CGAL
        '''
        cc_idx = []
        for srce0, trgt0, cell0 in self.je_idx:
            for srce1, trgt1, cell1 in self.je_idx:
                if (cell0 != cell1
                    and trgt0 == srce1
                    and trgt1 == srce0
                    and not (cell1, cell0) in cc_idx):
                    cc_idx.append((cell0, cell1))
        cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['cella', 'cellb'])
        return cc_idx


class Cell:
    '''
    Doesn't hold any data, just methods.

    I think it should be instanciated on demand, not systematically
    for the whole epithelium

    '''
    def __init__(self, eptm, index):

        self.__eptm = eptm
        self.__index = index

    ### This should be implemented in CGAL
    def je_orbit(self):
        '''
        Indexes of the cell's junction halfedges.

        '''
        mask, sub_idx = self.__eptm.je_idx.get_loc_level(self.__index,
                                                         level='cell',
                                                         drop_level=False)
        return sub_idx

    def jv_orbit(self):
        '''
        Index of the cell's junction vertices.

        '''
        je_orbit = self.je_orbit()
        return je_orbit.get_level_values('srce')

    @property
    def num_sides(self):
        return len(self.je_orbit())


class JunctionVertex:

    def __init__(self, eptm, index):

        self.__index = index #from CGAL
        self.__eptm = eptm #from CGAL


    def je_orbit(self):
        '''
        Indexes of the neighboring junction edges, returned
        as the indexes of the **outgoing** halfedges.
        '''

        mask, sub_idx = self.__eptm.je_idx.get_loc_level(self.__index,
                                                         level='srce',
                                                         drop_level=False)
        return sub_idx


    def cell_orbit(self):
        '''
        Index of the junction's cells.

        '''
        je_orbit = self.je_orbit()
        return je_orbit.get_level_values('cell')


    def jv_orbit(self):
        '''
        Index of the junction's neighbor junction vertices.

        '''
        je_orbit = self.je_orbit()
        return je_orbit.get_level_values('trgt')



class JunctionEdge():
    '''
    Really a HalfEdge ...
    '''


    def __init__(self, eptm, index):

        self.__index = index #from CGAL
        self.__eptm = eptm #from CGAL

    @property
    def source_idx(self):
        return self.__index[0]

    @property
    def target_idx(self):
        return self.__index[1]

    @property
    def cell_idx(self):
        return self.__index[2]

    @property
    def oposite_idx(self):
        jei = self.__eptm.je_idx_array
        return tuple(*jei[(jei[:, 0] == 1)*(jei[:, 1] == 0)])
