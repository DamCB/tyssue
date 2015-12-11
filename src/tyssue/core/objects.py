import numpy as np
import pandas as pd

# from .. import libtyssue_core as libcore
from . import generation
from .generation import make_df
from ..utils.utils import set_data_columns

import logging
log = logging.getLogger(name=__name__)

'''

The following data is an exemple of the `data_dicts`.
It is a nested dictionnary with two levels.

The first key design objects names: ('cell', 'je', 'jv') They will
correspond to the dataframes attributes of the Epithelium instance,
(e.g eptm.cell_df);

The second level keys design column names of the
above dataframes, and their default values as a (value, dtype) pair.


    data_dicts = {
        'cell': {
            ## Cell Geometry
            'perimeter': (0., np.float),
            'area': (0., np.float),
            ## Coordinates
            'x': (0., np.float),
            'y': (0., np.float),
            'z': (0., np.float),
            ## Topology
            'num_sides': (1, np.int),
            ## Masks
            'is_alive': (1, np.bool)},
        'jv': {
            ## Coordinates
            'x': (0., np.float),
            'y': (0., np.float),
            'z': (0., np.float),
            ## Masks
            'is_active': (1, np.bool)},
        'je': {
            ## Coordinates
            'dx': (0., np.float),
            'dy': (0., np.float),
            'dz': (0., np.float),
            'length': (0., np.float),
            ### Normals
            'nx': (0., np.float),
            'ny': (0., np.float),
            'nz': (0., np.float)}
        }
'''



class Epithelium:
    '''
    The whole tissue.

    '''
    coords = ['x', 'y', 'z']

    def __init__(self, identifier, datasets, coords=None):
        '''
        Creates an epithelium

        Parameters:
        -----------
        identifier: string
        datasets: dictionary of dataframes
        the datasets dict specifies the names, data columns
        and value types of the modeled tyssue

        '''
        if coords is not None:
            self.coords = coords
        # edge's dx, dy, dz
        self.dcoords = ['d'+c for c in self.coords]
        self.dim = len(self.coords)
        # edge's normals
        if self.dim == 3:
            self.ncoords = ['n'+c for c in self.coords]
        self.je_df, self.cell_df, self.jv_df = None, None, None
        self.identifier = identifier
        if not set(('cell', 'jv', 'je')).issubset(datasets):
            raise ValueError('''The `datasets` dictionnary should
            contain at least the 'cell', 'jv' and 'je' keys''')
        for name, data in datasets.items():
            setattr(self, '{}_df'.format(name), data)
        self.data_names = list(datasets.keys())
        self.je_mindex = pd.MultiIndex.from_arrays(self.je_idx.values.T,
                                                   names=['srce', 'trgt', 'cell'])
        self.update_num_sides()

    @classmethod
    def from_points(cls, identifier, points,
                    indices_dict,
                    points_dataset='jv',
                    data_dicts=None):
        '''
        TODO: not sure this works as expected with the new indexing
        '''

        if points.shape[1] == 2:
            coords = cls.coords[:2]
        elif points.shape[1] == 3:
            coords = cls.coords
        else:
            raise ValueError('the `points` argument must be'
                             ' a (Nv, 2) or (Nv, 3) array')

        # data_dicts = update_default(generation.data_dicts, data_dicts)
        datasets = {}
        for key, data_dict in data_dicts.items():
            datasets[key] = make_df(index=indices_dict[key],
                                    data_dict=data_dict)
        datasets[points_dataset][coords] = points

        return cls.__init__(identifier, datasets, coords)

    def set_geom(self, geom, **geom_specs):

        specs = geom.get_default_geom_specs()
        specs.update(**geom_specs)
        set_data_columns(self, specs)
        return specs

    def set_model(self, model, **mod_specs):

        specs = model.get_default_mod_specs()
        specs.update(**mod_specs)
        dim_specs = model.dimentionalize(specs)
        set_data_columns(self, dim_specs)
        return specs, dim_specs

    def update_num_sides(self):
        self.cell_df['num_sides'] = self.je_df.cell.value_counts().loc[
            self.cell_df.index]

    def update_mindex(self):
        self.je_mindex = pd.MultiIndex.from_arrays(self.je_idx.values.T,
                                                   names=['srce', 'trgt', 'cell'])
    def reset_topo(self):
        self.update_num_sides()
        self.update_mindex()

    @property
    def cell_idx(self):
        return self.cell_df.index

    @property
    def jv_idx(self):
        return self.jv_df.index

    @property
    def je_idx(self):
        return self.je_df[['srce', 'trgt', 'cell']]

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
        return self.je_df['srce']

    @property
    def e_trgt_idx(self):
        return self.je_df['trgt']

    @property
    def e_cell_idx(self):
        return self.je_df['cell']

    @property
    def je_idx_array(self):
        return np.vstack((self.e_srce_idx,
                          self.e_trgt_idx,
                          self.e_cell_idx)).T

    def _upcast(self, idx, df):

        upcast = df.loc[idx]
        upcast.index = self.je_df.index
        return upcast

    def upcast_srce(self, df):
        ''' Reindexes input data to self.je_idx
        by repeating the values for each source entry
        '''
        return self._upcast(self.e_srce_idx, df)

    def upcast_trgt(self, df):
        return self._upcast(self.e_trgt_idx, df)

    def upcast_cell(self, df):
        return self._upcast(self.e_cell_idx, df)

    def _lvl_sum(self, df, lvl):
        df_ = df.copy()
        df_.index = self.je_mindex
        return df_.sum(level=lvl)

    def sum_srce(self, df):
        return self._lvl_sum(df, 'srce')

    def sum_trgt(self, df):
        return self._lvl_sum(df, 'trgt')

    def sum_cell(self, df):
        return self._lvl_sum(df, 'cell')

    def get_orbits(self, center, periph):
        orbits = self.je_df.groupby(center).apply(
            lambda df: df[periph])
        return orbits

    def cell_polygons(self, coords):
        def _get_jvs_pos(cell):
            # TODO: return jes as a 2d array
            jes = _ordered_jes(cell)
            return np.array([self.jv_df.loc[idx[0], coords]
                             for idx in jes])
        polys = self.je_df.groupby('cell').apply(_get_jvs_pos)
        return polys

    def _build_cell_cell_indexes(self):
        '''
        This is hackish and not optimized,
        should be provided by CGAL
        '''
        cc_idx = []
        for srce0, trgt0, cell0 in self.je_idx:
            for srce1, trgt1, cell1 in self.je_idx:
                if (cell0 != cell1 and trgt0 == srce1 and
                    trgt1 == srce0 and not (cell1, cell0) in cc_idx):
                    cc_idx.append((cell0, cell1))
        cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['cella', 'cellb'])
        return cc_idx

    def get_valid(self):
        """Set true if the cell is a closed polygon
        """
        is_valid = self.je_df.groupby('cell').apply(_test_valid)
        self.je_df['is_valid'] = self.upcast_cell(is_valid)

    def get_invalid(self):
        """Set true if the cell is a closed polygon
        """
        is_invalid = self.je_df.groupby('cell').apply(_test_invalid)
        return self.upcast_cell(is_invalid)

    def sanitize(self):
        """Removes invalid cells and associated vertices
        """
        invalid_jes = self.get_invalid()
        self.remove(invalid_jes)

    def remove(self, je_out):
        """Removes edges and associated cells where je_out is True
        """
        to_remove = self.je_df[je_out].index
        self.je_df = self.je_df.drop(to_remove)
        all_jvs = np.unique(self.je_df[['srce', 'trgt']])
        self.jv_df = self.jv_df.loc[all_jvs]
        cell_idxs = self.je_df['cell'].unique()
        self.cell_df = self.cell_df.loc[cell_idxs]
        self.reset_topo()

    def cut_out(self, low_x, high_x, low_y, high_y):
        """Removes cells with vertices outside the
        region defined by low_x, low_y, high_x, high_y_
        TODO: find a way to generalize  this function
        to higher dims..
        """

        jv_out = ((self.jv_df['y'] < low_y) |
                  (self.jv_df['y'] > high_y) |
                  (self.jv_df['x'] < low_x) |
                  (self.jv_df['x'] > high_x))

        srce_out = self.upcast_srce(jv_out)
        trgt_out = self.upcast_trgt(jv_out)
        je_out = trgt_out | srce_out
        self.remove(je_out)

    def set_bbox(self, margin=1.):
        '''Sets the attribute `bbox` with pairs of values bellow
        and above the min and max of the jv coords, with a margin.
        '''
        self.bbox = np.array([[self.jv_df[c].min() - margin,
                               self.jv_df[c].max() + margin]
                              for c in self.coords])

def _ordered_jes(cell):
    """Returns the junction edges vertices of the cells
    organized clockwise
    """
    srces, trgts, cells = cell[['srce', 'trgt', 'cell']].values.T
    srce, trgt = srces[0], trgts[0]
    jes = [[srce, trgt, cell]]
    for cell in cells[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        jes.append([srce, trgt, cell])
    return jes

def _test_invalid(cell):
    """ Returns true iff the source and target sets of the cells polygon
    are different
    """
    s1 = set(cell['srce'])
    s2 = set(cell['trgt'])
    return s1 != s2


def _test_valid(cell):
    """ Returns true iff all sources are also targets for the cells polygon
    """
    s1 = set(cell['srce'])
    s2 = set(cell['trgt'])
    return s1 == s2


class Cell:
    '''
    Doesn't hold any data, just methods.

    I think it should be instanciated on demand, not systematically
    for the whole epithelium

    '''
    def __init__(self, eptm, index):

        self.__eptm = eptm
        self.__index = index

    # This should be implemented in CGAL
    def je_orbit(self):
        '''
        Indexes of the cell's junction halfedges.

        '''
        sub_idx = self.__eptm.je_idx[
            self.__eptm.je_idx['cell'] == self.__index].index
        return sub_idx

    def jv_orbit(self):
        '''
        Index of the cell's junction vertices.

        '''
        je_orbit = self.je_orbit()
        return self.__eptm.je_df.loc[je_orbit, 'srce']

    @property
    def num_sides(self):
        return len(self.je_orbit())


class JunctionVertex:

    def __init__(self, eptm, index):

        self.__index = index  # from CGAL
        self.__eptm = eptm  # from CGAL

    def je_orbit(self):
        '''
        Indexes of the neighboring junction edges, returned
        as the indexes of the **outgoing** halfedges.
        '''

        sub_idx = self.__eptm.je_idx[
            self.__eptm.je_idx['srce'] == self.__index].index
        return sub_idx

    def cell_orbit(self):
        '''
        Index of the junction's cells.

        '''
        je_orbit = self.je_orbit()
        return self.__eptm.je_df.loc[je_orbit, 'cell']

    def jv_orbit(self):
        '''
        Index of the junction's neighbor junction vertices.

        '''
        je_orbit = self.je_orbit()
        return self.__eptm.je_df.loc[je_orbit, 'trgt']


class JunctionEdge():
    '''
    Really a HalfEdge ...
    '''

    def __init__(self, eptm, index):

        self.__index = index  # from CGAL
        self.__eptm = eptm  # from CGAL

    @property
    def source_idx(self):
        return self.__eptm.je_df.loc[self.__index, 'srce']

    @property
    def target_idx(self):
        return self.__eptm.je_df.loc[self.__index, 'trgt']

    @property
    def cell_idx(self):
        return self.__eptm.je_df.loc[self.__index, 'cell']
