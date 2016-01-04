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

The first key design objects names: ('face', 'je', 'jv') They will
correspond to the dataframes attributes of the Epithelium instance,
(e.g eptm.face_df);

The second level keys design column names of the
above dataframes, and their default values as a (value, dtype) pair.


    data_dicts = {
        'face': {
            ## Face Geometry
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

    def __init__(self, identifier, datasets,
                 datadicts=None, coords=None):
        '''
        Creates an epithelium

        Parameters:
        -----------
        identifier: string
        datasets: dictionary of dataframes
        the datasets dict specifies the names, data columns
        and value types of the modeled tyssue

        '''
        if coords is  None:
            coords = ['x', 'y', 'z']
        self.coords = coords
        # edge's dx, dy, dz
        self.dcoords = ['d'+c for c in self.coords]
        self.dim = len(self.coords)
        # edge's normals
        if self.dim == 3:
            self.ncoords = ['n'+c for c in self.coords]
        #
        self.je_df, self.face_df, self.jv_df, self.cell_df = (None,) * 4
        self.identifier = identifier
        if not set(('face', 'jv', 'je')).issubset(datasets):
            raise ValueError('''The `datasets` dictionnary should
            contain at least the 'face', 'jv' and 'je' keys''')
        for name, data in datasets.items():
            setattr(self, '{}_df'.format(name), data)
        self.data_names = list(datasets.keys())
        self.element_names = ['srce', 'trgt',
                              'face', 'cell'][:len(self.data_names)]
        if datadicts is None:
            datadicts = {name:{} for name in self.data_names}
        self.datadicts = datadicts
        self.je_mindex = pd.MultiIndex.from_arrays(self.je_idx.values.T,
                                                   names=self.element_names)
        ## Topology (geometry independant)
        self.reset_topo()
        self.bbox = None
        self.set_bbox()

    def copy(self):
        raise NotImplementedError

    @classmethod
    def from_points(cls, identifier, points,
                    indices_dict, data_dicts,
                    points_dataset='jv'):
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


    def update_datadicts(self, new):
        for key, datadict in self.datadicts.items():
            if new.get(key) is not None:
                datadict.update(new[key])

    def set_geom(self, geom, **geom_specs):

        specs = geom.get_default_geom_specs()
        specs.update(**geom_specs)
        set_data_columns(self, specs)
        self.update_datadicts(specs)
        return specs

    def set_model(self, model, **mod_specs):

        specs = model.get_default_mod_specs()
        specs.update(**mod_specs)
        dim_specs = model.dimentionalize(specs)
        set_data_columns(self, dim_specs, reset=True)
        self.update_datadicts(dim_specs)
        return specs, dim_specs

    def update_num_sides(self):
        self.face_df['num_sides'] = self.je_df.face.value_counts().loc[
            self.face_df.index]

    def update_num_faces(self):
        self.cell_df['num_faces'] = self.je_df.cell.value_counts().loc[
            self.cell_df.index]


    def update_mindex(self):
        self.je_mindex = pd.MultiIndex.from_arrays(self.je_idx.values.T,
                                                   names=self.element_names)
    def reset_topo(self):
        self.update_num_sides()
        self.update_mindex()
        if 'cell' in self.data_names:
            self.update_num_faces()

    @property
    def datasets(self):
        datasets = {
            'je': self.je_df,
            'jv': self.jv_df,
            'face': self.face_df,
            }
        if 'cell' in self.data_names:
            datasets['cell'] = self.cell_df
        return datasets

    @datasets.getter
    def datasets(self, level):
        return getattr(self, '{}_df'.format(level))

    @property
    def face_idx(self):
        return self.face_df.index

    @property
    def cell_idx(self):
        return self.cell_df.index

    @property
    def jv_idx(self):
        return self.jv_df.index

    @property
    def je_idx(self):
        return self.je_df[self.element_names]

    @property
    def Nc(self):
        if 'cell' in self.element_names:
            return self.cell_df.shape[0]
        elif 'face' in self.element_names:
            return self.face_df.shape[0]

    @property
    def Nv(self):
        return self.jv_df.shape[0]

    @property
    def Nf(self):
        return self.face_df.shape[0]

    @property
    def Ne(self):
        return self.je_df.shape[0]

    @property
    def e_srce_idx(self):
        return self.je_df['srce']

    @property
    def e_trgt_idx(self):
        return self.je_df['trgt']

    @property
    def e_face_idx(self):
        return self.je_df['face']

    @property
    def e_cell_idx(self):
        return self.je_df['cell']

    @property
    def je_idx_array(self):
        return np.vstack((self.e_srce_idx,
                          self.e_trgt_idx,
                          self.e_face_idx)).T

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

    def upcast_face(self, df):
        return self._upcast(self.e_face_idx, df)

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

    def sum_face(self, df):
        return self._lvl_sum(df, 'face')

    def sum_cell(self, df):
        return self._lvl_sum(df, 'cell')

    def get_orbits(self, center, periph):
        orbits = self.je_df.groupby(center).apply(
            lambda df: df[periph])
        return orbits

    def face_polygons(self, coords):
        def _get_jvs_pos(face):
            # TODO: return jes as a 2d array
            try:
                jes = _ordered_jes(face)
            except IndexError:
                log.warning('Face is not closed')
                return np.nan
            return np.array([self.jv_df.loc[idx[0], coords]
                             for idx in jes])
        polys = self.je_df.groupby('face').apply(_get_jvs_pos).dropna()
        return polys

    def _build_face_face_indexes(self):
        '''
        This is hackish and not optimized,
        should be provided by CGAL
        '''
        cc_idx = []
        for srce0, trgt0, face0 in self.je_idx:
            for srce1, trgt1, face1 in self.je_idx:
                if (face0 != face1 and trgt0 == srce1 and
                    trgt1 == srce0 and not (face1, face0) in cc_idx):
                    cc_idx.append((face0, face1))
        cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['facea', 'faceb'])
        return cc_idx

    def get_valid(self):
        """Set true if the face is a closed polygon
        """
        is_valid = self.je_df.groupby('face').apply(_test_valid)
        self.je_df['is_valid'] = self.upcast_face(is_valid)

    def get_invalid(self):
        """Returns a mask over je for invalid faces
        """
        is_invalid = self.je_df.groupby('face').apply(_test_invalid)
        return self.upcast_face(is_invalid)

    def sanitize(self):
        """Removes invalid faces and associated vertices
        """
        invalid_jes = self.get_invalid()
        self.remove(invalid_jes)

    def remove(self, je_out):

        top_level = self.element_names[-1]
        log.info('Removing cells at the {} level'.format(top_level))
        fto_rm = self.je_df[je_out][top_level].unique()
        if not len(fto_rm):
            log.info('Nothing to remove')
            return
        fto_rm.sort()
        log.info('{} {} level elements will be removed'.format(len(fto_rm),
                                                               top_level))

        je_df_ = self.je_df.set_index(top_level,
                                      append=True).swaplevel(0, 1).sort_index()
        to_rm = np.concatenate([je_df_.loc[c].index.values
                                for c in fto_rm])
        to_rm.sort()
        self.je_df = self.je_df.drop(to_rm)

        remaining_jvs = np.unique(self.je_df[['srce', 'trgt']])
        self.jv_df = self.jv_df.loc[remaining_jvs]
        if top_level == 'face':
            self.face_df = self.face_df.drop(fto_rm)
        elif top_level == 'cell':
            remaining_faces = np.unique(self.je_df['face'])
            self.face_df = self.face_df.loc[remaining_faces]
            self.cell_df = self.cell_df.drop(fto_rm)
        self.reset_index()
        self.reset_topo()

    def cut_out(self, bbox, coords=None):
        """Removes faces with vertices outside the
        region defined by the bbox

        Parameters
        ----------
        bbox : sequence of shape (dim, 2)
             the bounding box as (min, max) pairs for
             each coordinates.
        coords : list of str of len dim
             the coords corresponding to the bbox.
        """
        if coords is None:
            coords = self.coords
        outs = pd.DataFrame(index=self.je_df.index,
                            columns=coords)
        for c, bounds in zip(coords, bbox):
            out_jv_ = ((self.jv_df[c] < bounds[0]) |
                       (self.jv_df[c] > bounds[1]))
            outs[c] = (self.upcast_srce(out_jv_) |
                       self.upcast_trgt(out_jv_))

        je_out = outs.sum(axis=1).astype(np.bool)
        return je_out

    def set_bbox(self, margin=1.):
        '''Sets the attribute `bbox` with pairs of values bellow
        and above the min and max of the jv coords, with a margin.
        '''
        self.bbox = np.array([[self.jv_df[c].min() - margin,
                               self.jv_df[c].max() + margin]
                              for c in self.coords])

    def reset_index(self):

        new_jvidx = pd.Series(np.arange(self.jv_df.shape[0]),
                              index=self.jv_df.index)
        self.je_df['srce'] = self.upcast_srce(new_jvidx)
        self.je_df['trgt'] = self.upcast_trgt(new_jvidx)
        new_fidx = pd.Series(np.arange(self.face_df.shape[0]),
                             index=self.face_df.index)
        self.je_df['face'] = self.upcast_face(new_fidx)

        self.jv_df.reset_index(drop=True, inplace=True)
        self.jv_df.index.name = 'jv'

        self.face_df.reset_index(drop=True, inplace=True)
        self.face_df.index.name = 'face'

        if 'cell' in self.data_names:
            new_cidx = pd.Series(np.arange(self.cell_df.shape[0]),
                                 index=self.cell_df.index)
            self.je_df['cell'] = self.upcast_cell(new_cidx)
            self.cell_df.reset_index(drop=True, inplace=True)
            self.cell_df.index.name = 'cell'

        self.je_df.reset_index(drop=True, inplace=True)
        self.je_df.index.name = 'je'


    def triangular_mesh(self, coords):
        '''
        Return a triangulation of an epithelial sheet (2D in a 3D space),
        with added edges between face barycenters and junction vertices.

        Parameters
        ----------
        coords: list of str:
          pair of coordinates corresponding to column names
          for self.face_df and self.jv_df

        Returns
        -------
        vertices: (self.Nf+self.Nv, 3) ndarray
           all the vertices' coordinates
        triangles: (self.Ne, 3) ndarray of ints
           triple of the vertices' indexes forming
           the triangular elements. For each junction edge, this is simply
           the index (srce, trgt, face). This is correctly oriented.
        face_mask: (self.Nf + self.Nv,) mask with 1 iff the vertex corresponds
           to a face center
        '''

        vertices = np.concatenate((self.face_df[coords],
                                   self.jv_df[coords]), axis=0)

        # edge indices as (Nf + Nv) * 3 array
        triangles = self.je_df[['srce', 'trgt', 'face']].values
        # The src, trgt, face triangle is correctly oriented
        # both jv_idx cols are shifted by Nf
        triangles[:, :2] += self.Nf

        face_mask = np.arange(self.Nf + self.Nv) < self.Nf
        return vertices, triangles, face_mask


def _ordered_jes(face):
    """Returns the junction edges vertices of the faces
    organized clockwise
    """
    srces, trgts, faces = face[['srce', 'trgt', 'face']].values.T
    srce, trgt = srces[0], trgts[0]
    jes = [[srce, trgt, face]]
    for face in faces[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        jes.append([srce, trgt, face])

    return jes

def _test_invalid(face):
    """ Returns true iff the source and target sets of the faces polygon
    are different
    """
    s1 = set(face['srce'])
    s2 = set(face['trgt'])
    return s1 != s2


def _test_valid(face):
    """ Returns true iff all sources are also targets for the faces polygon
    """
    s1 = set(face['srce'])
    s2 = set(face['trgt'])
    return s1 == s2



class Face:
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
        Indexes of the face's junction halfedges.

        '''
        sub_idx = self.__eptm.je_idx[
            self.__eptm.je_idx['face'] == self.__index].index
        return sub_idx

    def jv_orbit(self):
        '''
        Index of the face's junction vertices.

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

    def face_orbit(self):
        '''
        Index of the junction's faces.

        '''
        je_orbit = self.je_orbit()
        return self.__eptm.je_df.loc[je_orbit, 'face']

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
    def face_idx(self):
        return self.__eptm.je_df.loc[self.__index, 'face']
