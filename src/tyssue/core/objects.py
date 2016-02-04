import numpy as np
import pandas as pd

# from .. import libtyssue_core as libcore
# from . import generation
from ..utils.utils import set_data_columns, spec_updater
from ..config.json_parser import load_default
from .generation import make_df

import logging
log = logging.getLogger(name=__name__)

'''

The following data is an exemple of the `specs`.
It is a nested dictionnary with two levels.

The first key design objects names: ('face', 'je', 'jv') They will
correspond to the dataframes attributes of the Epithelium instance,
(e.g eptm.face_df);

The second level keys design column names of the above dataframes,
default values is allways infered from the python parsed type. Thus
`1` will be cast as `int`, `1.` as `float` and `True` as a `bool`.

    specs = {
        'face': {
            ## Face Geometry
            'perimeter': 0.,
            'area': 0.,
            ## Coordinates
            'x': 0.,
            'y': 0.,
            'z': 0.,
            ## Topology
            'num_sides': 6,
            ## Masks
            'is_alive': True},
        'jv': {
            ## Coordinates
            'x': 0.,
            'y': 0.,
            'z': 0.,
            ## Masks
            'is_active': True},
        'je': {
            ## Coordinates
            'dx': 0.,
            'dy': 0.,
            'dz': 0.,
            'length': 0.,
            ### Normals
            'nx': 0.,
            'ny': 0.,
            'nz': 1.}
        }
'''



class Epithelium:
    '''
    The whole tissue.

    '''

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
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

        # each of those has a separate dataframe, as well as entries in
        # the settings files
        frame_types =  {'je', 'jv', 'face',
                        'cell', 'cc'}

        # Really just to ensure the debugger is silent
        [   self.je_df,
            self.jv_df,
            self.face_df,
            self.cell_df,
            self.cc_df    ] = [None,] * 5

        self.identifier = identifier
        if not set(datasets).issubset(frame_types):
            raise ValueError('''The `datasets` dictionnary should
            contain keys in {}'''.format(frame_types))
        for name, data in datasets.items():
            setattr(self, '{}_df'.format(name), data)
        self.data_names = list(datasets.keys())
        self.element_names = ['srce', 'trgt',
                              'face', 'cell'][:len(self.data_names)]
        if specs is None:
            specs = {name:{} for name in self.data_names}
        self.specs = specs
        self.settings = {}
        self.je_mindex = pd.MultiIndex.from_arrays(self.je_idx.values.T,
                                                   names=self.element_names)
        ## Topology (geometry independant)
        self.reset_topo()
        self.bbox = None
        self.set_bbox()

    def copy(self):
        # TODO
        raise NotImplementedError



    @classmethod
    def from_points(cls, identifier, points,
                    indices_dict, specs,
                    points_dataset='jv'):
        '''
        TODO: not sure this works as expected with the new indexing
        '''

        if points.shape[1] == 2:
            coords = ['x', 'y']
        elif points.shape[1] == 3:
            coords = ['x', 'y', 'z']
        else:
            raise ValueError('the `points` argument must be'
                             ' a (Nv, 2) or (Nv, 3) array')

        datasets = {}
        for key, spec in specs.items():
            datasets[key] = make_df(index=indices_dict[key],
                                    spec=spec)
        datasets[points_dataset][coords] = points

        return cls.__init__(identifier, datasets, coords)


    def update_specs(self, new, reset=False):

        spec_updater(self.specs, new)
        if 'settings' in new:
            self.settings.update(new['settings'])
        set_data_columns(self.datasets, new, reset)

    def set_specs(self, domain, base,
                  new_specs=None,
                  default_base=None, reset=False):

        if base is None:
            self.update_specs(load_default(domain, default_base), reset)
        else:
            self.update_specs(load_default(domain, base), reset)
        if new_specs is not None:
            self.update_specs(new_specs, reset)

    def set_geom(self, base=None, new_specs=None, ):

        self.set_specs('geometry', base, new_specs,
                       default_base='core', reset=False)

    def set_model(self, base=None, new_specs=None, reset=True):

        self.set_specs('dynamics', base, new_specs,
                       default_base='core', reset=reset)

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
        datasets = {level: getattr(self, '{}_df'.format(level))
                    for level in self.data_names}
        return datasets

    # @datasets.getter
    # def datasets(self, level):
    #     return getattr(self, '{}_df'.format(level))

    @datasets.setter
    def datasets(self, level, new_df):
        setattr(self, '{}_df'.format(level), new_df)

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
        if 'cell' in self.data_names:
            return self.cell_df.shape[0]
        elif 'face' in self.data_names:
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

    def jverts_mesh(self, coords, vertex_normals=True):
        ''' Returns the vertex coordinates and a list of vertex indices
        for each face of the tissue.
        If `vertex_normals` is True, also returns the normals of each vertex
        (set as the average of the vertex' edges), suitable for .OBJ export
        '''
        vertices = self.jv_df[coords]
        faces = self.je_df.groupby('face').apply(ordered_jv_idxs)
        faces = faces.dropna()
        if vertex_normals:
            normals = (self.je_df.groupby('srce')[self.ncoords].mean() +
                       self.je_df.groupby('trgt')[self.ncoords].mean()) / 2.
            return vertices, faces, normals
        return vertices, faces


def _ordered_jes(face):
    """Returns the junction edges vertices of the faces
    organized clockwise
    """
    srces, trgts, faces = face[['srce', 'trgt', 'face']].values.T
    srce, trgt, face_ = srces[0], trgts[0], faces[0]
    jes = [[srce, trgt, face_]]
    for face_ in faces[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        jes.append([srce, trgt, face_])
    return jes


def ordered_jv_idxs(face):
    try:
        return [idxs[0] for idxs in _ordered_jes(face)]
    except IndexError:
        return np.nan


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


class CellCellMesh():
    """
    Class to manipulate cell centric models
    """

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
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
        self.cell_df, self.cc_df = None, None
        self.identifier = identifier
        if not {'cell', 'cc'}.issubset(datasets):
            raise ValueError('''The `datasets` dictionnary should
            contain at least the 'cell' and 'cc' keys ''')
        self.data_names = list(datasets.keys())
        self.element_names = ['srce', 'trgt']
        for name, data in datasets.items():
            setattr(self, '{}_df'.format(name), data)
        if specs is None:
            specs = {name:{} for name in self.data_names}
        self.specs = specs
        self.cc_mindex = pd.MultiIndex.from_arrays(
            self.cc_df[['srce', 'trgt']].values.T,
            names=['srce', 'trgt'])
        self.settings = {}

    def copy(self):
        # TODO
        raise NotImplementedError

    def update_specs(self, new):
        for key, datadict in self.specs.items():
            if new.get(key) is not None:
                datadict.update(new[key])
        if 'settings' in new:
            self.settings.update(new['settings'])

    def set_geom(self, geom, **geom_specs):

        specs = geom.get_default_geom_specs()
        specs.update(**geom_specs)
        self.update_specs(specs)
        return specs

    def set_model(self, model, **mod_specs):

        specs = model.get_default_mod_specs()
        specs.update(**mod_specs)
        dim_specs = model.dimentionalize(specs)
        self.update_specs(dim_specs)
        self.nrj_norm_factor = dim_specs['settings']['nrj_norm_factor']
        return specs, dim_specs


    @property
    def datasets(self):
        datasets = {
            'cc': self.cc_df,
            'cell': self.cell_df,
            }
        return datasets

    @datasets.getter
    def datasets(self, level):
        return getattr(self, '{}_df'.format(level))

    @property
    def cell_idx(self):
        return self.cell_df.index

    @property
    def cc_idx(self):
        return self.cc_df[self.element_names]

    @property
    def e_srce_idx(self):
        return self.cc_df['srce']

    @property
    def e_trgt_idx(self):
        return self.cc_df['trgt']


    @property
    def Nc(self):
        return self.cell_df.shape[0]

    @property
    def Ne(self):
        return self.cc_df.shape[0]

    def _upcast(self, idx, df):

        upcast = df.loc[idx]
        upcast.index = self.cc_df.index
        return upcast

    def upcast_srce(self, df):
        ''' Reindexes input data to self.cc_idx
        by repeating the values for each source entry
        '''
        return self._upcast(self.e_srce_idx, df)

    def upcast_trgt(self, df):
        return self._upcast(self.e_trgt_idx, df)

    def reset_topo(self):
        self.cc_mindex = pd.MultiIndex.from_arrays(self.cc_df['srce', 'trgt'].values,
                                                   names=['srce', 'trgt'])

    def _lvl_sum(self, df, lvl):
        df_ = df.copy()
        df_.index = self.cc_mindex
        return df_.sum(level=lvl)

    def sum_srce(self, df):
        return self._lvl_sum(df, 'srce')

    def sum_trgt(self, df):
        return self._lvl_sum(df, 'trgt')

    def set_bbox(self, margin=1.):
        '''Sets the attribute `bbox` with pairs of values bellow
        and above the min and max of the cell coords, with a margin.
        '''
        self.bbox = np.array([[self.cell_df[c].min() - margin,
                               self.cell_df[c].max() + margin]
                              for c in self.coords])
    def reset_index(self):

        new_cellidx = pd.Series(np.arange(self.cell_df.shape[0]),
                                index=self.cell_df.index)
        self.cc_df['srce'] = self.upcast_srce(new_cellidx)
        self.cc_df['trgt'] = self.upcast_trgt(new_cellidx)
        self.cell_df.reset_index(drop=True, inplace=True)
        self.cell_df.index.name = 'cell'
        self.cc_df.reset_index(drop=True, inplace=True)
        self.cc_df.index.name = 'cc'
