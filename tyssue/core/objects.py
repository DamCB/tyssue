'''
Core definitions



The following data is an exemple of the `specs`.
It is a nested dictionnary with two levels.

The first key designs the element name: ('face', 'edge', 'vert') They will
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
        'vert': {
            ## Coordinates
            'x': 0.,
            'y': 0.,
            'z': 0.,
            ## Masks
            'is_active': True},
        'edge': {
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


import warnings
import logging
import numpy as np
import pandas as pd
from collections import deque

from ..utils.utils import set_data_columns, spec_updater
log = logging.getLogger(name=__name__)


class Epithelium:
    '''
    The whole tissue.

    '''

    def __init__(self, identifier, datasets,
                 specs=None, coords=None):
        '''
        Creates an epithelium

        Parameters
        ----------
        identifier : string
        datasets : dictionary of dataframes
          the datasets dict specifies the names, data columns
          and value types of the modeled tyssue

        Note
        ----
        For efficiency reasons, we have to maintain monotonous RangeIndex
        for each dataset. Thus, _the index of an element can change_,
        and should not be used as an identifier.

        '''
        # backup container
        # TODO: pass the max backup number as a config argument
        self._backups = deque(maxlen=5)

        # each of those has a separate dataframe, as well as entries in
        # the specification files
        _frame_types = {'edge', 'vert', 'face',
                       'cell'}
        self.identifier = identifier
        if not set(datasets).issubset(_frame_types):
            raise ValueError('The `datasets` dictionnary should'
                             ' contain keys in {}'.format(_frame_types))
        self.datasets = datasets
        self.data_names = list(datasets.keys())
        self.element_names = ['srce', 'trgt',
                              'face', 'cell'][:len(self.data_names)]
        if coords is None:
            coords = [c for c in 'xyz' if c in datasets['vert'].columns]

        self.coords = coords
        # edge's dx, dy, dz
        self.dcoords = ['d'+c for c in self.coords]
        # edge's unit length vector
        self.ucoords = ['u'+c for c in self.coords]

        self.dim = len(self.coords)
        # edge's normals
        if self.dim == 3:
            self.ncoords = ['n'+c for c in self.coords]

        if specs is None:
            specs = {name: {} for name in self.data_names}
        if 'settings' not in specs:
            specs['settings'] = {}

        self.specs = specs
        self.update_specs(specs, reset=False)
        self.edge_mindex = pd.MultiIndex.from_arrays(self.edge_idx.values.T,
                                                     names=self.element_names)
        self.bbox = None
        self.set_bbox()

    @property
    def face_df(self):
        return self.datasets['face']

    @face_df.setter
    def face_df(self, value):
        self.datasets['face'] = value

    @property
    def edge_df(self):
        return self.datasets['edge']

    @edge_df.setter
    def edge_df(self, value):
        self.datasets['edge'] = value

    @property
    def cell_df(self):
        return self.datasets.get('cell', None)

    @cell_df.setter
    def cell_df(self, value):
        self.datasets['cell'] = value

    @property
    def vert_df(self):
        return self.datasets['vert']

    @vert_df.setter
    def vert_df(self, value):
        self.datasets['vert'] = value

    def copy(self, deep_copy=True):
        """
        Returns a copy of the epithelium

        Parameters
        ----------
        deep_copy: bool, default True
            if True, use a copy of the original object's datasets
            to create the new object. If False, datasets are not copied
        """
        if deep_copy:
            datasets = {element: df.copy()
                        for element, df in self.datasets.items()}
        else: #pragma: no cover
            log.info(
                "New epithelium object from {}"
                " without deep copy".format(
                    self.identifier))
            datasets = self.datasets

        identifier = self.identifier+'_copy'
        new = type(self)(identifier, datasets,
                         specs=self.specs, coords=self.coords)
        return new

    def backup(self):
        """Creates a copy of self and keeps a reference to it
        in the self._backups deque.

        """
        log.info('Backing up')
        self._backups.append(self.copy(deep_copy=True))

    def restore(self):
        '''Resets the eptithelium data to its last backed up state

        A copy of the current state prior to restoring is kept in the
        `_bad` attribute for inspection.
        '''

        log.info('Restoring')
        log.info('a copy of the unrestored epithelium'
                 ' is stored in the `_bad` attribute')
        bck = self._backups.pop()
        self._bad = self.copy(deep_copy=True)
        self.datasets = bck.datasets
        self.specs = bck.specs

    @property
    def settings(self):
        return self.specs['settings']

    def update_specs(self, new, reset=False):

        spec_updater(self.specs, new)
        set_data_columns(self.datasets, new, reset)

    def update_num_sides(self):
        self.face_df['num_sides'] = self.edge_df.face.value_counts()

    def update_num_faces(self):
        self.cell_df['num_faces'] = self.edge_df.groupby('cell').apply(
            lambda df: df['face'].unique().size)

    def update_mindex(self):
        self.edge_mindex = pd.MultiIndex.from_arrays(self.edge_idx.values.T,
                                                     names=self.element_names)

    def reset_topo(self):
        self.update_num_sides()
        self.update_mindex()
        if 'cell' in self.data_names:
            self.update_num_faces()

    @property
    def face_idx(self):
        return self.face_df.index

    @property
    def cell_idx(self):
        return self.cell_df.index

    @property
    def vert_idx(self):
        return self.vert_df.index

    @property
    def edge_idx(self):
        # Should it return self.edge_df.index instead ?
        return self.edge_df[self.element_names]

    @property
    def Nc(self):
        if 'cell' in self.data_names:
            return self.cell_df.shape[0]
        elif 'face' in self.data_names:
            return self.face_df.shape[0]
        return None

    @property
    def Nv(self):
        return self.vert_df.shape[0]

    @property
    def Nf(self):
        return self.face_df.shape[0]

    @property
    def Ne(self):
        return self.edge_df.shape[0]

    @property
    def e_srce_idx(self):
        return self.edge_df['srce']

    @property
    def e_trgt_idx(self):
        return self.edge_df['trgt']

    @property
    def e_face_idx(self):
        return self.edge_df['face']

    @property
    def e_cell_idx(self):
        return self.edge_df['cell']

    @property
    def edge_idx_array(self):
        return np.vstack((self.e_srce_idx,
                          self.e_trgt_idx,
                          self.e_face_idx)).T

    def _upcast(self, idx, df):
        ## Assumes a flat index
        upcast = df.take(idx)
        upcast.index = self.edge_df.index
        return upcast

    def upcast_cols(self, element, columns):
        """Syntactic sugar to upcast from the
        epithelium datasets.

        Parameters
        ----------
        element: {'srce'|'trgt'|'face'|'cell'}
           corresponding self.edge_df column over which to index
           if element is 'srce' or 'trgt', the upcast data will be
           taken form self.vert_df
        columns: index
           the column(s) to be taken from the input dataset.

        """
        if element in ['srce', 'trgt']:
            dataset = 'vert'
        else:
            dataset = element
        return self._upcast(self.edge_df[element],
                            self.datasets[dataset][columns])

    def upcast_srce(self, df):
        ''' Reindexes input data to self.edge_idx
        by repeating the values for each source entry
        '''
        return self._upcast(self.edge_df['srce'], df)

    def upcast_trgt(self, df):
        ''' Reindexes input data to self.edge_idx
        by repeating the values for each target entry
        '''
        return self._upcast(self.edge_df['trgt'], df)

    def upcast_face(self, df):
        ''' Reindexes input data to self.edge_idx
        by repeating the values for each face entry
        '''
        return self._upcast(self.edge_df['face'], df)

    def upcast_cell(self, df):
        ''' Reindexes input data to self.edge_idx
        by repeating the values for each cell entry
        '''
        return self._upcast(self.edge_df['cell'], df)

    def _lvl_sum(self, df, lvl):
        df_ = df
        if isinstance(df, pd.Series):
            df_ = df.to_frame()
        elif lvl not in df.columns:
            df_ = df.copy()
        df_[lvl] = self.edge_df[lvl]
        return df_.groupby(lvl).sum()

    def sum_srce(self, df):
        return self._lvl_sum(df, 'srce')

    def sum_trgt(self, df):
        return self._lvl_sum(df, 'trgt')

    def sum_face(self, df):
        return self._lvl_sum(df, 'face')

    def sum_cell(self, df):
        return self._lvl_sum(df, 'cell')

    def get_orbits(self, center, periph):
        """Returns a dataframe with a `(center, edge)` MultiIndex with `periph`
        elements.

        Parmeters
        ---------
        center: str,
            the name of the center element for example 'face', 'srce'
        periph: str,
            the name of the periphery elements, for example 'trgt', 'cell'

        Example
        -------
        >>> cell_verts = sheet.get_orbits('face', 'srce')
        >>> cell_verts.loc[45]
        edge
        218    75
        219    78
        220    76
        221    81
        222    90
        223    87
        Name: srce, dtype: int64

        """
        orbits = self.edge_df.groupby(center).apply(
            lambda df: df[periph])
        return orbits

    def idx_lookup(self, elem_id, element):
        """returns the current index of the element
        with the 'id' column equal to elem_id
        """
        df = self.datasets[element]['id']
        idx = df[df==elem_id].index
        if len(idx):
            return idx[0]
        else:
            return None

    def face_polygons(self, coords):
        def _get_verts_pos(face):
            try:
                edges = _ordered_edges(face)
            except IndexError:
                #- BC -#
                # I'm still trying to figure
                # out a way to raise this exception
                # with altered datasets but to no avail
                # Leaving it included in coverage.
                log.warning('Face is not closed')
                log.warning(face)
                return np.nan
            return np.array([self.vert_df.loc[idx[0], coords]
                             for idx in edges])
        polys = self.edge_df.groupby('face').apply(_get_verts_pos).dropna()
        return polys

    def validate(self):
        """returns True if the mesh is validated

        e.g. has only closed polygons and polyhedra
        """
        return np.alltrue(~self.get_invalid())

    def get_valid(self):
        """Set the 'is_valid' column to true if the faces are all closed polygons,
        and the cells closed polyhedra.
        """
        is_valid_face = self.edge_df.groupby('face').apply(_test_valid)
        is_valid = self.upcast_face(is_valid_face)
        if 'cell' in self.data_names:
            is_valid_cell = self.edge_df.groupby('cell').apply(
                _is_closed_cell)
            is_valid = is_valid | self.upcast_cell(is_valid_cell)
        self.edge_df['is_valid'] = is_valid
        return is_valid

    def get_invalid(self):
        """Returns a mask over self.edge_df for invalid faces
        """
        is_invalid_face = self.edge_df.groupby('face').apply(_test_invalid)
        invalid_edges = self.upcast_face(is_invalid_face)
        if 'cell' in self.data_names:
            is_invalid_cell = 1 - self.edge_df.groupby('cell').apply(
                _is_closed_cell)
            invalid_edges = invalid_edges | self.upcast_cell(is_invalid_cell)
        self.edge_df['is_valid'] = ~invalid_edges
        return invalid_edges

    def sanitize(self):
        """Removes invalid faces and associated vertices
        """
        invalid_edges = self.get_invalid()
        self.remove(invalid_edges)

    def remove(self, edge_out):
        """Remove the edges indexed by `edge_out` associated with all
        the cells and faces containing those edges
        """
        top_level = self.element_names[-1]
        log.info('Removing cells at the %s level', top_level)
        fto_rm = self.edge_df.loc[edge_out, top_level].unique()
        if not fto_rm.shape[0]:
            log.info('Nothing to remove')
            return
        if fto_rm.shape[0] == self.datasets[top_level].shape[0]:
            raise ValueError('sanitize would delete the whole epithlium')

        fto_rm.sort()
        log.info('{} {} level elements will be removed'.format(len(fto_rm),
                                                               top_level))

        edge_df_ = self.edge_df.set_index(
            top_level,
            append=True).swaplevel(0, 1).sort_index()
        to_rm = np.concatenate([edge_df_.loc[c].index.values
                                for c in fto_rm])
        to_rm.sort()
        self.edge_df = self.edge_df.drop(to_rm)

        remaining_verts = np.unique(self.edge_df[['srce', 'trgt']])
        self.vert_df = self.vert_df.loc[remaining_verts]
        if top_level == 'face':
            self.face_df = self.face_df.drop(fto_rm)
        elif top_level == 'cell':
            remaining_faces = np.unique(self.edge_df['face'])
            self.face_df = self.face_df.loc[remaining_faces]
            self.cell_df = self.cell_df.drop(fto_rm)
        self.reset_index()
        self.reset_topo()

    def cut_out(self, bbox, coords=None):
        """Returns the index of edges with
        at least one vertex outside of the
        bounding box

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
        outs = pd.DataFrame(index=self.edge_df.index,
                            columns=coords)
        for c, bounds in zip(coords, bbox):
            out_vert_ = ((self.vert_df[c] < bounds[0]) |
                         (self.vert_df[c] > bounds[1]))
            outs[c] = (self.upcast_srce(out_vert_) |
                       self.upcast_trgt(out_vert_))

        edge_out = outs.sum(axis=1).astype(np.bool)
        return self.edge_df[edge_out].index

    def set_bbox(self, margin=1.):
        '''Sets the attribute `bbox` with pairs of values bellow
        and above the min and max of the vert coords, with a margin.
        '''
        self.bbox = np.array([[self.vert_df[c].min() - margin,
                               self.vert_df[c].max() + margin]
                              for c in self.coords])

    def reset_index(self):

        new_vertidx = pd.Series(np.arange(self.vert_df.shape[0]),
                                index=self.vert_df.index)
        # Here we use loc and not the take from upcast

        self.edge_df['srce'] = new_vertidx.loc[self.edge_df['srce']].values
        self.edge_df['trgt'] = new_vertidx.loc[self.edge_df['trgt']].values

        new_fidx = pd.Series(np.arange(self.face_df.shape[0]),
                             index=self.face_df.index)

        self.edge_df['face'] = new_fidx.loc[self.edge_df['face']].values

        self.vert_df.reset_index(drop=True, inplace=True)
        self.vert_df.index.name = 'vert'

        self.face_df.reset_index(drop=True, inplace=True)
        self.face_df.index.name = 'face'

        if 'cell' in self.data_names:
            new_cidx = pd.Series(np.arange(self.cell_df.shape[0]),
                                 index=self.cell_df.index)
            self.edge_df['cell'] = new_cidx.loc[self.edge_df['cell']].values
            self.cell_df.reset_index(drop=True, inplace=True)
            self.cell_df.index.name = 'cell'

        self.edge_df.reset_index(drop=True, inplace=True)
        self.edge_df.index.name = 'edge'

    def triangular_mesh(self, coords):
        '''
        Return a triangulation of an epithelial sheet (2D in a 3D space),
        with added edges between face barycenters and junction vertices.

        Parameters
        ----------
        coords: list of str:
          pair of coordinates corresponding to column names
          for self.face_df and self.vert_df

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
                                   self.vert_df[coords]), axis=0)

        # edge indices as (Nf + Nv) * 3 array
        triangles = self.edge_df[['srce', 'trgt', 'face']].values
        # The src, trgt, face triangle is correctly oriented
        # both vert_idx cols are shifted by Nf
        triangles[:, :2] += self.Nf

        face_mask = np.arange(self.Nf + self.Nv) < self.Nf
        return vertices, triangles, face_mask

    def vertex_mesh(self, coords, vertex_normals=True):
        ''' Returns the vertex coordinates and a list of vertex indices
        for each face of the tissue.
        If `vertex_normals` is True, also returns the normals of each vertex
        (set as the average of the vertex' edges), suitable for .OBJ export
        '''
        # - BC -#
        # This method only works on 3D-epithelium
        vertices = self.vert_df[coords]
        faces = self.edge_df.groupby('face').apply(ordered_vert_idxs)
        faces = faces.dropna()
        if vertex_normals:
            normals = (self.edge_df.groupby('srce')[self.ncoords].mean() +
                       self.edge_df.groupby('trgt')[self.ncoords].mean()) / 2.
            return vertices.values, faces.values, normals.values
        return vertices.values, faces.values

    def validate_closed_cells(self):
        is_closed = self.edge_df.groupby('cell').apply(_is_closed_cell)
        return is_closed


def _ordered_edges(face_edges):
    """Returns "srce", "trgt" and "face" indices
    organized clockwise for each edge.

    Parameters
    ----------
    face_edges: `pd.DataFrame`
        exerpt of an edge_df for a single face

    Returns
    -------
    edges: list of 3 ints
        srce, trgt, face indices, ordered
    """
    srces, trgts, faces = face_edges[['srce', 'trgt', 'face']].values.T
    srce, trgt, face_ = srces[0], trgts[0], faces[0]
    edges = [[srce, trgt, face_]]
    for face_ in faces[1:]:
        srce, trgt = trgt, trgts[srces == trgt][0]
        edges.append([srce, trgt, face_])
    return edges


def ordered_vert_idxs(face):
    try:
        return [idxs[0] for idxs in _ordered_edges(face)]
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


def get_opposite_faces(eptm):
    """Populates the 'opposite' column of eptm.face_df with the index of
    the opposite face or -1 if the face has no opposite.

    """
    face_v = eptm.edge_df.groupby('face').apply(lambda df: frozenset(df['srce']))
    face_v2 = pd.Series(data=face_v.index, index=face_v.values)
    grouped = face_v2.groupby(level=0)
    cardinal = grouped.apply(len)
    if cardinal.max() > 2:
        raise ValueError('Invalid topology,'
                         ' incorrect faces: {}'.format(cardinal.argmax()))
    eptm.face_df['opposite'] = -1

    face_pairs = []
    grouped.apply(lambda s: face_pairs.append(list(s.values))
                  if len(s) == 2 else np.nan).dropna()
    face_pairs = np.array(face_pairs)
    eptm.face_df.loc[face_pairs[:, 0], 'opposite'] = face_pairs[:, 1]
    eptm.face_df.loc[face_pairs[:, 1], 'opposite'] = face_pairs[:, 0]


def _next_edge(edf):

    edf['edge'] = edf.index
    next_edge = edf.set_index(
        'srce', append=False).loc[
            edf['trgt'], 'edge'].values
    return pd.Series(index=edf.index, data=next_edge)

def _prev_edge(edf):

    edf['edge'] = edf.index
    next_edge = edf.set_index(
        'trgt', append=False).loc[
            edf['srce'], 'edge'].values
    return pd.Series(index=edf.index, data=next_edge)


def get_next_edges(sheet):
    '''
    returns a pd.Series with the index of the next
    edge for each edge
    '''
    next_e = sheet.edge_df.groupby('face').apply(_next_edge)
    next_e.index = next_e.index.droplevel('face')
    return next_e.sort_index()


def get_prev_edges(sheet):
    '''
    returns a pd.Series with the index of the next
    edge for each edge
    '''
    prev_e = sheet.edge_df.groupby('face').apply(_prev_edge)
    prev_e.index = prev_e.index.droplevel('face')
    return prev_e.sort_index()


def _is_closed_cell(e_df):
    edges = e_df[['srce', 'trgt']]
    for e, (srce, trgt) in edges.iterrows():
        if (edges[(edges['srce'] == trgt) &
                  (edges['trgt'] == srce)].index.size != 1):
            return False
    return True
