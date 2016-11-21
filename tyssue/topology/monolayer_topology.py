import logging
import numpy as np

from ..geometry.bulk_geometry import MonoLayerGeometry
from ..core.sheet import Sheet
from ..geometry.sheet_geometry import SheetGeometry
from .bulk_topology import get_division_vertices
from .bulk_topology import cell_division as bulk_division
from .sheet_topology import type1_transition as sheet_t1
from .sheet_topology import get_division_edges as sheet_division_edges


logger = logging.getLogger(name=__name__)


def cell_division(monolayer, mother,
                  orientation='vertical',
                  psi=0):
    """
    Divides the cell mother in the monolayer.

    Parameters
    ----------
    * monolayer: a :class:`Monolayer` instance
    * mother: int, the index of the cell to devide
    * orientation: str, {"vertical" | "horizontal"}
      if "horizontal", performs a division in the equatorial
      plane of the cell. If "vertical" (the default), performs
      a division along the basal-apical axis of the cell
    * psi: float, default 0
      extra rotation angle of the division plane
      around the basal-apical plane

    Returns
    -------
    * daughter: int, the index of the daughter cell
    """

    ab_axis = MonoLayerGeometry.basal_apical_axis(monolayer, mother)
    plane_normal = np.asarray(ab_axis)

    if orientation == 'horizontal':
        vertices = get_division_vertices(monolayer,
                                         mother=mother,
                                         plane_normal=plane_normal)
    elif orientation == 'vertical':
        apical_sheet = monolayer.get_sub_sheet('apical')
        m_apical_face = monolayer.edge_df[
            (monolayer.edge_df['cell'] == mother) &
            (monolayer.edge_df['segment'] == 'apical')]['face'].iloc[0]
        apical_edges = sheet_division_edges(apical_sheet,
                                            m_apical_face,
                                            SheetGeometry)
        basal_edges = []
        for ae in apical_edges:
            basal_edges.append(find_basal_edge(monolayer, ae))
        division_edges = list(apical_edges) + basal_edges
        vertices = get_division_vertices(monolayer,
                                         division_edges=division_edges)
    else:
        raise ValueError('''orientation argument not understood,
should be either "horizontal" or "vertical", not {}'''.format(orientation))

    daughter = bulk_division(monolayer, mother,
                             MonoLayerGeometry, vertices)

    # Correct segment assignations for the septum
    septum = monolayer.face_df.index[-2:]
    septum_edges = monolayer.edge_df.index[-2*len(vertices):]
    if orientation == 'vertical':
        monolayer.face_df.loc[septum, 'segment'] = 'sagittal'
        monolayer.edge_df.loc[septum_edges, 'segment'] = 'sagittal'
        _assign_vert_segment(monolayer, mother, vertices)

    elif orientation == 'horizontal':
        monolayer.face_df.loc[septum[0], 'segment'] = 'apical'
        monolayer.face_df.loc[septum[1], 'segment'] = 'basal'
        monolayer.edge_df.loc[septum_edges[:len(vertices)],
                              'segment'] = 'apical'
        monolayer.edge_df.loc[septum_edges[len(vertices):],
                              'segment'] = 'basal'
        monolayer.vert_df.loc[vertices, 'segment'] = 'apical'

    return daughter


def _assign_vert_segment(monolayer, cell, vertices):

    for v in vertices:
        segs = set(monolayer.edge_df[
            monolayer.edge_df['srce'] == v]['segment'])
        if 'apical' in segs:
            monolayer.vert_df.loc[v, 'segment'] = 'apical'
        elif 'basal' in segs:
            monolayer.vert_df.loc[v, 'segment'] = 'basal'
        else:
            monolayer.vert_df.loc[v, 'segment'] = 'sagittal'


def find_basal_edge(monolayer, apical_edge):
    """Returns the basal edge parallel to the apical edge passed
    in argument.

    Parameters
    ----------
    monolayer: a :class:`Monolayer` instance

    """
    srce, trgt, cell = monolayer.edge_df.loc[apical_edge,
                                             ['srce', 'trgt', 'cell']]
    cell_edges = monolayer.edge_df[monolayer.edge_df['cell'] == cell]
    srce_segment = monolayer.vert_df.loc[cell_edges['srce'].values,
                                         'segment']
    srce_segment.index = cell_edges.index
    trgt_segment = monolayer.vert_df.loc[cell_edges['trgt'].values,
                                         'segment']
    trgt_segment.index = cell_edges.index
    b_trgt, = cell_edges[(srce_segment == 'apical') &
                         (trgt_segment == 'basal') &
                         (cell_edges['srce'] == srce)]['trgt']
    b_srce, = cell_edges[(srce_segment == 'basal') &
                         (trgt_segment == 'apical') &
                         (cell_edges['trgt'] == trgt)]['srce']
    b_edge, = cell_edges[(cell_edges['srce'] == b_srce) &
                         (cell_edges['trgt'] == b_trgt)].index
    return b_edge


def type1_transition(monolayer, apical_edge, epsilon=0.1):
    """Performs a type 1 transition on the apical and basal meshes
    """
    v0_a, v1_a, fb_a, cb_a = monolayer.edge_df.loc[
        apical_edge, ['srce', 'trgt', 'face', 'cell']]
    basal_edge = find_basal_edge(monolayer, apical_edge)
    v0_b, v1_b, fb_b, cb_b = monolayer.edge_df.loc[
        basal_edge, ['srce', 'trgt', 'face', 'cell']]
    if monolayer.face_df.loc[fb_a, 'num_sides'] < 4:
        logger.warning('''Face %s has 3 sides,
type 1 transition is not allowed''' % fb_a)
        return


def layer_t1_transition(monolayer, edge01, epsilon=0.1):

    vert0, vert1, face_ba, cell_b = monolayer.edge_df.loc[
        edge01, ['srce', 'trgt', 'face', 'cell']].astype(int)
    segment = monolayer.edge_df.loc[edge01, 'segment']
    if monolayer.face_df.loc[face_ba, 'num_sides'] < 4:
        logger.warning('''Face %s has 3 sides,
type 1 transition is not allowed''' % face_ba)
        return
    edges01_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert0) &
                                 (monolayer.edge_df['trgt'] == vert1)]
    edges10_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert1) &
                                 (monolayer.edge_df['trgt'] == vert0)]
    if not len(edges10_.index):
        raise ValueError('opposite edge to {} with '
                         'source {} and target {} not found'.format(
                             edge01, vert0, vert1))
    edges10 = edges10_.index
    edges01 = edges01_.index

    face_da = edges10_[edges10_['segment'] == segment]['face']
    face_ds = edges10_[edges10_['segment'] != segment]['face']
    face_bs = edges01_[edges01_['segment'] != segment]['face']

    if monolayer.face_df.loc[face_da, 'num_sides'] < 4:
        logger.warning('''Face %s has 3 sides,
        type 1 transition is not allowed''' % face_da)
        return

    vert5 = monolayer.edge_df[(monolayer.edge_df['srce'] == vert0) &
                              (monolayer.edge_df['face'] == face_da)]['trgt']
    edges05_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert0) &
                                 (monolayer.edge_df['trgt'] == vert5)]
    edges05 = edges05_.index

    edges50_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert5) &
                                 (monolayer.edge_df['trgt'] == vert0)]
    edges50 = edges50_.index
    face_aa = edges50_[edges50_['segment'] == segment]['face']
    face_as = edges50_[edges50_['segment'] != segment]['face']

    vert3 = monolayer.edge_df[(monolayer.edge_df['srce'] == vert1) &
                              (monolayer.edge_df['face'] == face_ba)]['trgt']

    edges31_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert3) &
                                 (monolayer.edge_df['trgt'] == vert1)]
    edges31 = edges31_.index
    faces_c = edges31_['face']

    edges13_ = monolayer.edge_df[(monolayer.edge_df['srce'] == vert1) &
                                 (monolayer.edge_df['trgt'] == vert3)]
    edges13 = edges13_.index

    # rearangements
    monolayer.edge_df.loc[edge01, 'face'] = int(face_c)
    monolayer.edge_df.loc[edge10, 'face'] = int(face_a)
    monolayer.edge_df.loc[edges13, ['srce', 'trgt', 'face']] = vert0, vert3
    monolayer.edge_df.loc[edges31, ['srce', 'trgt', 'face']] = vert3, vert0

    monolayer.edge_df.loc[edges50, ['srce', 'trgt', 'face']] = vert5, vert1
    monolayer.edge_df.loc[edges05, ['srce', 'trgt', 'face']] = vert1, vert5

    # Displace the vertices
    mean_pos = (monolayer.vert_df.loc[vert0, monolayer.coords] +
                monolayer.vert_df.loc[vert1, monolayer.coords]) / 2
    face_b_pos = monolayer.face_df.loc[face_b, monolayer.coords]
    monolayer.vert_df.loc[vert0, monolayer.coords] = (mean_pos -
                                                      (mean_pos - face_b_pos) *
                                                      epsilon)
    face_d_pos = monolayer.face_df.loc[face_d, monolayer.coords]
    monolayer.vert_df.loc[vert1, monolayer.coords] = (mean_pos -
                                                      (mean_pos - face_d_pos) *
                                                      epsilon)
    monolayer.reset_topo()
