import logging
import numpy as np

from ..geometry.bulk_geometry import MonoLayerGeometry
from .bulk_topology import get_division_vertices
from .bulk_topology import cell_division as bulk_division
from .sheet_topology import type1_transition as sheet_t1

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
        plane_normal = ab_axis
    elif orientation == 'vertical':
        # put the normal along the x axis
        plane_normal = ab_axis[[0, 2, 1]] * np.array([0, 1, -1])
        if psi != 0:
            cp, sp = np.cos(psi), np.sin(psi)
            rot = np.array([[1,  0,  0],
                            [0,  cp, sp],
                            [0, -sp, cp]])
            plane_normal = np.dot(rot, plane_normal)
    else:
        raise ValueError('''orientation argument not understood,
should be either "horizontal" or "vertical", not {}'''.format(orientation))

    vertices = get_division_vertices(monolayer, mother,
                                     plane_normal)
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
    srce, trgt, face, cell = monolayer.edge_df.loc[['srce', 'trgt',
                                                    'face', 'cell']]
    apical_sheet = monolayer.get_sub_sheet('apical')
    sheet_t1(apical_sheet, apical_edge, epsilon)
    basal_edge = find_basal_edge(monolayer, apical_edge)
    basal_sheet = monolayer.get_sub_sheet('basal')
    sheet_t1(basal_sheet, basal_edge, epsilon)
