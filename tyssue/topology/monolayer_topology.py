from collections import defaultdict
import logging
import numpy as np
import itertools

from ..geometry.bulk_geometry import MonoLayerGeometry
from .bulk_topology import get_division_vertices
from .bulk_topology import cell_division as bulk_division

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
    """

    ab_axis = MonoLayerGeometry.basal_apical_axis(mother)
    if orientation == 'horizontal':
        plane_normal = ab_axis
    elif orientation == 'vertical':
        # put the normal along the x axis
        plane_normal = ab_axis[[2, 1, 0]]
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

    elif orientation == 'horizontal':
        monolayer.face_df.loc[septum[0], 'segment'] = 'apical'
        monolayer.face_df.loc[septum[1], 'segment'] = 'basal'
        monolayer.edge_df.loc[septum_edges[:len(vertices)],
                              'segment'] = 'apical'
        monolayer.edge_df.loc[septum_edges[len(vertices):],
                              'segment'] = 'basal'

    return daughter


def _assign_vert_segment(monolayer, cell, vertices):

    ab_axis = MonoLayerGeometry.basal_apical_axis(cell)
    cell_edges = monolayer.edge_df[monolayer.edge_df["cell"] == cell]
    vert_pos_ = monolayer.vert_df.loc[set(cell_edges['srce'])]
    vert_pos = vert_pos_.loc[vertices]
    for c in monolayer.coords:
        vert_pos[c] -= monolayer.cell_df.loc[cell, c]
