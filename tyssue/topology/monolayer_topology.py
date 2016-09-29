from collections import defaultdict
import logging
import numpy as np
import itertools

from .sheet_topology import face_division
from .base_topology import add_vert
from ..geometry.bulk_geometry import BulkGeometry, MonoLayerGeometry

logger = logging.getLogger(name=__name__)


def _get_division_edges(monolayer, mother, orientation, psi=0):

    rotated = MonoLayerGeometry.cell_projected_pos(monolayer,
                                                   mother, psi)
    mother_edges = monolayer.edge_df[monolayer.edge_df['cell'] == mother]
    if orientation == 'vertical':
        srce_x = rotated.loc[mother_edges['srce'], 'x']
        srce_x.index = mother_edges.index
        trgt_x = rotated.loc[mother_edges['trgt'], 'x']
        trgt_x.index = mother_edges.index
        division_edges = mother_edges[((srce_x >= 0) &
                                       (trgt_x <= 0)) |
                                      ((srce_x <= 0) &
                                       (trgt_x >= 0))]
        # remove duplicate sagittal edges
        division_edges = division_edges[
            division_edges['segment'] != 'sagittal']
        seg_count = division_edges['segment'].value_counts()
        assert(seg_count.loc['basal'] == 2)
        assert(seg_count.loc['apical'] == 2)

    elif orientation == 'horizontal':
        srce_x = rotated.loc[mother_edges['srce'], 'z']
        srce_x.index = mother_edges.index
        trgt_x = rotated.loc[mother_edges['trgt'], 'z']
        trgt_x.index = mother_edges.index
        division_edges = mother_edges[((srce_x <= 0) &
                                       (trgt_x >= 0))]
        assert(np.all(division_edges['segment'] == 'sagittal'))
    else:
        raise ValueError('''Orientation not understood, should be
either "horizontal" or "vertical''')

    # Order the returned edges so that their centers
    # are oriented counterclockwize in the division plane
    # in preparation for septum creation
    srce_pos = rotated.loc[division_edges['srce'],
                           monolayer.coords].values
    trgt_pos = rotated.loc[division_edges['trgt'],
                           monolayer.coords].values
    centers = (srce_pos + trgt_pos)/2
    theta = np.arctan2(centers[:, 2], centers[:, 1])
    division_edges = division_edges.iloc[np.argsort(theta)]
    return division_edges


def cell_division(monolayer, mother,
                  orientation='vertical'):
    division_edges = _get_division_edges(monolayer, mother,
                                         orientation)
    vertices = []
    for edge in division_edges.index:
        vert_i, *new_edges = add_vert(monolayer, edge)
        vertices.append(vert_i)




def horizontal_division(monolayer, mother):

    cell_cols = monolayer.cell_df.loc[mother]
    monolayer.cell_df = monolayer.cell_df.append(cell_cols,
                                                 ignore_index=True)
    daughter = monolayer.cell_df.index[-1]
    #  mother cell's edges
    m_data = monolayer.edge_df[monolayer.edge_df['cell'] == mother]

    sagittal_faces = list(m_data[m_data['segment'] ==
                                 'sagittal']['face'].unique())
    face_orbit = monolayer.get_orbits('face', 'srce').groupby(
        level='face').apply(lambda df: set(df))
    # grab opposite faces
    opp_faces = {}
    for face in sagittal_faces:
        pair = face_orbit[face_orbit == face_orbit.loc[face]].index
        opposite = pair[pair != face]
        if len(opposite) == 1:
            opp_faces[face] = opposite[0]
        else:
            opp_faces[face] = -1
    # divide the sagittal faces & their opposites
    new_faces = {}
    for face, opp_face in opp_faces.items():
        if opp_face > 0:
            face_cols = monolayer.face_df.loc[[face, opp_face]]
            monolayer.face_df = monolayer.face_df.append(face_cols,
                                                         ignore_index=True)
            new_faces[face] = monolayer.face_df.index[-2]
            new_faces[opp_face] = monolayer.face_df.index[-1]
        else:
            face_cols = monolayer.face_df.loc[face]
            monolayer.face_df = monolayer.face_df.append(face_cols,
                                                         ignore_index=True)
            new_faces[face] = monolayer.face_df.index[-1]

    # grab the sagittal edges oriented upward
    srce_segment = monolayer.upcast_srce(
        monolayer.vert_df['segment']).loc[m_data.index]
    trgt_segment = monolayer.upcast_trgt(
        monolayer.vert_df['segment']).loc[m_data.index]

    apical_edges = m_data[m_data['segment'] == 'apical'].index
    monolayer.edge_df.loc[apical_edges, 'cell'] = daughter

    sagittal_edges = m_data[(m_data['segment'] == 'sagittal') &
                            (srce_segment == 'basal') &
                            (trgt_segment == 'apical')]
    # split the sagittal edges
    new_verts = {}
    face_verts = defaultdict(set)
    for edge, edge_data in sagittal_edges.iterrows():
        new_vert, old_pll, new_pll, old_opp, new_opp = add_vert(monolayer,
                                                                edge)
        new_verts[edge] = new_vert
        for old, new in zip(old_pll+old_opp, new_pll+new_opp):
            old_face = monolayer.edge_df.loc[old, 'face']
            if old_face not in new_faces:
                continue
            new_face = new_faces[old_face]
            monolayer.edge_df.loc[new, 'face'] = new_face
            monolayer.edge_df.loc[new, 'cell'] = daughter
            face_verts[old_face].add(new_vert)
    for vs in face_verts.values():
        assert len(vs) == 2

    # add the new horizontal faces

    apical_edges = m_data[(m_data['segment'] == 'apical')]
    mother_apical_face = apical_edges['face'].iloc[0]
    apical_face_cols = monolayer.face_df.loc[mother_apical_face]
    monolayer.face_df.append(apical_face_cols, ignore_index=True)
    new_apical = monolayer.face_df.index[-1]

    basal_edges = m_data[(m_data['segment'] == 'basal')]
    mother_basal_face = basal_edges['face'].iloc[0]
    basal_face_cols = monolayer.face_df.loc[mother_basal_face]
    monolayer.face_df.append(basal_face_cols, ignore_index=True)
    new_basal = monolayer.face_df.index[-1]

    # add the horizontal edges
    for edge, edge_data in sagittal_edges.iterrows():
        old_face = edge_data['face']
        new_face = new_faces[old_face]
        vert_a = new_verts[edge]
        vert_b = face_verts[old_face].difference({vert_a}).pop()
        edge_cols = monolayer.edge_df.loc[edge].copy()
        edge_cols['srce'] = vert_a
        edge_cols['trgt'] = vert_b
        edge_cols['face'] = old_face
        edge_cols['cell'] = mother
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=False)
        edge_cols = monolayer.edge_df.loc[edge].copy()
        edge_cols['srce'] = vert_b
        edge_cols['trgt'] = vert_a
        edge_cols['face'] = new_face
        edge_cols['cell'] = daughter
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=False)

        edge_cols = apical_edges.iloc[0].copy()
        edge_cols['srce'] = vert_b
        edge_cols['trgt'] = vert_a
        edge_cols['face'] = new_apical
        edge_cols['cell'] = mother
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=False)

        edge_cols = basal_edges.iloc[0].copy()
        edge_cols['srce'] = vert_a
        edge_cols['trgt'] = vert_b
        edge_cols['face'] = new_basal
        edge_cols['cell'] = daughter
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=False)
    return daughter


def vertical_division(monolayer, mother, *args, **kwargs):

    mother_edges = monolayer.edge_df[
        monolayer.edge_df['cell'] == mother]
    mother_faces = set(mother_edges['face'])

    apical_face, = mother_edges[
        mother_edges['segment'] == 'apical']['face'].unique()
    apical_daughter = face_division(monolayer,
                                    apical_face,
                                    BulkGeometry, *args, **kwargs)
