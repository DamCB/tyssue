from collections import defaultdict
import logging

logger = logging.getLogger(name=__name__)


def add_vert(monolayer, edge):
    """
    Adds a vertex in the middle of the edge,

    The edge and all its parallel and opposite
    edges, i.e those who share the same vertices
    are split as well.
    """

    srce, trgt = monolayer.edge_df.loc[edge, ['srce', 'trgt']]
    parallels = monolayer.edge_df[(monolayer.edge_df['srce'] == srce) &
                                  (monolayer.edge_df['trgt'] == trgt)].index
    opposites = monolayer.edge_df[(monolayer.edge_df['srce'] == trgt) &
                                  (monolayer.edge_df['trgt'] == srce)].index
    new_vert = monolayer.vert_df.loc[[srce, trgt]].mean()
    monolayer.vert_df = monolayer.vert_df.append(new_vert, ignore_index=True)
    new_vert = monolayer.vert_df.index[-1]

    old_pll = list(parallels)
    new_pll = []
    for d_edge in parallels:  # includes the original edge
        monolayer.edge_df.loc[d_edge, 'trgt'] = new_vert
        edge_cols = monolayer.edge_df.loc[d_edge]
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=True)
        new_edge = monolayer.edge_df.index[-1]
        monolayer.edge_df.loc[new_edge, 'srce'] = new_vert
        monolayer.edge_df.loc[new_edge, 'trgt'] = trgt
        new_pll.append(new_edge)

    old_opp = list(opposites)
    new_opp = []
    for d_edge in opposites:
        monolayer.edge_df.loc[d_edge, 'srce'] = new_vert
        edge_cols = monolayer.edge_df.loc[d_edge]
        monolayer.edge_df = monolayer.edge_df.append(edge_cols,
                                                     ignore_index=True)
        new_edge = monolayer.edge_df.index[-1]
        monolayer.edge_df.loc[new_edge, 'srce'] = trgt
        monolayer.edge_df.loc[new_edge, 'trgt'] = new_vert
        new_opp.append(new_edge)

    return new_vert, old_pll, new_pll, old_opp, new_opp


def cell_division(monolayer, mother,
                  orientation='horizontal'):

    if orientation != 'horizontal':
        raise NotImplementedError('Only horizontal orientation'
                                  ' is supported at the moment')
    cell_cols = monolayer.cell_df.loc[mother]
    monolayer.cell_df = monolayer.cell_df.append(cell_cols,
                                                 ignore_index=True)
    daughter = monolayer.cell_df.index[-1]

    m_data = monolayer.edge_df[monolayer.edge_df['cell'] == mother]

    sagittal_faces = list(m_data[m_data['segment'] ==
                                 'sagittal']['face'].unique())
    face_orbit = monolayer.get_orbits('face', 'srce').groupby(
        level='face').apply(lambda df: set(df))
    opp_faces = {}
    for face in sagittal_faces:
        pair = face_orbit[face_orbit == face_orbit.loc[face]].index
        opposite = pair[pair != face]
        if len(opposite) == 1:
            opp_faces[face] = opposite[0]
        else:
            opp_faces[face] = -1
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

    srce_segment = monolayer.upcast_srce(
        monolayer.vert_df['segment']).loc[m_data.index]
    trgt_segment = monolayer.upcast_trgt(
        monolayer.vert_df['segment']).loc[m_data.index]

    apical_edges = m_data[m_data['segment'] == 'apical'].index
    monolayer.edge_df.loc[apical_edges, 'cell'] = daughter

    sagittal_edges = m_data[(m_data['segment'] == 'sagittal') &
                            (srce_segment == 'basal') &
                            (trgt_segment == 'apical')]
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
