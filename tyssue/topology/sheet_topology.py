import numpy as np
import logging

logger = logging.getLogger(name=__name__)


def type1_transition(sheet, edge01, epsilon=0.1):
    """Performs a type 1 transition around the edge edge01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    """
    # Grab the neighbours
    vert0, vert1, cell_b = sheet.edge_df.loc[
        edge01, ['srce', 'trgt', 'face']].astype(int)
    if sheet.face_df.loc[cell_b, 'num_sides'] < 4:
        logger.warning('''Face %s has 3 sides,
        type 1 transition is not allowed''' % cell_b)
        return

    edge10_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                            (sheet.edge_df['trgt'] == vert0)]
    if len(edge10_.index) < 1:
        raise ValueError('opposite edge to {} with '
                         'source {} and target {} not found'.format(
                             edge01, vert0, vert1))
    edge10 = edge10_.index[0]

    cell_d = int(edge10_.loc[edge10, 'face'])
    if sheet.face_df.loc[cell_d, 'num_sides'] < 4:
        logger.warning('''Face %s has 3 sides,
        type 1 transition is not allowed''' % cell_b)
        return

    edge05_ = sheet.edge_df[(sheet.edge_df['srce'] == vert0) &
                            (sheet.edge_df['face'] == cell_d)]
    edge05 = edge05_.index[0]
    vert5 = int(edge05_.loc[edge05, 'trgt'])

    edge50_ = sheet.edge_df[(sheet.edge_df['srce'] == vert5) &
                            (sheet.edge_df['trgt'] == vert0)]
    edge50 = edge50_.index[0]
    cell_a = int(edge50_.loc[edge50, 'face'])

    edge13_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                            (sheet.edge_df['face'] == cell_b)]
    edge13 = edge13_.index[0]
    vert3 = int(edge13_.loc[edge13, 'trgt'])

    edge31_ = sheet.edge_df[(sheet.edge_df['srce'] == vert3) &
                            (sheet.edge_df['trgt'] == vert1)]
    edge31 = edge31_.index[0]
    cell_c = int(edge31_.loc[edge31, 'face'])

    edge13_ = sheet.edge_df[(sheet.edge_df['srce'] == vert1) &
                            (sheet.edge_df['face'] == cell_b)]
    edge13 = edge13_.index[0]
    vert3 = int(edge13_.loc[edge13, 'trgt'])

    # Perform the rearangements

    sheet.edge_df.loc[edge01, 'face'] = int(cell_c)
    sheet.edge_df.loc[edge10, 'face'] = int(cell_a)
    sheet.edge_df.loc[edge13, ['srce', 'trgt', 'face']] = vert0, vert3, cell_b
    sheet.edge_df.loc[edge31, ['srce', 'trgt', 'face']] = vert3, vert0, cell_c

    sheet.edge_df.loc[edge50, ['srce', 'trgt', 'face']] = vert5, vert1, cell_a
    sheet.edge_df.loc[edge05, ['srce', 'trgt', 'face']] = vert1, vert5, cell_d

    # Displace the vertices
    mean_pos = (sheet.vert_df.loc[vert0, sheet.coords] +
                sheet.vert_df.loc[vert1, sheet.coords]) / 2
    cell_b_pos = sheet.face_df.loc[cell_b, sheet.coords]
    sheet.vert_df.loc[vert0, sheet.coords] = (mean_pos -
                                              (mean_pos - cell_b_pos) *
                                              epsilon)
    cell_d_pos = sheet.face_df.loc[cell_d, sheet.coords]
    sheet.vert_df.loc[vert1, sheet.coords] = (mean_pos -
                                              (mean_pos - cell_d_pos) *
                                              epsilon)
    sheet.reset_topo()
    # Type 1 transitions might create 3 sided cells, we remove those
    for tri_face in sheet.face_df[sheet.face_df['num_sides'] == 3].index:
        remove_face(sheet, tri_face)


def add_vert(sheet, edge):
    """
    Adds a vertex in the middle of the edge,
    which is split as is its opposite
    """

    srce, trgt = sheet.edge_df.loc[edge, ['srce', 'trgt']]
    opposite = sheet.edge_df[(sheet.edge_df['srce'] == trgt) &
                             (sheet.edge_df['trgt'] == srce)]
    if len(opposite):
        opp_edge = opposite.index[0]
    else:
        opp_edge = None

    new_vert = sheet.vert_df.loc[[srce, trgt]].mean()
    sheet.vert_df = sheet.vert_df.append(new_vert, ignore_index=True)
    new_vert = sheet.vert_df.index[-1]
    sheet.edge_df.loc[edge, 'trgt'] = new_vert

    edge_cols = sheet.edge_df.loc[edge]
    sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
    new_edge = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge, 'srce'] = new_vert
    sheet.edge_df.loc[new_edge, 'trgt'] = trgt

    if opp_edge is not None:
        sheet.edge_df.loc[opp_edge, 'srce'] = new_vert
        edge_cols = sheet.edge_df.loc[opp_edge]
        sheet.edge_df = sheet.edge_df.append(edge_cols, ignore_index=True)
        new_opp_edge = sheet.edge_df.index[-1]
        sheet.edge_df.loc[new_opp_edge, 'trgt'] = new_vert
        sheet.edge_df.loc[new_opp_edge, 'srce'] = trgt
    else:
        new_opp_edge = None

    return new_vert, new_edge, new_opp_edge


def cell_division(sheet, mother, geom,
                  angle=None, axis='x'):

    if not sheet.face_df.loc[mother, 'is_alive']:
        logger.warning('Cell {} is not alive and cannot devide'.format(mother))
        return
    edge_a, edge_b = get_division_edges(sheet, mother, geom,
                                        angle=None, axis='x')
    if edge_a is None:
        return
    daughter = face_division(sheet, edge_a, edge_b)
    return daughter


def get_division_edges(sheet, mother, geom,
                       angle=None, axis='x'):

    if angle is None:
        angle = np.random.random() * np.pi

    m_data = sheet.edge_df[sheet.edge_df['face'] == mother]
    if angle == 0:
        face_pos = sheet.face_df.loc[mother, sheet.coords]
        rot_pos = sheet.vert_df[sheet.coords].copy()
        for c in sheet.coords:
            rot_pos.loc[:, c] = rot_pos[c] - face_pos[c]
    else:
        rot_pos = geom.face_projected_pos(sheet, mother, psi=angle)

    srce_pos = rot_pos.loc[m_data['srce'], axis]
    srce_pos.index = m_data.index
    trgt_pos = rot_pos.loc[m_data['trgt'], axis]
    trgt_pos.index = m_data.index
    try:
        edge_a = m_data[(srce_pos < 0) & (trgt_pos >= 0)].index[0]
        edge_b = m_data[(srce_pos >= 0) & (trgt_pos < 0)].index[0]
    except IndexError:
        print('Failed')
        logger.error('Division of Cell {} failed'.format(mother))
        return None, None
    return edge_a, edge_b


def face_division(sheet, edge_a, edge_b):
    """
    Divides the face associated with edges
    indexed by `edge_a` and `edge_b`, splitting it
    in the middle of those edes.
    """
    mother = sheet.edge_df.loc[edge_a, 'face']
    vert_a, new_edge_a, new_opp_edge_a = add_vert(sheet,
                                                  edge_a)
    vert_b, new_edge_b, new_opp_edge_b = add_vert(sheet,
                                                  edge_b)
    sheet.vert_df.index.name = 'vert'

    face_cols = sheet.face_df.loc[mother]
    sheet.face_df = sheet.face_df.append(face_cols,
                                         ignore_index=True)
    sheet.face_df.index.name = 'face'
    daughter = int(sheet.face_df.index[-1])

    edge_cols = sheet.edge_df.loc[new_edge_b]
    sheet.edge_df = sheet.edge_df.append(edge_cols,
                                         ignore_index=True)
    new_edge_m = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_m, 'srce'] = vert_b
    sheet.edge_df.loc[new_edge_m, 'trgt'] = vert_a

    sheet.edge_df = sheet.edge_df.append(edge_cols,
                                         ignore_index=True)
    new_edge_d = sheet.edge_df.index[-1]
    sheet.edge_df.loc[new_edge_d, 'srce'] = vert_a
    sheet.edge_df.loc[new_edge_d, 'trgt'] = vert_b

    # ## Discover daughter edges
    m_data = sheet.edge_df[sheet.edge_df['face'] == mother]
    daughter_edges = [new_edge_d]
    vert_a, vert_b
    srce, trgt = vert_a, vert_b
    srces, trgts = m_data[['srce', 'trgt']].values.T

    while trgt != vert_a:
        srce, trgt = trgt, trgts[srces == trgt][0]
        daughter_edges.append(m_data[(m_data['srce'] == srce) &
                                     (m_data['trgt'] == trgt)].index[0])

    # daughter_edges = list(m_data[srce_pos < 0].index) + [new_edge_b,
    #                                                      new_edge_d]
    sheet.edge_df.loc[daughter_edges, 'face'] = daughter
    sheet.edge_df.index.name = 'edge'
    sheet.reset_topo()
    return daughter


def remove_face(sheet, face):

    if np.isnan(sheet.face_df.loc[face, 'num_sides']):
        logger.info('Face %i is not valid, aborting')
        return

    edges = sheet.edge_df[sheet.edge_df['face'] == face]
    verts = edges['srce'].values

    out_orbits = sheet.get_orbits('srce', 'trgt')
    in_orbits = sheet.get_orbits('trgt', 'srce')

    new_vert_data = sheet.vert_df.loc[verts].mean()
    sheet.vert_df = sheet.vert_df.append(new_vert_data,
                                         ignore_index=True)
    new_vert = sheet.vert_df.index[-1]
    for v in verts:
        out_jes = out_orbits.loc[v].index
        sheet.edge_df.loc[out_jes, 'srce'] = new_vert

        in_jes = in_orbits.loc[v].index
        sheet.edge_df.loc[in_jes, 'trgt'] = new_vert

    sheet.edge_df = sheet.edge_df[sheet.edge_df['srce'] !=
                                  sheet.edge_df['trgt']]

    sheet.edge_df = sheet.edge_df[sheet.edge_df['face'] != face].copy()
    # fidx = sheet.face_df.index.delete(face)
    sheet.face_df.loc[face] = np.nan
    sheet.face_df.loc[face, 'is_alive'] = 0

    logger.info('removed {} of {} vertices '
                .format(len(verts), sheet.vert_df.shape[0]))
    logger.info('face {} is now dead '
                .format(face))

    vidx = sheet.vert_df.index.delete(verts)
    sheet.vert_df = sheet.vert_df.loc[vidx].copy()
    sheet.reset_index()
    sheet.reset_topo()
    return new_vert
