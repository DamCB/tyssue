import numpy as np
import logging

logger = logging.getLogger(name=__name__)


def type1_transition(sheet, je01, epsilon=0.1):
    """Performs a type 1 transition around the edge je01

    See ../../doc/illus/t1_transition.png for a sketch of the definition
    of the vertices and cells letterings
    """
    # Grab the neighbours
    jv0, jv1, cell_b = sheet.je_df.loc[
        je01, ['srce', 'trgt', 'face']].astype(int)

    je10_ = sheet.je_df[(sheet.je_df['srce'] == jv1) &
                        (sheet.je_df['trgt'] == jv0)]
    je10 = je10_.index[0]
    cell_d = int(je10_.loc[je10, 'face'])

    je05_ = sheet.je_df[(sheet.je_df['srce'] == jv0) &
                        (sheet.je_df['face'] == cell_d)]
    je05 = je05_.index[0]
    jv5 = int(je05_.loc[je05, 'trgt'])

    je50_ = sheet.je_df[(sheet.je_df['srce'] == jv5) &
                        (sheet.je_df['trgt'] == jv0)]
    je50 = je50_.index[0]
    cell_a = int(je50_.loc[je50, 'face'])

    je13_ = sheet.je_df[(sheet.je_df['srce'] == jv1) &
                        (sheet.je_df['face'] == cell_b)]
    je13 = je13_.index[0]
    jv3 = int(je13_.loc[je13, 'trgt'])

    je31_ = sheet.je_df[(sheet.je_df['srce'] == jv3) &
                        (sheet.je_df['trgt'] == jv1)]
    je31 = je31_.index[0]
    cell_c = int(je31_.loc[je31, 'face'])

    je13_ = sheet.je_df[(sheet.je_df['srce'] == jv1) &
                        (sheet.je_df['face'] == cell_b)]
    je13 = je13_.index[0]
    jv3 = int(je13_.loc[je13, 'trgt'])

    # Perform the rearangements

    sheet.je_df.loc[je01, 'face'] = int(cell_c)
    sheet.je_df.loc[je10, 'face'] = int(cell_a)
    sheet.je_df.loc[je13, ['srce', 'trgt', 'face']] = jv0, jv3, cell_b
    sheet.je_df.loc[je31, ['srce', 'trgt', 'face']] = jv3, jv0, cell_c

    sheet.je_df.loc[je50, ['srce', 'trgt', 'face']] = jv5, jv1, cell_a
    sheet.je_df.loc[je05, ['srce', 'trgt', 'face']] = jv1, jv5, cell_d

    # Displace the vertices
    mean_pos = (sheet.jv_df.loc[jv0, sheet.coords] +
                sheet.jv_df.loc[jv1, sheet.coords]) / 2
    cell_b_pos = sheet.face_df.loc[cell_b, sheet.coords]
    sheet.jv_df.loc[jv0, sheet.coords] = (mean_pos -
                                          (mean_pos - cell_b_pos) * epsilon)
    cell_d_pos = sheet.face_df.loc[cell_d, sheet.coords]
    sheet.jv_df.loc[jv1, sheet.coords] = (mean_pos -
                                          (mean_pos - cell_d_pos) * epsilon)
    sheet.reset_topo()


def add_jv(sheet, je):

    srce, trgt = sheet.je_df.loc[je, ['srce', 'trgt']]
    opposite = sheet.je_df[(sheet.je_df['srce'] == trgt)
                           & (sheet.je_df['trgt'] == srce)]
    opp_je = opposite.index[0]

    new_jv = sheet.jv_df.loc[[srce, trgt]].mean()
    sheet.jv_df = sheet.jv_df.append(new_jv, ignore_index=True)
    new_jv = sheet.jv_df.index[-1]
    sheet.je_df.loc[je, 'trgt'] = new_jv
    sheet.je_df.loc[opp_je, 'srce'] = new_jv

    je_cols = sheet.je_df.loc[je]
    sheet.je_df = sheet.je_df.append(je_cols, ignore_index=True)
    new_je = sheet.je_df.index[-1]
    sheet.je_df.loc[new_je, 'srce'] = new_jv
    sheet.je_df.loc[new_je, 'trgt'] = trgt

    je_cols = sheet.je_df.loc[opp_je]
    sheet.je_df = sheet.je_df.append(je_cols, ignore_index=True)
    new_opp_je = sheet.je_df.index[-1]
    sheet.je_df.loc[new_opp_je, 'trgt'] = new_jv
    sheet.je_df.loc[new_opp_je, 'srce'] = trgt
    return new_jv, new_je, new_opp_je


def cell_division(sheet, mother, geom, angle=None):

    if not sheet.face_df.loc[mother, 'is_alive']:
        logger.warning('Cell {} is not alive and cannot devide'.format(mother))
        return

    if angle is None:
        angle = np.random.random() * np.pi

    m_data = sheet.je_df[sheet.je_df['face'] == mother]
    rot_pos = geom.face_projected_pos(sheet, mother, psi=angle)
    srce_pos = rot_pos.loc[m_data['srce'], 'x']
    srce_pos.index = m_data.index
    trgt_pos = rot_pos.loc[m_data['trgt'], 'x']
    trgt_pos.index = m_data.index

    je_a = m_data[(srce_pos < 0) & (trgt_pos > 0)].index[0]
    je_b = m_data[(srce_pos > 0) & (trgt_pos < 0)].index[0]


    jv_a, new_je_a, new_opp_je_a = add_jv(sheet, je_a)
    jv_b, new_je_b, new_opp_je_b = add_jv(sheet, je_b)

    face_cols = sheet.face_df.loc[mother]
    sheet.face_df = sheet.face_df.append(face_cols,
                                         ignore_index=True)
    daughter = int(sheet.face_df.index[-1])


    je_cols = sheet.je_df.loc[new_je_b]
    sheet.je_df = sheet.je_df.append(je_cols, ignore_index=True)
    new_je_m = sheet.je_df.index[-1]
    sheet.je_df.loc[new_je_m, 'srce'] = jv_b
    sheet.je_df.loc[new_je_m, 'trgt'] = jv_a

    sheet.je_df = sheet.je_df.append(je_cols, ignore_index=True)
    new_je_d = sheet.je_df.index[-1]
    sheet.je_df.loc[new_je_d, 'srce'] = jv_a
    sheet.je_df.loc[new_je_d, 'trgt'] = jv_b

    daughter_jes = list(m_data[srce_pos < 0].index) + [new_je_b, new_je_d]
    sheet.je_df.loc[daughter_jes, 'face'] = daughter
    sheet.reset_topo()
    geom.update_all(sheet)
