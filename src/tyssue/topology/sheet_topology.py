

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
