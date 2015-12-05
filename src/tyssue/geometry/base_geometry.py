def scale(sheet, delta, coords):
    ''' Scales the coordinates `coords`
    by a factor `delta`
    '''
    sheet.jv_df[coords] = sheet.jv_df[coords] * delta

def update_dcoords(sheet, coords):
    '''
    Update the edge vector coordinates  on the
    `coords` basis (`default_coords` by default). Modifies the corresponding
    columns (i.e `['dx', 'dy', 'dz']`) in sheet.edge_df.
    '''
    dcoords = ['d'+c for c in coords]
    data = sheet.jv_df[coords]
    srce_pos = sheet.upcast_srce(data).values
    trgt_pos = sheet.upcast_trgt(data).values

    sheet.je_df[dcoords] = (trgt_pos - srce_pos)


def update_length(sheet, coords):
    '''
    Updates the edge_df `length` column on the `coords` basis
    '''
    dcoords = ['d' + c for c in coords]
    sheet.je_df['length'] = np.linalg.norm(sheet.je_df[dcoords],
                                           axis=1)


def update_centroid(sheet, coords):
    '''
    Updates the cell_df `coords` columns as the cell's vertices
    center of mass.
    '''
    upcast_pos = sheet.upcast_srce(sheet.jv_df[coords])
    sheet.cell_df[coords] = upcast_pos.groupby(level='cell').mean()
