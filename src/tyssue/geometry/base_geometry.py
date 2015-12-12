import numpy as np

def scale(sheet, delta, coords):
    ''' Scales the coordinates `coords`
    by a factor `delta`
    '''
    sheet.jv_df[coords] = sheet.jv_df[coords] * delta

def update_dcoords(sheet):
    '''
    Update the edge vector coordinates  on the
    `coords` basis (`default_coords` by default). Modifies the corresponding
    columns (i.e `['dx', 'dy', 'dz']`) in sheet.edge_df.
    '''
    data = sheet.jv_df[sheet.coords]
    srce_pos = sheet.upcast_srce(data).values
    trgt_pos = sheet.upcast_trgt(data).values

    sheet.je_df[sheet.dcoords] = (trgt_pos - srce_pos)


def update_length(sheet):
    '''
    Updates the edge_df `length` column on the `coords` basis
    '''
    sheet.je_df['length'] = np.linalg.norm(sheet.je_df[sheet.dcoords],
                                           axis=1)


def update_centroid(sheet):
    '''
    Updates the face_df `coords` columns as the face's vertices
    center of mass.
    '''
    upcast_pos = sheet.upcast_srce(sheet.jv_df[sheet.coords])
    upcast_pos = upcast_pos.set_index(sheet.je_df['face'], append=True)
    sheet.face_df[sheet.coords] = upcast_pos.mean(level='face')
