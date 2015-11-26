import numpy as np


def get_default_geom_specs():
    default_geom_specs = {
        "cell": {
            "num_sides": (6, np.int),
            },
        "jv": {
            "rho": (0., np.float),
            "basal_shift": (4., np.float), # previously rho_lumen
            },
        "je": {
            "nx": (0., np.float),
            "ny": (0., np.float),
            "nz": (0., np.float),
            },
        "settings": {
            "geometry": "cylindrical",
            "height_axis": 'z'
            }
        }
    return default_geom_specs


def scale(sheet, delta, coords):
    ''' Scales the coordinates `coords`
    by a factor `delta`
    '''
    sheet.jv_df[coords] = sheet.jv_df[coords] * delta


def update_all(sheet, coords=None, **geom_spec_kw):
    '''
    Updates the sheet geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the cell centroids
    * the normals to each edge associated face
    * the cell areas
    * the vertices heights (depends on geometry)
    * the cell volumes (depends on geometry)

    '''
    if coords is None:
        coords = sheet.coords
    geom_spec = get_default_geom_specs()
    geom_spec.update(**geom_spec_kw)

    update_dcoords(sheet, coords)
    update_length(sheet, coords)
    update_centroid(sheet, coords)
    if geom_spec['settings']['geometry'] == 'cylindrical':
        update_height_cylindrical(sheet, coords,
                                  geom_spec['settings'])
    elif geom_spec['settings']['geometry'] == 'flat':
        update_height_flat(sheet, geom_spec['settings'])
    update_normals(sheet, coords)
    update_areas(sheet, coords)
    update_perimeters(sheet)
    update_vol(sheet)


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


def update_normals(sheet, coords):
    '''
    Updates the cell_df `coords` columns as the cell's vertices
    center of mass.
    '''
    cell_pos = sheet.upcast_cell(sheet.cell_df[coords]).values
    srce_pos = sheet.upcast_srce(sheet.jv_df[coords]).values
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords]).values

    normals = np.cross(srce_pos - cell_pos, trgt_pos - srce_pos)
    if len(coords) == 2:
        sheet.je_df['nz'] = normals
    else:
        ncoords = ['n' + c for c in coords]
        sheet.je_df[ncoords] = normals

def update_num_sides(sheet):

    sheet.cell_df['num_sides'] = sheet.je_idx.get_level_values(
        'cell').value_counts().sort_index()


def update_areas(sheet, coords):
    '''
    Updates the normal coordniate of each (srce, trgt, cell) face.
    '''
    ncoords = ['n' + c for c in coords]
    sheet.je_df['sub_area'] = np.linalg.norm(sheet.je_df[ncoords], axis=1) / 2
    sheet.cell_df['area'] = sheet.je_df['sub_area'].groupby(level='cell').sum()


def update_perimeters(sheet):
    '''
    Updates the perimeter of each cell.
    '''

    sheet.cell_df['perimeter'] = sheet.je_df['length'].groupby(
        level='cell').sum()


def update_vol(sheet):
    '''
    Note that this is an approximation of the sheet geometry
    package.

    '''
    sheet.je_df['sub_vol'] = sheet.upcast_srce(sheet.jv_df['height']) * sheet.je_df['sub_area']
    sheet.cell_df['vol'] = sheet.je_df['sub_vol'].sum(level='cell')

# ### Cylindrical geometry specific

def update_height_cylindrical(sheet, coords, settings):
    '''
    Updates each cell height in a cylindrical geometry.
    e.g. cell anchor is assumed to lie at a distance
    `parameters['basal_shift']` from the third axis of
    the triplet `coords`
    '''
    w = settings['height_axis']
    u, v = (c for c in coords if c != w)
    sheet.jv_df['rho'] = np.hypot(sheet.jv_df[v],
                                  sheet.jv_df[u])
    sheet.jv_df['height'] = (sheet.jv_df['rho'] -
                             sheet.jv_df['basal_shift'])


# ### Flat geometry specific

def update_height_flat(sheet, settings):
    '''
    Updates each cell height in a flat geometry.
    e.g. cell anchor is assumed to lie at a distance
    `parameters['basal_shift']` from the plane where
    the coordinate `coord` is equal to 0
    '''
    coord = settings['height_axis']
    sheet.jv_df['rho'] = sheet.jv_df[coord]
    sheet.jv_df['height'] = sheet.jv_df[coord] - sheet.jv_df['basal_shift']
