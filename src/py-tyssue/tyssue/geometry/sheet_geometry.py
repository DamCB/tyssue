import numpy as np
import pandas as pd

default_params = {"rho_lumen": 4.0,
                  "geometry": "cylindrical",
                  "height_axis": 'z'}

default_coords = ('x', 'y', 'z')


def update_all(sheet, coords=default_coords, 
               parameters=None):
    '''
    Updates the sheet geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the cell centroids
    * the normals to each edge associated face
    * the cells area
    '''

    if parameters is None:
        parameters = default_params
    parameters.update(default_params)

    update_dcoords(sheet, coords)
    update_length(sheet, coords)
    update_centroid(sheet, coords)
    update_normals(sheet, coords)
    update_areas(sheet, coords)
    update_perimeters(sheet)
    if parameters['geometry'] == 'cylindrical':
        update_height_cylindrical(sheet, parameters)
    elif parameters['geometry'] == 'flat':
        update_height_flat(sheet, parameters)
    update_vol(sheet)


def update_dcoords(sheet, coords=default_coords):
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


def update_length(sheet, coords=default_coords):
    '''
    Updates the edge_df `length` column on the `coords` basis
    '''
    dcoords = ['d' + c for c in coords]
    sheet.je_df['length'] = np.linalg.norm(sheet.je_df[dcoords],
                                           axis=1)


def update_centroid(sheet, coords=default_coords):
    '''
    Updates the cell_df `coords` columns as the cell's vertices
    center of mass.
    '''
    upcast_pos = sheet.upcast_srce(sheet.jv_df[coords])
    sheet.cell_df[coords] = upcast_pos.groupby(level='cell').mean()


def update_normals(sheet, coords=default_coords):
    '''
    Updates the cell_df `coords` columns as the cell's vertices
    center of mass.
    '''
    cell_pos = sheet.upcast_cell(sheet.cell_df[coords]).values
    srce_pos = sheet.upcast_srce(sheet.jv_df[coords]).values
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords]).values

    normals = np.cross(srce_pos - cell_pos, trgt_pos - cell_pos)
    if len(coords) == 2:
        sheet.je_df['nz'] = normals
    else:
        ncoords = ['n' + c for c in coords]
        sheet.je_df[ncoords] = normals


def update_areas(sheet, coords=default_coords):
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
    package. Cells are assumed to be anchored by their center
    '''
    sheet.cell_df['vol'] = sheet.cell_df['height'] * sheet.cell_df['area']

# ### Cylindrical geometry specific

def update_height_cylindrical(sheet, parameters,
                              coords=default_coords):
    '''
    Updates each cell height in a cylindrical geometry.
    e.g. cell anchor is assumed to lie at a distance 
    `parameters['rho_lumen']` from the third axis of
    the triplet `coords`
    '''
    w = parameters['height_axis']
    u, v = (c for c in coords if c != w)

    sheet.cell_df['height'] = np.hypot(sheet.cell_df[v],
                                       sheet.cell_df[u]) - parameters['rho_lumen']
    sheet.jv_df['height'] = np.hypot(sheet.jv_df[v],
                                     sheet.jv_df[u]) - parameters['rho_lumen']

# ### Flat geometry specific

def update_height_flat(sheet, parameters,
                       coord='z'):
    '''
    Updates each cell height in a flat geometry.
    e.g. cell anchor is assumed to lie at a distance 
    `parameters['rho_lumen']` from the plane where
    the coordinate `coord` is equal to 0
    '''
    sheet.cell_df['height'] = sheet.cell_df[coord] - parameters['rho_lumen']
    sheet.jv_df['height'] = sheet.jv_df[coord] - parameters['rho_lumen']
