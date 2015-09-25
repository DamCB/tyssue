import numpy as np
import pandas as pd


def update_all(sheet, coords=['x', 'y', 'z']):
    '''
    Updates the sheet geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the cell centroids
    * the normals to each edge associated face
    * the cells area
    '''

    update_dcoords(sheet, coords)
    update_length(sheet, coords)
    update_centroid(sheet, coords)
    update_normals(sheet, coords)
    update_areas(sheet, coords)
    update_perimeters(sheet, coords)


def update_dcoords(sheet, coords=['x', 'y', 'z']):
    '''
    Update the edge vector coordinates  on the
    `coords` basis (`['x', 'y', 'z']` by default). Modifies the corresponding
    columns (i.e `['dx', 'dy', 'dz']`) in sheet.edge_df.
    '''
    dcoords = ['d'+c for c in coords]
    data = sheet.jv_df[coords]
    srce_pos = sheet.upcast_srce(data).values
    trgt_pos = sheet.upcast_trgt(data).values

    sheet.je_df[dcoords] = (trgt_pos - srce_pos)


def update_length(sheet, coords=['x', 'y', 'z']):
    '''
    Updates the edge_df `length` column on the `coords` basis
    '''
    dcoords = ['d' + c for c in coords]
    sheet.je_df['length'] = np.linalg.norm(sheet.je_df[dcoords],
                                           axis=1)


def update_centroid(sheet, coords=['x', 'y', 'z']):
    '''
    Updates the cell_df `coords` columns as the cell's vertices
    center of mass.
    '''
    upcast_pos = sheet.upcast_srce(sheet.jv_df[coords])
    # Now it's easy to compute the centroid of each cell
    sheet.cell_df[coords] = upcast_pos.groupby(level='cell').mean()


def update_normals(sheet, coords=['x', 'y', 'z']):
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


def update_areas(sheet, coords=['x', 'y', 'z']):
    '''
    Updates the normal coordniate of each (srce, trgt, cell) face.
    '''

    ncoords = ['n' + c for c in coords]
    sheet.je_df['sub_area'] = np.linalg.norm(sheet.je_df[ncoords], axis=1) / 2
    sheet.cell_df['area'] = sheet.je_df['sub_area'].groupby(level='cell').sum()


def update_perimeters(sheet, coords=['x', 'y', 'z']):
    '''
    Updates the perimeter of each cell.
    '''

    sheet.cell_df['perimeter'] = sheet.je_df['length'].groupby(
        level='cell').sum()
