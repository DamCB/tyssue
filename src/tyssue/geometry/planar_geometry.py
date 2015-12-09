import numpy as np

from .base_geometry import (scale, update_dcoords,
                            update_length, update_centroid)


def get_default_geom_specs():
    default_geom_specs = {
        "cell": {
            "num_sides": (6, np.int),
            },
        "je": {
            "nz": (0., np.float),
            },
        "settings": {
            "geometry": "plannar",
            }
        }
    return default_geom_specs

def update_all(sheet, coords=None, **geom_spec_kw):
    '''
    Updates the sheet geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the cell centroids
    * the normals to each edge associated face
    * the cell areas
    '''
    if coords is None:
        coords = sheet.coords
    geom_spec = get_default_geom_specs()
    geom_spec.update(**geom_spec_kw)

    update_dcoords(sheet, coords)
    update_length(sheet, coords)
    update_centroid(sheet, coords)
    update_normals(sheet, coords)
    update_areas(sheet)
    update_perimeters(sheet)


def update_num_sides(sheet):

    sheet.cell_df['num_sides'] = sheet.je_idx.get_level_values(
        'cell').value_counts().loc[sheet.cell_df.index]


def update_perimeters(sheet):
    '''
    Updates the perimeter of each cell.
    '''

    sheet.cell_df['perimeter'] = sheet.je_df['length'].groupby(
        level='cell').sum()


def update_normals(sheet, coords):

    cell_pos = sheet.upcast_cell(sheet.cell_df[coords]).values
    srce_pos = sheet.upcast_srce(sheet.jv_df[coords]).values
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords]).values

    normals = np.cross(srce_pos - cell_pos, trgt_pos - srce_pos)
    sheet.je_df["nz"] = normals


def update_areas(sheet):
    '''
    Updates the normal coordniate of each (srce, trgt, cell) face.
    '''
    sheet.je_df['sub_area'] = np.abs(sheet.je_df['nz']) / 2
    sheet.cell_df['area'] = sheet.je_df['sub_area'].groupby(level='cell').sum()
