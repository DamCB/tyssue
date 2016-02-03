import numpy as np

from .base_geometry import (update_dcoords, scale,
                            update_length, update_centroid)


def get_default_geom_specs():
    default_geom_specs = {
        "face": {
            "num_sides": 6,
            },
        "je": {
            "nz": 0.,
            },
        "settings": {
            "geometry": "plannar",
            }
        }
    return default_geom_specs

def update_all(sheet, **geom_spec_kw):
    '''
    Updates the sheet geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the face centroids
    * the normals to each edge associated face
    * the face areas
    '''
    geom_spec = get_default_geom_specs()
    geom_spec.update(**geom_spec_kw)

    update_dcoords(sheet)
    update_length(sheet)
    update_centroid(sheet)
    update_normals(sheet)
    update_areas(sheet)
    update_perimeters(sheet)


def update_perimeters(sheet):
    '''
    Updates the perimeter of each face.
    '''
    sheet.face_df['perimeter'] = sheet.sum_face(sheet.je_df['length'])


def update_normals(sheet):

    coords = sheet.coords
    face_pos = sheet.upcast_face(sheet.face_df[coords]).values
    srce_pos = sheet.upcast_srce(sheet.jv_df[coords]).values
    trgt_pos = sheet.upcast_trgt(sheet.jv_df[coords]).values

    normals = np.cross(srce_pos - face_pos, trgt_pos - srce_pos)
    sheet.je_df["nz"] = normals


def update_areas(sheet):
    '''
    Updates the normal coordniate of each (srce, trgt, face) face.
    '''
    sheet.je_df['sub_area'] = np.abs(sheet.je_df['nz']) / 2
    sheet.face_df['area'] = sheet.sum_face(sheet.je_df['sub_area'])
