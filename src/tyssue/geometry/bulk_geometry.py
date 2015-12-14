import numpy as np
from .base_geometry import update_dcoords, scale, update_length
from .planar_geometry import update_perimeters
from .sheet_geometry import update_normals, update_areas


def get_default_geom_specs():
    default_geom_specs = {
        "cell": {
            'num_faces': (0, np.int),
            'vol': (0, np.float),
            'area':  (0, np.float),
            },
        "face": {
            "num_sides": (6, np.int),
            },
        "je": {
            "nx": (0., np.float),
            "ny": (0., np.float),
            "nz": (0., np.float),
            },
        }
    return default_geom_specs

def update_all(eptm):
    '''
    Updates the eptm geometry by updating:
    * the edge vector coordinates
    * the edge lengths
    * the face centroids
    * the normals to each edge associated face
    * the face areas
    * the vertices heights (depends on geometry)
    * the face volumes (depends on geometry)

    '''
    update_dcoords(eptm)
    update_length(eptm)
    update_centroid(eptm)
    update_normals(eptm)
    update_areas(eptm)
    update_perimeters(eptm)
    update_vol(eptm)


def update_vol(eptm):
    '''

    '''
    face_pos = eptm.upcast_face(eptm.face_df[eptm.coords])
    cell_pos = eptm.upcast_cell(eptm.cell_df[eptm.coords])

    eptm.je_df['sub_vol'] = np.dot((cell_pos - face_pos).values,
                                   eptm.je_df[eptm.ncoords]) / 6

    eptm.cell_df['vol'] = eptm.sum_cell(eptm.je_df['sub_vol'])

def update_centroid(eptm):

    upcast_pos = eptm.upcast_srce(eptm.jv_df[eptm.coords])
    upcast_pos = upcast_pos.set_index(eptm.je_mindex)
    eptm.face_df[eptm.coords] = upcast_pos.mean(level='face')
    eptm.cell_df[eptm.coords] = upcast_pos.mean(level='cell')
