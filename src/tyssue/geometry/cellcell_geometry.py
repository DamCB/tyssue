import numpy as np

def get_default_geom_specs():
    default_geom_specs = {
        "cc": {
            "nz": 0.,
            },
        }
    return default_geom_specs



def scale(ccmesh, delta, coords):
    ccmesh.cell_df[coords] = ccmesh.ccmesh.cell_df[coords] * delta

def update_dcoords(ccmesh):

    data = ccmesh.cell_df[ccmesh.coords]
    srce_pos = ccmesh.upcast_srce(data)
    trgt_pos = ccmesh.upcast_trgt(data)
    ccmesh.cc_df[ccmesh.dcoords] = trgt_pos - srce_pos

def update_length(ccmesh):
    ccmesh.cc_df['length'] = np.linalg.norm(ccmesh.cc_df[ccmesh.dcoords], axis=1)

def update_all(ccmesh):
    update_dcoords(ccmesh)
    update_length(ccmesh)
