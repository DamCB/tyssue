import numpy as np

# excluding this function from tests
# as it is pretty trivial.
def get_default_geom_specs():  # pragma: no cover
    default_geom_specs = {"cc": {"nz": 0.0}}
    return default_geom_specs


def scale(ccmesh, delta, coords):
    # original line (spurious 'ccmesh' on right hand side) :
    # ccmesh.cell_df[coords] = ccmesh.ccmesh.cell_df[coords] * delta
    ccmesh.cell_df[coords] = ccmesh.cell_df[coords] * delta


def update_dcoords(ccmesh):
    data = ccmesh.cell_df[ccmesh.coords]
    srce_pos = ccmesh.upcast_srce(data)
    trgt_pos = ccmesh.upcast_trgt(data)
    ccmesh.cc_df[ccmesh.dcoords] = trgt_pos - srce_pos


def update_length(ccmesh):
    ccmesh.cc_df["length"] = np.linalg.norm(ccmesh.cc_df[ccmesh.dcoords], axis=1)


def update_all(ccmesh):
    update_dcoords(ccmesh)
    update_length(ccmesh)
