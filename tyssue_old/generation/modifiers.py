"""This module provides utlities to modify an input tissue through extrusion or subdivision
"""
import pandas as pd
import numpy as np
from ..config.geometry import bulk_spec


def extrude(apical_datasets, method="homotecy", scale=0.3, vector=[0, 0, -1]):
    """Extrude a sheet to form a monlayer epithelium

    Parameters
    ----------
    * apical_datasets: dictionnary of three DataFrames,
    'vert', 'edge', 'face'
    * method: str, optional {'homotecy'|'translation'|'normals'}
    default 'homotecy'
    * scale: float, optional
    the scale factor for homotetic scaling, default 0.3.
    * vector: sequence of three floats, optional,
    used for the translation

    default [0, 0, -1]

    if `method == 'homotecy'`, the basal layer is scaled down from the
    apical one homoteticaly w/r to the center of the coordinate
    system, by a factor given by `scale`

    if `method == 'translation'`, the basal vertices are translated from
    the apical ones by the vector `vect`

    if `method == 'normals'`, basal vertices are translated from
    the apical ones along the normal of the surface at each vertex,
    by a vector whose size is given by `scale`

    """
    apical_vert = apical_datasets["vert"]
    apical_face = apical_datasets["face"]
    apical_edge = apical_datasets["edge"]

    apical_vert["segment"] = "apical"
    apical_face["segment"] = "apical"
    apical_edge["segment"] = "apical"

    coords = list("xyz")
    datasets = {}

    Nv = apical_vert.index.max() + 1
    Ne = apical_edge.index.max() + 1
    Nf = apical_face.index.max() + 1

    basal_vert = apical_vert.copy()

    basal_vert.index = basal_vert.index + Nv
    basal_vert["segment"] = "basal"

    cell_df = apical_face[coords].copy()
    cell_df.index.name = "cell"
    cell_df["is_alive"] = 1

    basal_face = apical_face.copy()
    basal_face.index = basal_face.index + Nf
    basal_face[coords] = basal_face[coords] * 1 / 3.0
    basal_face["segment"] = "basal"
    basal_face["is_alive"] = 1

    apical_edge["cell"] = apical_edge["face"]
    basal_edge = apical_edge.copy()
    # ## Flip edge so that normals are outward
    basal_edge[["srce", "trgt"]] = basal_edge[["trgt", "srce"]] + Nv
    basal_edge["face"] = basal_edge["face"] + Nf
    basal_edge.index = basal_edge.index + Ne
    basal_edge["segment"] = "basal"

    lateral_face = pd.DataFrame(
        index=apical_edge.index + 2 * Nf, columns=apical_face.columns
    )
    lateral_face["segment"] = "lateral"
    lateral_face["is_alive"] = 1

    lateral_edge = pd.DataFrame(
        index=np.arange(2 * Ne, 6 * Ne), columns=apical_edge.columns
    )

    lateral_edge["cell"] = np.repeat(apical_edge["cell"].values, 4)
    lateral_edge["face"] = np.repeat(lateral_face.index.values, 4)
    lateral_edge["segment"] = "lateral"

    lateral_edge.loc[np.arange(2 * Ne, 6 * Ne, 4), ["srce", "trgt"]] = apical_edge[
        ["trgt", "srce"]
    ].values

    lateral_edge.loc[np.arange(2 * Ne + 1, 6 * Ne, 4), "srce"] = apical_edge[
        "srce"
    ].values
    lateral_edge.loc[np.arange(2 * Ne + 1, 6 * Ne, 4), "trgt"] = basal_edge[
        "trgt"
    ].values

    lateral_edge.loc[np.arange(2 * Ne + 2, 6 * Ne, 4), ["srce", "trgt"]] = basal_edge[
        ["trgt", "srce"]
    ].values

    lateral_edge.loc[np.arange(2 * Ne + 3, 6 * Ne, 4), "srce"] = basal_edge[
        "srce"
    ].values
    lateral_edge.loc[np.arange(2 * Ne + 3, 6 * Ne, 4), "trgt"] = apical_edge[
        "trgt"
    ].values

    if method == "homotecy":
        basal_vert[coords] = basal_vert[coords] * scale
    elif method == "translation":
        for c, u in zip(coords, vector):
            basal_vert[c] = basal_vert[c] + u
    elif method == "normals":
        field = apical_edge.groupby("srce")[["nx", "ny", "nz"]].mean()
        field = -field.values * scale / np.linalg.norm(field, axis=1)[:, None]
        basal_vert[coords] = basal_vert[coords] + field
    else:
        raise ValueError(
            """
`method` argument not understood, supported values are
'homotecy', 'translation' or 'normals'
        """
        )

    datasets["cell"] = cell_df
    datasets["vert"] = pd.concat([apical_vert, basal_vert])
    datasets["vert"]["is_active"] = 1
    datasets["edge"] = pd.concat([apical_edge, basal_edge, lateral_edge])
    datasets["face"] = pd.concat([apical_face, basal_face, lateral_face])
    datasets["edge"]["is_active"] = 1
    specs = bulk_spec()

    for elem in ["vert", "edge", "face", "cell"]:
        datasets[elem].index.name = elem
        for col, value in specs[elem].items():
            if not col in datasets[elem]:
                datasets[elem][col] = value

    if (method == "normals") and (scale < 0):
        datasets["edge"][["srce", "trgt"]] = datasets["edge"][["trgt", "srce"]]
    return datasets


def create_anchors(sheet):
    """Adds an edge linked to every vertices at the boundary
    and create anchor vertices
    """
    anchor_specs = {
        "face": {"at_border": 0},
        "vert": {"at_border": 0, "is_anchor": 0},
        "edge": {"at_border": 0, "is_anchor": 0},
    }

    sheet.update_specs(anchor_specs)
    # ## Edges with no opposites denote the boundary

    free_edge = sheet.edge_df[sheet.edge_df["opposite"] == -1]
    free_vert = sheet.vert_df.loc[free_edge["srce"]]
    free_face = sheet.face_df.loc[free_edge["face"]]

    sheet.edge_df.loc[free_edge.index, "at_border"] = 1
    sheet.vert_df.loc[free_vert.index, "at_border"] = 1
    sheet.face_df.loc[free_face.index, "at_border"] = 1

    # ## Make a copy of the boundary vertices
    anchor_vert_df = free_vert.reset_index(drop=True)
    anchor_vert_df[sheet.coords] = anchor_vert_df[sheet.coords] * 1.01
    anchor_vert_df.index = anchor_vert_df.index + sheet.Nv
    anchor_vert_df["is_anchor"] = 1
    anchor_vert_df["at_border"] = 0
    anchor_vert_df["is_active"] = 0

    sheet.vert_df = pd.concat([sheet.vert_df, anchor_vert_df])
    sheet.vert_df.index.name = "vert"
    anchor_edge_df = pd.DataFrame(
        index=np.arange(sheet.Ne, sheet.Ne + free_vert.shape[0]),
        columns=sheet.edge_df.columns,
    )

    anchor_edge_df["srce"] = free_vert.index
    anchor_edge_df["trgt"] = anchor_vert_df.index
    anchor_edge_df["line_tension"] = 0
    anchor_edge_df["is_anchor"] = 1
    anchor_edge_df["face"] = -1
    anchor_edge_df["at_border"] = 0
    sheet.edge_df = pd.concat([sheet.edge_df, anchor_edge_df], sort=True)
    sheet.edge_df.index.name = "edge"
    sheet.reset_topo()


def subdivide_faces(eptm, faces):
    """Adds a vertex at the center of each face, and returns a
    new dataset

    Parameters
    ----------
    eptm: a :class:`Epithelium` instance
    faces: list,
     indices of the faces to be subdivided

    Returns
    -------
    new_dset: dict
      a dataset with the new faces devided

    """

    face_df = eptm.face_df.loc[faces]

    remaining = eptm.face_df.index.delete(faces)
    untouched_faces = eptm.face_df.loc[remaining]
    edge_df = pd.concat([eptm.edge_df[eptm.edge_df["face"] == face] for face in faces])
    verts = set(edge_df["srce"])
    vert_df = eptm.vert_df.loc[verts]

    Nsf = face_df.shape[0]
    Nse = edge_df.shape[0]

    eptm.vert_df["subdiv"] = 0
    untouched_faces["subdiv"] = 0
    eptm.edge_df["subdiv"] = 0

    new_vs_idx = pd.Series(np.arange(eptm.Nv, eptm.Nv + Nsf), index=face_df.index)
    upcast_new_vs = new_vs_idx.loc[edge_df["face"]].values

    new_vs = pd.DataFrame(
        index=pd.Index(np.arange(eptm.Nv, eptm.Nv + Nsf), name="vert"),
        columns=vert_df.columns,
    )
    new_es = pd.DataFrame(
        index=pd.Index(np.arange(eptm.Ne, eptm.Ne + 2 * Nse), name="edge"),
        columns=edge_df.columns,
    )

    new_vs["subdiv"] = 1
    # new_fs['subdiv'] = 1
    new_es["subdiv"] = 1
    if "cell" in edge_df.columns:
        new_es["cell"] = np.concatenate([edge_df["cell"], edge_df["cell"]])
    new_vs[eptm.coords] = face_df[eptm.coords].values
    # eptm.edge_df.loc[edge_df.index, 'face'] = new_fs.index
    # new_es['face'] = np.concatenate([new_fs.index,
    #                                  new_fs.index])
    new_es["face"] = np.concatenate([edge_df["face"], edge_df["face"]])
    new_es["srce"] = np.concatenate([edge_df["trgt"].values, upcast_new_vs])
    new_es["trgt"] = np.concatenate([upcast_new_vs, edge_df["srce"].values])
    new_dset = {
        "edge": pd.concat([eptm.edge_df, new_es]),
        "face": eptm.face_df,  # pd.concat([untouched_faces, new_fs]),
        "vert": pd.concat([eptm.vert_df, new_vs]),
    }

    if "cell" in edge_df.columns:
        new_dset["cell"] = eptm.cell_df

    return new_dset
