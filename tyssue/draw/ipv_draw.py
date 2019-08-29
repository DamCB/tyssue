"""3D visualisation inside the notebook.
"""
import warnings
import numpy as np
import pandas as pd
from matplotlib import cm
from ipywidgets import interact

from ..config.draw import sheet_spec
from ..utils.utils import spec_updater, get_sub_eptm

try:
    import ipyvolume as ipv
except ImportError:
    print(
        """
This module needs ipyvolume to work.
You can install it with:
$ conda install -c conda-forge ipyvolume
"""
    )


def browse_history(history, coords=["x", "y", "z"], **draw_specs_kw):

    times = history.time_stamps
    num_frames = times.size
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
    sheet = history.retrieve(0)
    ipv.clear()
    fig, meshes = sheet_view(sheet, coords, **draw_specs_kw)

    lim_inf = sheet.vert_df[sheet.coords].min().min()
    lim_sup = sheet.vert_df[sheet.coords].max().max()
    ipv.xyzlim(lim_inf, lim_sup)

    def set_frame(i=0):
        fig.animation = 0
        t = times[i]
        meshes = _get_meshes(history.retrieve(t), coords, draw_specs)
        update_view(fig, meshes)

    ipv.show()
    interact(set_frame, i=(0, num_frames - 1))


def update_view(fig, meshes):

    for old, new in zip(fig.meshes, meshes):
        old.x = new.x
        old.y = new.y
        old.z = new.z
        old.color = new.color
        old.triangles = new.triangles
        old.lines = new.lines


def sheet_view(sheet, coords=["x", "y", "z"], **draw_specs_kw):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    fig: a :class:`ipyvolume.widgets.Figure` widget
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """

    # ipv.style.use(["dark", "minimal"])
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
    fig = ipv.gcf()
    fig.meshes = fig.meshes + _get_meshes(sheet, coords, draw_specs)
    box_size = max(*(np.ptp(sheet.vert_df[u]) for u in sheet.coords))
    border = 0.05 * box_size
    lim_inf = sheet.vert_df[sheet.coords].min().min() - border
    lim_sup = sheet.vert_df[sheet.coords].max().max() + border
    ipv.xyzlim(lim_inf, lim_sup)
    return fig, fig.meshes


def view_ipv(sheet, coords=["x", "y", "z"], **edge_specs):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    fig: a :class:`ipyvolume.widgets.Figure` widget
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    warnings.warn("`view_ipv` is deprecated, use the more generic `sheet_view`")
    mesh = edge_mesh(sheet, coords, **edge_specs)
    fig = ipv.gcf()
    fig.meshes = fig.meshes + [mesh]
    box_size = max(*(np.ptp(sheet.vert_df[u]) for u in sheet.coords))
    border = 0.05 * box_size
    lim_inf = sheet.vert_df[sheet.coords].min().min() - border
    lim_sup = sheet.vert_df[sheet.coords].max().max() + border
    ipv.xyzlim(lim_inf, lim_sup)

    return fig, mesh


def edge_mesh(sheet, coords, **edge_specs):
    """
    Creates a ipyvolume Mesh of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    spec = sheet_spec()["edge"]
    spec.update(**edge_specs)
    if callable(spec["color"]):
        spec["color"] = spec["color"](sheet)

    if isinstance(spec["color"], str):
        color = spec["color"]
    elif hasattr(spec["color"], "__len__"):
        color = _wire_color_from_sequence(spec, sheet)[:, :3]

    u, v, w = coords
    mesh = ipv.Mesh(
        x=sheet.vert_df[u],
        y=sheet.vert_df[v],
        z=sheet.vert_df[w],
        lines=sheet.edge_df[["srce", "trgt"]].astype(dtype=np.uint32),
        color=color,
    )
    return mesh


def face_mesh(sheet, coords, **face_draw_specs):
    """
    Creates a ipyvolume Mesh of the face polygons
    """
    Ne, Nf = sheet.Ne, sheet.Nf
    if callable(face_draw_specs["color"]):
        face_draw_specs["color"] = face_draw_specs["color"](sheet)

    if isinstance(face_draw_specs["color"], str):
        color = face_draw_specs["color"]

    elif hasattr(face_draw_specs["color"], "__len__"):
        color = _face_color_from_sequence(face_draw_specs, sheet)[:, :3]

    if "visible" in sheet.face_df.columns:
        edges = sheet.edge_df[sheet.upcast_face(sheet.face_df["visible"])].index
        _sheet = get_sub_eptm(sheet, edges)
        if _sheet is not None:
            sheet = _sheet
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                edges = edges.values.astype(np.uint32)
                indexer = np.concatenate([faces, edges + Nf, edges + Ne + Nf])
                color = color.take(indexer, axis=0)

    epsilon = face_draw_specs.get("epsilon", 0)
    up_srce = sheet.edge_df[["s" + c for c in coords]]
    up_trgt = sheet.edge_df[["t" + c for c in coords]]

    Ne, Nf = sheet.Ne, sheet.Nf

    if epsilon > 0:
        up_face = sheet.edge_df[["f" + c for c in coords]].values
        up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
        up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    mesh_ = np.concatenate(
        [sheet.face_df[coords].values, up_srce.values, up_trgt.values]
    )

    triangles = np.vstack(
        [sheet.edge_df["face"], np.arange(Ne) + Nf, np.arange(Ne) + Ne + Nf]
    ).T.astype(dtype=np.uint32)

    mesh = ipv.Mesh(
        x=mesh_[:, 0], y=mesh_[:, 1], z=mesh_[:, 2], triangles=triangles, color=color
    )
    return mesh


def _wire_color_from_sequence(edge_spec, sheet):
    """
    """
    color_ = edge_spec["color"]
    cmap = cm.get_cmap(edge_spec.get("colormap", "viridis"))
    if color_.shape in [(sheet.Nv, 3), (sheet.Nv, 4)]:
        return np.asarray(color_)
    elif color_.shape == (sheet.Nv,):
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Nv, 3)) * 0.7
        return cmap((color_ - color_.min()) / np.ptp(color_))

    elif color_.shape in [(sheet.Ne, 3), (sheet.Ne, 4)]:
        color_ = pd.DataFrame(color_, index=sheet.edge_df.index)
        color_["srce"] = sheet.edge_df["srce"]
        color_ = color_.groupby("srce").mean().values
        return color_
    elif color_.shape == (sheet.Ne,):
        color_ = pd.DataFrame(color_, index=sheet.edge_df.index)
        color_["srce"] = sheet.edge_df["srce"]
        color_ = color_.groupby("srce").mean().values.ravel()
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Nv, 3)) * 0.7
        return cmap((color_ - color_.min()) / np.ptp(color_))
    else:
        raise ValueError("The 'color' value of the spec doesn't have a correct shape.")


def _face_color_from_sequence(face_spec, sheet):
    color_ = face_spec["color"]

    cmap = cm.get_cmap(face_spec.get("colormap", "viridis"))
    Nf, Ne = sheet.Nf, sheet.Ne
    color_min, color_max = face_spec.get("color_range", (color_.min(), color_.max()))

    face_mesh_shape = Nf + 2 * Ne

    if color_.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
        return np.concatenate([color_, color_, color_])

    elif color_.shape == (sheet.Nf,):
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((face_mesh_shape, 3)) * 0.5

        normed = (color_ - color_min) / (color_max - color_min)
        up_color = sheet.upcast_face(normed).values
        return cmap(np.concatenate([normed, up_color, up_color]))

    else:
        raise ValueError(
            "shape of `face_spec['color']` must be either (Nf, 3), (Nf, 4) or (Nf,)"
        )


def _get_meshes(sheet, coords, draw_specs):

    meshes = []
    edge_spec = draw_specs["edge"]
    if edge_spec["visible"]:
        edges = edge_mesh(sheet, coords, **edge_spec)
        meshes.append(edges)
    else:
        edges = None

    face_spec = draw_specs["face"]
    if face_spec["visible"]:
        faces = face_mesh(sheet, coords, **face_spec)
        meshes.append(faces)
    else:
        faces = None
    return meshes
