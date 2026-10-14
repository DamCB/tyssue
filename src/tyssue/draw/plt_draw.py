"""
Matplotlib based plotting
"""
import logging
import pathlib
import shutil
import subprocess
import sys
import tempfile
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ipywidgets import interactive
from matplotlib import colormaps
from matplotlib.collections import LineCollection, PatchCollection, PolyCollection
from matplotlib.patches import Arc, FancyArrow, PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.collections as mcollections
import matplotlib.patches as mpatches

from ..config.draw import sheet_spec
from ..utils.utils import get_sub_eptm, spec_updater

IMAGEMAGICK_INSTALL_HINT = {
    "darwin": "brew install imagemagick",
    "win32": "https://imagemagick.org/script/download.php#windows",
}


def imagemagick_cmd():
    """Return the ImageMagick CLI to use for gif conversion, or None if absent.

    ``convert`` is not used on Windows, where it is an unrelated system tool.
    """
    magick = shutil.which("magick")  # ImageMagick 7
    if magick is not None:
        return magick
    if sys.platform != "win32":
        return shutil.which("convert")  # ImageMagick 6
    return None


def _require_imagemagick():
    """Return the ImageMagick CLI, raising an actionable error if unavailable."""
    cmd = imagemagick_cmd()
    if cmd is None:
        hint = IMAGEMAGICK_INSTALL_HINT.get(sys.platform, "apt install imagemagick")
        msg = (
            "ImageMagick is required to write gifs but was not found on PATH. "
            f"Install it with: {hint}"
        )
        raise RuntimeError(msg)
    return cmd


COORDS = ["x", "y"]
COORDS3D = ["x", "y", "z"]

log = logging.getLogger(__name__)


def deep_update(base, updates):
    for key, value in updates.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def patch_2d_collections_to_3d(ax):
    """Replace any 2D LineCollections on a 3D axes with proper Line3DCollection instances."""
    replacements = []
    for col in ax.collections:
        if type(col) is mcollections.LineCollection:
            segments = col.get_segments()
            segments_3d = [
                seg if seg.shape[1] == 3 else np.hstack([seg, np.zeros((len(seg), 1))])
                for seg in segments
            ]
            new_col = Line3DCollection(segments_3d)
            new_col.set_color(col.get_colors())
            new_col.set_linewidth(col.get_linewidths())
            replacements.append((col, new_col))

    for old, new in replacements:
        old.remove()
        ax.add_collection3d(new)


def browse_history(
    history,
    coords=["x", "y"],
    start=None,
    stop=None,
    size=None,
    draw_func=None,
    margin=5,
    **draw_kwds,
):
    """Returns a browser widget with 2D plots of the epithelium"""
    if draw_func is None:
        if draw_kwds.get("mode") in ("quick", None):
            draw_func = quick_edge_draw
        else:
            draw_func = sheet_view

    times = history.slice(start, stop, size)
    size = times.size
    x, y = coords = draw_kwds.get("coords", history.sheet.coords[:2])

    sheet0 = history.retrieve(0)
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    delta = (bounds.loc["max"] - bounds.loc["min"]).max()
    margin = delta * margin / 100
    xlim = bounds.loc["min", x] - margin, bounds.loc["max", x] + margin
    ylim = bounds.loc["min", y] - margin, bounds.loc["max", y] + margin

    def set_frame(i=0):
        t = times[i]
        sheet = history.retrieve(t)
        fig = plt.figure(2)
        ax = fig.subplots()
        fig, ax = draw_func(sheet, ax=ax, **draw_kwds)
        ax.set(xlim=xlim, ylim=ylim)
        plt.show()

    widget = interactive(set_frame, i=(0, size - 1))
    widget.layout.height = "500px"
    return widget


def create_gif(
    history,
    output,
    num_frames=None,
    interval=None,
    draw_func=None,
    margin=5,
    dpi=200,
    **draw_kwds,
):
    """Creates an animated gif of the recorded history.

    You need imagemagick on your system for this function to work.

    Parameters
    ----------

    history : a :class:`tyssue.History` object
    output : path to the output gif file
    num_frames : int, the number of frames in the gif
    interval : tuples, define begin and end frame of the gif
    draw_func : a drawing function
         this function must take a `sheet` object as first argument
         and return a `fig, ax` pair. Defaults to quick_edge_draw
         (aka sheet_view with quick mode)
    margin : int, the graph margins in percents, default 5
         if margin is -1, let the draw function decide

    **draw_kwds are passed to the drawing function

    """
    if draw_func is None:
        draw_func = sheet_view

    graph_dir = pathlib.Path(tempfile.mkdtemp())
    x, y = coords = draw_kwds.get("coords", history.sheet.coords[:2])
    sheet0 = history.retrieve(0)
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    delta = (bounds.loc["max"] - bounds.loc["min"]).max()
    margin = delta * margin / 100
    xlim = bounds.loc["min", x] - margin, bounds.loc["max", x] + margin
    ylim = bounds.loc["min", y] - margin, bounds.loc["max", y] + margin

    if interval is None:
        start, stop = None, None
    else:
        start, stop = interval[0], interval[1]

    for i, (t, sheet) in enumerate(history.browse(start, stop, num_frames)):
        try:
            fig, ax = draw_func(sheet, **draw_kwds)
            plt.title(f"t = {t:.2f}")
        except Exception as e:
            print(f"Droped frame {i}")
            print(e)
            continue

        if isinstance(ax, plt.Axes) and margin >= 0:
            ax.set(xlim=xlim, ylim=ylim)
        fig.savefig(
            graph_dir / f"movie_{i:04d}.png",
            dpi = dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

    try:
        magick = _require_imagemagick()
        subprocess.run(
            [magick, (graph_dir / "movie_*.png").as_posix(), output], check=True
        )
    finally:
        shutil.rmtree(graph_dir)

def create_gif_3d(
    history,
    output,
    num_frames=None,
    interval=None,
    draw_func=None,
    margin=5,
    dpi=200,
    view_angle=(30, 45),
    dynamic_draw_kwds=None,
    legend = None,
    cull_back_edges=False,
    **draw_kwds,
):
    """Creates an animated 3D gif of the recorded history.

    You need imagemagick on your system for this function to work.
    The draw_func must accept an `ax` keyword argument and plot into
    the provided Axes3D instance.

    Parameters
    ----------
    history : a :class:`tyssue.History` object
    output : path to the output gif file
    num_frames : int, the number of frames in the gif
    interval : tuple, define begin and end frame of the gif
    draw_func : a drawing function
        Must take a `sheet` object as first argument and return a
        `fig, ax` pair. Must accept an `ax` keyword argument so it
        can plot into the pre-created Axes3D. Defaults to sheet_view.
    margin : int, graph margins in percent, default 5.
        If -1, let the draw function decide.
    dpi : int, resolution of each saved frame, default 200
    view_angle : tuple (elev, azim), default (30, 45)
        Elevation and azimuth angles for the 3D camera.
    dynamic_draw_kwds : list of functions or None
        list of functions that are called to update the draw_kds

        Example::

            dynamic_draw_kwds={
                "face_colors": lambda sheet: sheet.face_df["myogen"].values,
            }

    **draw_kwds are passed unchanged to the drawing function
    """
    if draw_func is None:
        draw_func = sheet_view_3d  # default to the 3D view

    draw_kwds.setdefault("view_angle", view_angle)

    if dynamic_draw_kwds is None:
        dynamic_draw_kwds = []

    graph_dir = pathlib.Path(tempfile.mkdtemp())

    if interval is None:
        start, stop = None, None
    else:
        start, stop = interval[0], interval[1]

    coords = draw_kwds.get("coords", history.sheet.coords[:3])
    x, y, z = coords[0], coords[1], coords[2]
    sheet0 = history.retrieve(0)
    bounds = sheet0.vert_df[coords].describe().loc[["min", "max"]]
    delta = (bounds.loc["max"] - bounds.loc["min"]).max()
    margin_val = delta * margin / 100
    xlim = bounds.loc["min", x] - margin_val, bounds.loc["max", x] + margin_val
    ylim = bounds.loc["min", y] - margin_val, bounds.loc["max", y] + margin_val
    zlim = bounds.loc["min", z] - margin_val, bounds.loc["max", z] + margin_val

    # same figure size for every frame, taller for tissues elongated along z
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    fig_w = 6.4
    fig_h = min(12.0, 4.8 * max(1.0, zr / max(xr, yr, 1e-9)))

    for i, (t, sheet) in enumerate(history.browse(start, stop, num_frames)):
        try:
            if len(dynamic_draw_kwds) > 0:
                for func in dynamic_draw_kwds:
                    update_kwds = func(sheet)
                    draw_kwds = deep_update(draw_kwds, update_kwds)

            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_subplot(111, projection="3d")
            ax.view_init(elev=view_angle[0], azim=view_angle[1])

            fig, ax = draw_func(sheet, ax=ax, legend=legend, cull_back_edges=cull_back_edges, **draw_kwds)
            patch_2d_collections_to_3d(ax)
            ax.set(xlim=xlim, ylim=ylim, zlim=zlim)
            # box aspect and tick size from the fixed limits, the same every frame
            _set_axes_proportional_3d(ax)
            _auto_tick_fontsize_3d(ax, base_size=8, min_size=4)
            ax.set_title(f"t = {t:.2f}")

        except Exception as e:
            print(f"Dropped frame {i}")
            print(e)
            continue

        # no bbox_inches="tight": all frames must have the same size
        fig.savefig(
            graph_dir / f"movie_{i:04d}.png",
            dpi=dpi,
        )
        plt.close(fig)

    try:
        magick = _require_imagemagick()
        subprocess.run(
            [magick, (graph_dir / "movie_*.png").as_posix(), output], check=True
        )
    finally:
        shutil.rmtree(graph_dir)

def sheet_view(sheet, coords=COORDS, ax=None, cbar_axis=None, legend=None, **draw_specs_kw):
    """Base view function, parametrizable
    through draw_secs
    The default sheet_spec specification is:
    {
        "edge": {
            "visible": true,
            "width": 0.5,
            "head_width": 0.0,
            "length_includes_head": true,
            "shape": "right",
            "color": "#2b5d0a",
            "alpha": 0.8,
            "zorder": 1,
            "colormap": "viridis"
        },
        "vert": {
            "visible": false,
            "s": 100,
            "color": "#000a4b",
            "alpha": 0.3,
            "zorder": 2
        },
        "grad": {
            "color":"#000a4b",
            "alpha":0.5,
            "width":0.04
        },
        "face": {
            "visible": false,
            "color":"#8aa678",
            "alpha": 1.0,
            "zorder": -1
        },
        "axis": {
            "autoscale": true,
            "color_bar": false,
            "color_bar_cmap":"viridis",
            "color_bar_range":false,
            "color_bar_label":false,
            "color_bar_target":"face"
        }
    }

    Note
    ----

    Important note for quantitative colormap plots: make sure to normalize your
    values before getting the colors using

         draw_specs["face"]["color"] = cmap(pandas_holding_quantity_of_interest)

    For each plot normalize with respect to the current values
    (max and min) such that they lie between and including 0 to 1.
    Note that if you want to keep a constant colorbar range you have
    to choose the normalization to match the max and min of the color
    bar range you chose.
    """
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    if (ax is None) and (cbar_axis is None):
        fig = plt.figure()
        grid0 = plt.GridSpec(10, 10)
        grid0.update(wspace=0.0)
        ax = fig.add_subplot(grid0[:, :9])
    else:
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

    vert_spec = draw_specs["vert"]
    if vert_spec["visible"]:
        ax = draw_vert(sheet, coords, ax, **vert_spec)

    edge_spec = draw_specs["edge"]
    if edge_spec["visible"]:
        ax = draw_edge(sheet, coords, ax, **edge_spec)

    face_spec = draw_specs["face"]
    if face_spec["visible"]:
        ax = draw_face(sheet, coords, ax, **face_spec)

    axis_spec = draw_specs.get("axis", {})
    if axis_spec.get("autoscale"):
        ax.autoscale()
        ax.set_aspect("equal")
    else:
        ax.set_xlim(axis_spec["x_min"], axis_spec["x_max"])
        ax.set_ylim(axis_spec["y_min"], axis_spec["y_max"])
        ax.set_aspect("equal")

    if not axis_spec.get("color_bar"):
        return fig, ax
    else:
        cbar_axis = fig.add_subplot(grid0[:, 9])
        cmap = colormaps[axis_spec.get("color_bar_cmap")]
        if not axis_spec.get("color_bar_range"):
            warnings.warn(
                """Since the quanity of interest should be normalized
to pick face colours, color bar range should always be specified
according to the normalization used. Default 0 to 1 range is used.
"""
            )
            norm = mpl.colors.Normalize(0.0, 1.0)
        else:
            norm = mpl.colors.Normalize(
                vmin=axis_spec.get("color_bar_range")[0],
                vmax=axis_spec.get("color_bar_range")[1],
            )

        cb1 = mpl.colorbar.ColorbarBase(
            cbar_axis, cmap=cmap, norm=norm, orientation="vertical"
        )
        if not axis_spec.get("color_bar_label"):
            cb1.set_label("a.u.")
        else:
            cb1.set_label(axis_spec.get("color_bar_label"))

        if legend is not None:
            handles = [
                mpatches.Patch(color=color, label=label)
                for label, color in legend.items()
            ]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1))

        return fig, ax

def sheet_view_3d(sheet, coords=COORDS, ax=None, view_angle=(30, 45), cull_back_edges=False, legend=None, draw_order=("face", "vert", "edge"), **draw_specs_kw):
    """3D version of sheet_view using Axes3D.

    Parameters
    ----------
    sheet : a tyssue Sheet object
    coords : list of 3 coordinate names, default COORDS
    ax : an Axes3D instance, or None to create a new one
    view_angle : tuple (elev, azim), default (30, 45)
    draw_order : tuple or list of {"face", "vert", "edge"}, default ("face", "vert", "edge")
        Order in which the elements are drawn. Elements drawn later appear on
        top. Any element omitted from this sequence is not drawn.
    **draw_specs_kw : passed to the draw spec updater
    """
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    valid_elements = {"face", "vert", "edge"}
    unknown = set(draw_order) - valid_elements
    if unknown:
        raise ValueError(
            f"Unknown element(s) in draw_order: {sorted(unknown)}. "
            f"Valid elements are {sorted(valid_elements)}."
        )

    for element in draw_order:
        spec = draw_specs[element]
        if not spec["visible"]:
            continue
        if element == "face":
            ax = draw_face_3d(sheet, coords, ax, **spec)
        elif element == "vert":
            ax = draw_vert_3d(sheet, coords, ax, **spec)
        elif element == "edge":
            ax = draw_edge_3d(sheet, coords, ax, view_angle=view_angle, cull_back_edges=cull_back_edges, **spec)

    if legend is not None:
        handles = [
            mpatches.Patch(color=color, label=label)
            for label, color in legend.items()
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1))

    ax.autoscale()
    _set_axes_proportional_3d(ax)
    _auto_tick_fontsize_3d(ax, base_size=8, min_size=4)
    return fig, ax

def draw_faces_highlighted(
    sheet,
    face_indices,
    highlight_color,
    coords=COORDS,
    ax=None,
    alpha=1.0,
    background_color=(1.0, 1.0, 1.0, 1.0),
    show_edges=True,
):
    """
    Draw full tissue, highlighting selected faces in a given color
    and rendering all others in white.

    Parameters
    ----------
    sheet : Sheet
    face_indices : array-like
        Indices of faces to highlight
    highlight_color : color-like
        Matplotlib color (e.g. "#ff0000", "red", RGBA)
    coords : tuple
    ax : matplotlib axis, optional
    alpha : float
    background_color : RGBA tuple
    show_edges : bool
    """

    from matplotlib.colors import to_rgba

    # Convert highlight color to RGBA
    hi_rgba = np.array(to_rgba(highlight_color))
    bg_rgba = np.array(background_color)

    # Build per-face RGBA array
    face_colors = np.tile(bg_rgba, (sheet.Nf, 1))

    face_idx = sheet.face_df.index
    mask = face_idx.isin(face_indices)

    face_colors[mask] = hi_rgba
    face_colors[:, 3] *= alpha  # apply alpha uniformly

    draw_specs = {
        "face": {
            "visible": True,
            "color": face_colors,
        },
        "edge": {
            "visible": show_edges,
        },
        "vert": {
            "visible": False,
        },
        "axis": {
            "autoscale": True,
            "color_bar": False,
        },
    }

    fig, ax = sheet_view(
        sheet,
        coords=coords,
        ax=ax,
        **draw_specs,
    )

    return fig, ax

def draw_face(sheet, coords, ax, **draw_spec_kw):
    """Draws epithelial sheet polygonal faces in matplotlib
    Keyword values can be specified at the element
    level as columns of the sheet.face_df
    """

    draw_spec = sheet_spec()["face"]
    draw_spec.update(**draw_spec_kw)
    collection_specs = parse_face_specs(draw_spec, sheet)

    if "visible" in sheet.face_df.columns:
        edges = sheet.edge_df[sheet.upcast_face(sheet.face_df["visible"])].index
        if edges.shape[0]:
            _sheet = get_sub_eptm(sheet, edges)
            sheet = _sheet
            color = collection_specs["facecolors"]
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                collection_specs["facecolors"] = color.take(faces, axis=0)
        else:
            warnings.warn("No face is visible")

    if not sheet.is_ordered:
        sheet_ = sheet.copy()
        sheet_.reset_index(order=True)
        polys = sheet_.face_polygons(coords)
    else:
        polys = sheet.face_polygons(coords)
    p = PolyCollection(polys, closed=True, **collection_specs)
    ax.add_collection(p)
    return ax


def draw_vert(sheet, coords, ax, **draw_spec_kw):
    """Draw junction vertices in matplotlib."""
    draw_spec = sheet_spec()["vert"]
    draw_spec.update(**draw_spec_kw)

    x, y = coords
    if "z_coord" in sheet.vert_df.columns:
        pos = sheet.vert_df.sort_values("z_coord")[coords]
    else:
        pos = sheet.vert_df[coords]
    ax.scatter(pos[x], pos[y], **draw_spec_kw)
    return ax


def draw_edge(sheet, coords, ax, **draw_spec_kw):
    """"""
    draw_spec = sheet_spec()["edge"]
    draw_spec.update(**draw_spec_kw)
    arrow_specs, collections_specs = _parse_edge_specs(draw_spec, sheet)
    dx, dy = ("d" + c for c in coords)
    sx, sy = ("s" + c for c in coords)
    tx, ty = ("t" + c for c in coords)

    if draw_spec.get("head_width"):

        app_length = (
            np.hypot(sheet.edge_df[dx], sheet.edge_df[dy]) * sheet.edge_df.length.mean()
        )
        patches = [
            FancyArrow(*edge[[sx, sy, dx, dy]], **arrow_specs)
            for idx, edge in sheet.edge_df[app_length > 1e-6].iterrows()
        ]
        ax.add_collection(
            PatchCollection(patches, match_original=False, **collections_specs)
        )
    else:
        segments = sheet.edge_df[[sx, sy, tx, ty]].to_numpy().reshape((-1, 2, 2))
        ax.add_collection(LineCollection(segments, **collections_specs))
    return ax


def draw_vert_3d(sheet, coords, ax, **draw_spec_kw):
    """Draw junction vertices in 3D matplotlib."""
    draw_spec = sheet_spec()["vert"]
    draw_spec.update(**draw_spec_kw)

    x, y, z = coords
    if "z_coord" in sheet.vert_df.columns:
        pos = sheet.vert_df.sort_values("z_coord")[coords]
    else:
        pos = sheet.vert_df[coords]

    ax.scatter(pos[x], pos[y], pos[z], **draw_spec_kw)
    ax.autoscale()
    return ax


def draw_edge_3d(sheet, coords, ax, view_angle=(30, 45), cull_back_edges=False, **draw_spec_kw):
    draw_spec = sheet_spec()["edge"]
    draw_spec.update(**draw_spec_kw)
    _, collections_specs = _parse_edge_specs(draw_spec, sheet)

    sx, sy, sz = ("s" + c for c in coords)
    tx, ty, tz = ("t" + c for c in coords)

    edge_df = sheet.edge_df

    if cull_back_edges:
        mx = (edge_df[sx] + edge_df[tx]) / 2
        my = (edge_df[sy] + edge_df[ty]) / 2

        # Only use azimuth — culling is purely in xy for a z-axis cylinder
        azim = np.deg2rad(view_angle[1])
        view_dir_xy = np.array([np.cos(azim), np.sin(azim)])

        # Centroid in xy only
        cx, cy = mx.mean(), my.mean()
        outward_xy = np.stack([mx - cx, my - cy], axis=1)

        dots = outward_xy @ view_dir_xy
        edge_df = edge_df[dots > 0]

    segments = (
        edge_df[[sx, sy, sz, tx, ty, tz]]
        .to_numpy()
        .reshape((-1, 2, 3))
    )
    ax.add_collection3d(Line3DCollection(segments, **collections_specs))
    return ax

def draw_face_3d(sheet, coords, ax, **draw_spec_kw):
    """Draw epithelial sheet polygonal faces as a Poly3DCollection."""
    draw_spec = sheet_spec()["face"]
    draw_spec.update(**draw_spec_kw)
    collection_specs = parse_face_specs(draw_spec, sheet)

    if "visible" in sheet.face_df.columns:
        edges = sheet.edge_df[sheet.upcast_face(sheet.face_df["visible"])].index
        if edges.shape[0]:
            _sheet = get_sub_eptm(sheet, edges)
            sheet = _sheet
            color = collection_specs["facecolors"]
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                collection_specs["facecolors"] = color.take(faces, axis=0)
        else:
            warnings.warn("No face is visible")

    if not sheet.is_ordered:
        sheet_ = sheet.copy()
        sheet_.reset_index(order=True)
        polys = sheet_.face_polygons(coords)
    else:
        polys = sheet.face_polygons(coords)

    p = Poly3DCollection(polys, closed=True, **collection_specs)
    ax.add_collection3d(p)
    return ax

def parse_face_specs(face_draw_specs, sheet):

    collection_specs = {}
    color = face_draw_specs.get("color")

    if callable(color):
        color = color(sheet)
        face_draw_specs["color"] = color

    if color is None:
        return {}
    elif isinstance(color, str):
        collection_specs["facecolors"] = color
    elif hasattr(color, "__len__"):
        collection_specs["facecolors"] = _face_color_from_sequence(
            face_draw_specs, sheet
        )
    if "alpha" in face_draw_specs:
        collection_specs["alpha"] = face_draw_specs["alpha"]

    return collection_specs


def _face_color_from_sequence(face_spec, sheet):
    color_ = face_spec["color"]
    cmap = colormaps[face_spec.get("colormap", "viridis")]
    color_min, color_max = face_spec.get("color_range", (color_.min(), color_.max()))

    if color_.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
        return color_

    elif color_.shape == (sheet.Nf,):
        if np.ptp(color_) < 1e-10:
            log.info("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Nf, 3)) * 0.5

        normed = (color_ - color_min) / (color_max - color_min)
        return cmap(normed)

    else:
        raise ValueError(
            "shape of `face_spec['color']` must be either (Nf, 3), (Nf, 4) or (Nf,)"
        )


def _parse_edge_specs(edge_draw_specs, sheet):

    arrow_keys = ["head_width", "length_includes_head", "shape"]
    arrow_specs = {
        key: val for key, val in edge_draw_specs.items() if key in arrow_keys
    }
    collection_specs = {}
    if arrow_specs.get("head_width"):  # draw arrows
        color_key = "edgecolors"
    else:
        color_key = "colors"

    if "color" in edge_draw_specs:
        if callable(edge_draw_specs["color"]):
            edge_draw_specs["color"] = edge_draw_specs["color"](sheet)

        if isinstance(edge_draw_specs["color"], str):
            collection_specs[color_key] = edge_draw_specs["color"]
        elif hasattr(edge_draw_specs["color"], "__len__"):
            collection_specs[color_key] = _wire_color_from_sequence(
                edge_draw_specs, sheet
            )

    if "width" in edge_draw_specs:
        collection_specs["linewidths"] = edge_draw_specs["width"]
    if "alpha" in edge_draw_specs:
        collection_specs["alpha"] = edge_draw_specs["alpha"]
    return arrow_specs, collection_specs


def _wire_color_from_sequence(edge_spec, sheet):
    """"""
    color_ = edge_spec["color"]

    color_min, color_max = edge_spec.get("color_range", (color_.min(), color_.max()))
    cmap = colormaps[edge_spec.get("colormap", "viridis")]
    if color_.shape in [(sheet.Nv, 3), (sheet.Nv, 4)]:
        return (sheet.upcast_srce(color_) + sheet.upcast_trgt(color_)) / 2
    elif color_.shape == (sheet.Nv,):
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Ne, 3)) * 0.7
        if not hasattr(color_, "index"):
            color_ = pd.Series(color_, index=sheet.vert_df.index)
        color_ = (sheet.upcast_srce(color_) + sheet.upcast_trgt(color_)) / 2
        return cmap((color_ - color_min) / (color_max - color_min))

    elif color_.shape in [(sheet.Ne, 3), (sheet.Ne, 4)]:
        return color_
    elif color_.shape == (sheet.Ne,):
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Nv, 3)) * 0.7
        return cmap((color_ - color_min) / (color_max - color_min))


def quick_edge_draw(sheet, coords=["x", "y"], ax=None, **draw_spec_kw):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    lines_x, lines_y = _get_lines(sheet, coords)
    ax.plot(lines_x, lines_y, **draw_spec_kw)
    ax.set_aspect("equal")
    return fig, ax


def _get_lines(sheet, coords):

    lines_x, lines_y = np.zeros(2 * sheet.Ne), np.zeros(2 * sheet.Ne)
    scoords = ["s" + c for c in coords]
    tcoords = ["t" + c for c in coords]
    if set(scoords + tcoords).issubset(sheet.edge_df.columns):
        srce_x, srce_y = sheet.edge_df[scoords].values.T
        trgt_x, trgt_y = sheet.edge_df[tcoords].values.T
    else:
        srce_x, srce_y = sheet.upcast_srce(sheet.vert_df[coords]).values.T
        trgt_x, trgt_y = sheet.upcast_trgt(sheet.vert_df[coords]).values.T

    lines_x[::2] = srce_x
    lines_x[1::2] = trgt_x
    lines_y[::2] = srce_y
    lines_y[1::2] = trgt_y
    # Trick from https://github.com/matplotlib/
    # matplotlib/blob/master/lib/matplotlib/tri/triplot.py#L65
    lines_x = np.insert(lines_x, slice(None, None, 2), np.nan)
    lines_y = np.insert(lines_y, slice(None, None, 2), np.nan)
    return lines_x, lines_y


def _set_axes_proportional_3d(ax):
    x_range = ax.get_xlim3d()[1] - ax.get_xlim3d()[0]
    y_range = ax.get_ylim3d()[1] - ax.get_ylim3d()[0]
    z_range = ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
    ax.set_box_aspect([x_range, y_range, z_range])

def _auto_tick_fontsize_3d(ax, base_size=8, min_size=4):
    ranges = np.array([
        ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
        ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
        ax.get_zlim3d()[1] - ax.get_zlim3d()[0],
    ])
    max_range = ranges.max()
    size = max(min_size, round(base_size * min(ranges) / max_range))
    for ax_obj in [ax.xaxis, ax.yaxis, ax.zaxis]:
        ax_obj.set_tick_params(labelsize=size)

def plot_forces(
    sheet, geom, model, coords, scaling, ax=None, approx_grad=None, **draw_specs_kw
):
    """Plot the net forces at each vertex, with their amplitudes multiplied
    by `scaling`. To be clear, this is the oposite of the gradient - grad E.
    """
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
    gcoords = ["g" + c for c in coords]
    if approx_grad is not None:
        app_grad = approx_grad(sheet, geom, model)
        grad_i = (
            pd.DataFrame(
                index=sheet.vert_df[sheet.vert_df.is_active.astype(bool)].index,
                data=app_grad.reshape((-1, len(sheet.coords))),
                columns=["g" + c for c in sheet.coords],
            )
            * scaling
        )
    else:
        grad_i = model.compute_gradient(sheet, components=False) * scaling
        grad_i = grad_i.loc[sheet.vert_df["is_active"].astype(bool)]
    sheet.vert_df[gcoords] = -grad_i[gcoords]  # F = -grad E

    if "extract" in draw_specs:
        sheet = sheet.extract_bounding_box(**draw_specs["extract"])

    if ax is None:
        fig, ax = quick_edge_draw(sheet, coords)
    else:
        fig = ax.get_figure()

    arrows = sheet.vert_df[coords + gcoords]
    for _, arrow in arrows.iterrows():
        ax.arrow(*arrow, **draw_specs["grad"])
    return fig, ax


def plot_scaled_energies(sheet, geom, model, scales, ax=None):
    """Plot scaled energies

    Parameters
    ----------
    sheet: a:class: Sheet object
    geom: a :class:`Geometry` class
    model: a :class:'Model'
    scales: np.linspace of float

    Returns
    -------
    fig: a :class:matplotlib.figure.Figure instance
    ax: :class:matplotlib.Axes instance, default None
    """

    from ..utils import scaled_unscaled

    def get_energies():
        energies = np.array([e.mean() for e in model.compute_energy(sheet, True)])

        return energies

    energies = np.array(
        [scaled_unscaled(get_energies, scale, sheet, geom) for scale in scales]
    )
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    ax.plot(scales, energies.sum(axis=1), "k-", lw=4, alpha=0.3, label="total")
    for e, label in zip(energies.T, model.labels):
        ax.plot(scales, e, label=label)
    ax.legend()
    return fig, ax


def get_arc_data(sheet):

    srce_pos = sheet.upcast_srce(sheet.vert_df[sheet.coords])
    trgt_pos = sheet.upcast_trgt(sheet.vert_df[sheet.coords])

    radius = 1 / sheet.edge_df["curvature"]

    e_x = sheet.edge_df["dx"] / sheet.edge_df["length"]
    e_y = sheet.edge_df["dy"] / sheet.edge_df["length"]

    center_x = (srce_pos.x + trgt_pos.x) / 2 - e_y * (radius - sheet.edge_df["sagitta"])

    center_y = (srce_pos.y + trgt_pos.y) / 2 - e_x * (radius - sheet.edge_df["sagitta"])

    alpha = sheet.edge_df["arc_chord_angle"]
    beta = sheet.edge_df["chord_orient"]

    # Ok, I admit a fair amount of trial and
    # error to get to the stuff below :-p
    rot = beta - np.sign(alpha) * np.pi / 2
    theta1 = (-alpha + rot) * np.sign(alpha)
    theta2 = (alpha + rot) * np.sign(alpha)

    center_data = pd.DataFrame.from_dict(
        {
            "radius": np.abs(radius),
            "x": center_x,
            "y": center_y,
            "theta1": theta1,
            "theta2": theta2,
        }
    )
    return center_data


def curved_view(sheet, radius_cutoff=1e3):

    center_data = get_arc_data(sheet)
    fig, ax = sheet_view(sheet, **{"edge": {"visible": False}})

    curves = []
    for idx, edge in center_data.iterrows():
        if edge["radius"] > radius_cutoff:
            st = sheet.edge_df.loc[idx, ["srce", "trgt"]]
            xy = sheet.vert_df.loc[st, sheet.coords]
            patch = PathPatch(Path(xy))
        else:
            patch = Arc(
                edge[["x", "y"]],
                2 * edge["radius"],
                2 * edge["radius"],
                theta1=edge["theta1"] * 180 / np.pi,
                theta2=edge["theta2"] * 180 / np.pi,
            )
        curves.append(patch)
    ax.add_collection(PatchCollection(curves, False, **{"facecolors": "none"}))
    ax.autoscale()
    return fig, ax


def plot_junction(eptm, edge_index, coords=["x", "y"]):
    """Plots local graph around a junction, for debugging purposes."""
    v10, v11 = eptm.edge_df.loc[edge_index, ["srce", "trgt"]]
    fig, ax = plt.subplots()
    ax.scatter(*eptm.vert_df.loc[[v10, v11], coords].values.T, marker="+", s=300)
    v10_out = set(eptm.edge_df[eptm.edge_df["srce"] == v10]["trgt"]) - {v11}
    v11_out = set(eptm.edge_df[eptm.edge_df["srce"] == v11]["trgt"]) - {v10}
    verts = v10_out.union(v11_out)

    ax.scatter(*eptm.vert_df.loc[v10_out, coords].values.T)
    ax.scatter(*eptm.vert_df.loc[v11_out, coords].values.T)
    x, y = coords
    for _, edge in eptm.edge_df.query(f"srce == {v10}").iterrows():
        ax.plot(
            edge[["s" + x, "t" + x]],
            edge[["s" + y, "t" + y]],
            lw=3,
            alpha=0.3,
            c="r",
        )

    for _, edge in eptm.edge_df.query(f"srce == {v11}").iterrows():
        ax.plot(
            edge[["s" + x, "t" + x]],
            edge[["s" + y, "t" + y]],
            "k--",
        )

    for v in verts:
        for _, edge in eptm.edge_df.query(f"srce == {v}").iterrows():
            if edge["trgt"] in {v10, v11}:
                continue
            ax.plot(
                edge[["s" + x, "t" + x]],
                edge[["s" + y, "t" + y]],
                "k",
                lw=0.4,
            )

    fig.set_size_inches(12, 12)
    return fig, ax
