"""
Matplotlib based plotting
"""
import logging
import pathlib
import shutil
import subprocess
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

from ..config.draw import sheet_spec
from ..utils.utils import get_sub_eptm, spec_updater

COORDS = ["x", "y"]

log = logging.getLogger(__name__)


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
        except Exception as e:
            print(f"Droped frame {i}")
            print(e)
            continue

        if isinstance(ax, plt.Axes) and margin >= 0:
            ax.set(xlim=xlim, ylim=ylim)
        fig.savefig(graph_dir / f"movie_{i:04d}.png")
        plt.close(fig)

    try:
        subprocess.run(["convert", (graph_dir / "movie_*.png").as_posix(), output])
    except Exception as e:
        print(
            "Converting didn't work, make sure imagemagick is available on your system"
        )
        raise e

    finally:
        shutil.rmtree(graph_dir)


def sheet_view(sheet, coords=COORDS, ax=None, cbar_axis=None, **draw_specs_kw):
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
