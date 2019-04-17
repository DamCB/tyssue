"""
Matplotlib based plotting
"""
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import Polygon, FancyArrow, Arc, PathPatch
from matplotlib.collections import PatchCollection, PolyCollection
from ..config.draw import sheet_spec
from ..utils.utils import spec_updater, get_sub_eptm

COORDS = ["x", "y"]


def sheet_view(sheet, coords=COORDS, ax=None, **draw_specs_kw):
    """ Base view function, parametrizable
    through draw_secs

    The default sheet_spec specification is:

    {'edge': {
      'visible': True,
      'width': 0.5,
      'head_width': 0.2, # arrow head width for the edges
      'length_includes_head': True, # see matplotlib Arrow artist doc
      'shape': 'right',
      'color': '#2b5d0a', # can be an array
      'alpha': 0.8,
      'zorder': 1,
      'colormap': 'viridis'},
     'vert': {
      'visible': True,
      's': 100,
      'color': '#000a4b',
      'alpha': 0.3,
      'zorder': 2},
     'face': {
      'visible': False,
      'color': '#8aa678',
      'alpha': 1.0,
      'zorder': -1}
      }
    """
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
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

    ax.autoscale()
    ax.set_aspect("equal")
    ax.grid()
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
        _sheet = get_sub_eptm(sheet, edges)
        if _sheet is not None:
            sheet = _sheet
            color = collection_specs["facecolors"]
            if isinstance(color, np.ndarray):
                faces = sheet.face_df["face_o"].values.astype(np.uint32)
                collection_specs["facecolors"] = color.take(faces, axis=0)

    polys = sheet.face_polygons(coords)
    p = PolyCollection(polys, closed=True, **collection_specs)
    ax.add_collection(p)
    return ax


def parse_face_specs(face_draw_specs, sheet):

    collection_specs = {}
    color = face_draw_specs.get("color")
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
    cmap = cm.get_cmap(face_spec.get("colormap", "viridis"))
    color_min, color_max = face_spec.get("color_range", (color_.min(), color_.max()))

    if color_.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
        return color_

    elif color_.shape == (sheet.Nf,):
        if np.ptp(color_) < 1e-10:
            warnings.warn("Attempting to draw a colormap " "with a uniform value")
            return np.ones((sheet.Nf, 3)) * 0.5

        normed = (color_ - color_min) / (color_max - color_min)
        return cmap(normed)

    else:
        raise ValueError(
            "shape of `face_spec['color']` must be either (Nf, 3), (Nf, 4) or (Nf,)"
        )


def draw_vert(sheet, coords, ax, **draw_spec_kw):
    """Draw junction vertices in matplotlib
    """
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
    """
    """
    draw_spec = sheet_spec()["edge"]
    draw_spec.update(**draw_spec_kw)

    x, y = coords
    dx, dy = ("d" + c for c in coords)
    app_length = np.hypot(sheet.edge_df[dx], sheet.edge_df[dy])

    patches = []
    arrow_specs, collections_specs = parse_edge_specs(draw_spec, sheet)

    for idx, edge in sheet.edge_df[app_length > 1e-6].iterrows():
        srce = int(edge["srce"])
        arrow = FancyArrow(
            sheet.vert_df.loc[srce, x],
            sheet.vert_df.loc[srce, y],
            sheet.edge_df.loc[idx, dx],
            sheet.edge_df.loc[idx, dy],
            **arrow_specs
        )
        patches.append(arrow)
    ax.add_collection(PatchCollection(patches, False, **collections_specs))
    return ax


def parse_edge_specs(edge_draw_specs, sheet):

    arrow_keys = ["head_width", "length_includes_head", "shape"]
    arrow_specs = {
        key: val for key, val in edge_draw_specs.items() if key in arrow_keys
    }
    collection_specs = {}
    if "color" in edge_draw_specs:
        if isinstance(edge_draw_specs["color"], str):
            collection_specs["edgecolors"] = edge_draw_specs["color"]
        elif hasattr(edge_draw_specs["color"], "__len__"):
            collection_specs["edgecolors"] = _wire_color_from_sequence(
                edge_draw_specs, sheet
            )

    if "width" in edge_draw_specs:
        collection_specs["linewidths"] = edge_draw_specs["width"]
    if "alpha" in edge_draw_specs:
        collection_specs["alpha"] = edge_draw_specs["alpha"]
    return arrow_specs, collection_specs


def _wire_color_from_sequence(edge_spec, sheet):
    """
    """
    color_ = edge_spec["color"]
    color_min, color_max = edge_spec.get("color_range", (color_.min(), color_.max()))
    cmap = cm.get_cmap(edge_spec.get("colormap", "viridis"))
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

    x, y = coords
    srce_x = sheet.upcast_srce(sheet.vert_df[x]).values
    srce_y = sheet.upcast_srce(sheet.vert_df[y]).values
    trgt_x = sheet.upcast_trgt(sheet.vert_df[x]).values
    trgt_y = sheet.upcast_trgt(sheet.vert_df[y]).values

    lines_x, lines_y = np.zeros(2 * sheet.Ne), np.zeros(2 * sheet.Ne)
    lines_x[::2] = srce_x
    lines_x[1::2] = trgt_x
    lines_y[::2] = srce_y
    lines_y[1::2] = trgt_y
    # Trick from https://github.com/matplotlib/
    # matplotlib/blob/master/lib/matplotlib/tri/triplot.py#L65
    lines_x = np.insert(lines_x, slice(None, None, 2), np.nan)
    lines_y = np.insert(lines_y, slice(None, None, 2), np.nan)
    ax.plot(lines_x, lines_y, **draw_spec_kw)
    ax.set_aspect("equal")
    return fig, ax


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
                index=sheet.active_verts,
                data=app_grad.reshape((-1, len(sheet.coords))),
                columns=["g" + c for c in sheet.coords],
            )
            * scaling
        )

    else:
        grad_i = model.compute_gradient(sheet, components=False) * scaling

    arrows = pd.DataFrame(columns=coords + gcoords, index=sheet.vert_df.index)
    arrows[coords] = sheet.vert_df[coords]
    arrows[gcoords] = -grad_i[gcoords]  # F = -grad E

    if ax is None:
        fig, ax = quick_edge_draw(sheet, coords)
    else:
        fig = ax.get_figure()

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


def plot_analytical_to_numeric_comp(sheet, model, geom, isotropic_model, nondim_specs):

    iso = isotropic_model
    fig, ax = plt.subplots(figsize=(8, 8))

    deltas = np.linspace(0.1, 1.8, 50)

    lbda = nondim_specs["edge"]["line_tension"]
    gamma = nondim_specs["face"]["contractility"]

    ax.plot(
        deltas,
        iso.isotropic_energy(deltas, nondim_specs),
        "k-",
        label="Analytical total",
    )
    try:
        ax.plot(sheet.delta_o, iso.isotropic_energy(sheet.delta_o, nondim_specs), "ro")
    except:
        pass
    ax.plot(deltas, iso.elasticity(deltas), "b-", label="Analytical volume elasticity")
    ax.plot(
        deltas,
        iso.contractility(deltas, gamma),
        color="orange",
        ls="-",
        label="Analytical contractility",
    )
    ax.plot(deltas, iso.tension(deltas, lbda), "g-", label="Analytical line tension")

    ax.set_xlabel(r"Isotropic scaling $\delta$")
    ax.set_ylabel(r"Isotropic energie $\bar E$")

    energies = iso.isotropic_energies(sheet, model, geom, deltas, nondim_specs)
    # energies = energies / norm
    ax.plot(
        deltas,
        energies[:, 2],
        "bo:",
        lw=2,
        alpha=0.8,
        label="Computed volume elasticity",
    )
    ax.plot(
        deltas, energies[:, 0], "go:", lw=2, alpha=0.8, label="Computed line tension"
    )
    ax.plot(
        deltas,
        energies[:, 1],
        ls=":",
        marker="o",
        color="orange",
        label="Computed contractility",
    )
    ax.plot(
        deltas, energies.sum(axis=1), "ko:", lw=2, alpha=0.8, label="Computed total"
    )

    ax.legend(loc="upper left", fontsize=10)
    ax.set_ylim(0, 1.2)
    print(sheet.delta_o, deltas[energies.sum(axis=1).argmin()])

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
