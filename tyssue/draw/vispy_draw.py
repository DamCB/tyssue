import numpy as np
import pandas as pd

import vispy as vp
from vispy import app, scene
from ..config.draw import sheet_spec
from ..utils.utils import spec_updater


def face_visual(sheet, coords=None, **draw_specs_kw):

    if coords is None:
        coords = list("xyz")

    draw_specs = sheet_spec()["face"]
    draw_specs.update(draw_specs_kw)
    if "visible" in sheet.face_df.columns:
        vertices = np.concatenate(
            (sheet.face_df[coords], sheet.vert_df[coords]), axis=0
        )

        upcast_visible = sheet.upcast_face(sheet.face_df["visible"])
        visible_edges = sheet.edge_df[upcast_visible]
        # edge indices as (Nf + Nv) * 3 array
        faces = visible_edges[["srce", "trgt", "face"]].to_numpy()
        # The src, trgt, face triangle is correctly oriented
        # both vert_idx cols are shifted by Nf
        faces[:, :2] += sheet.Nf
    else:
        vertices, faces = sheet.triangular_mesh(coords)

    color = None

    if isinstance(draw_specs["color"], str):
        face_colors = None
        color = draw_specs["color"]

    colors = np.asarray(draw_specs["color"])
    if colors.shape == (3,):
        face_colors = pd.DataFrame(index=sheet.edge_df.index, columns=["R", "G", "B"])
        for channel, val in zip("RGB", colors):
            face_colors[channel] = val

    elif colors.shape == (4,):
        face_colors = pd.DataFrame(
            index=sheet.edge_df.index, columns=["R", "G", "B", "A"]
        )
        for channel, val in zip("RGBA", colors):
            face_colors[channel] = val

    elif colors.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
        face_colors = pd.DataFrame(
            index=sheet.face_df.index,
            data=colors,
            columns=["R", "G", "B", "A"][: colors.shape[1]],
        )
        face_colors = sheet.upcast_face(face_colors)

    elif colors.shape in [(sheet.Ne, 3), (sheet.Ne, 4)]:
        face_colors = pd.DataFrame(
            index=sheet.edge_df.index,
            data=colors,
            columns=["R", "G", "B", "A"][: colors.shape[1]],
        )

    mesh = scene.visuals.Mesh(
        vertices=vertices, faces=faces, face_colors=face_colors, color=color
    )
    return mesh


def edge_visual(sheet, coords=None, **draw_specs_kw):

    draw_specs = sheet_spec()["edge"]
    draw_specs.update(draw_specs_kw)
    if coords is None:
        coords = list("xyz")

    color = None
    if isinstance(draw_specs["color"], str):
        color = draw_specs["color"]

    else:
        colors = np.asarray(draw_specs["color"])
        if colors.shape == (3,):
            color = pd.DataFrame(
                index=sheet.vert_df.index, columns=["R", "G", "B", "A"]
            )
            for channel, val in zip("RGB", colors):
                color[channel] = val
            color["A"] = draw_specs.get("alpha", 1.0)

        elif colors.shape == (4,):
            color = pd.DataFrame(
                index=sheet.vert_df.index, columns=["R", "G", "B", "A"]
            )
            for channel, val in zip("RGBA", colors):
                color[channel] = val

        elif colors.shape == (sheet.Ne, 3):
            color = pd.DataFrame(
                index=sheet.edge_df.index, data=colors, columns=["R", "G", "B"]
            )
            color["A"] = draw_specs.get("alpha", 1.0)
            # Strangely, color spec is on a vertex, not segment, basis
            color["srce"] = sheet.edge_df["srce"]
            color = color.groupby("srce").mean()

        elif colors.shape == (sheet.Ne, 4):
            color = pd.DataFrame(
                index=sheet.edge_df.index, data=colors, columns=["R", "G", "B", "A"]
            )
            # Strangely, color spec is on a vertex, not segment, basis
            color["srce"] = sheet.edge_df["srce"]
            color = color.groupby("srce").mean()

        elif colors.shape == (sheet.Nv, 3):
            color = pd.DataFrame(
                index=sheet.vert_df.index, data=colors, columns=["R", "G", "B"]
            )
            color["A"] = draw_specs.get("alpha", 1.0)

        elif colors.shape == (sheet.Nv, 4):
            color = pd.DataFrame(
                index=sheet.vert_df.index, data=colors, columns=["R", "G", "B", "A"]
            )
        else:
            raise ValueError(
                """Shape of the color argument doesn't"""
                """ mach the number of edges """
            )

    wire_pos = sheet.vert_df[coords].values
    connect = sheet.edge_df[["srce", "trgt"]].values
    wire = scene.visuals.Line(
        pos=wire_pos, connect=connect, color=color, width=draw_specs["width"]
    )
    return wire


def vp_view(sheet, coords=None, interactive=True, **draw_specs_kw):

    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)

    if coords is None:
        coords = ["x", "y", "z"]
    canvas = scene.SceneCanvas(keys="interactive", show=True)
    view = canvas.central_widget.add_view()
    view.camera = "turntable"
    view.camera.aspect = 1
    view.bgcolor = vp.color.Color("#222222")

    if draw_specs["face"]["visible"]:
        mesh = face_visual(sheet, coords, **draw_specs["face"])
        view.add(mesh)

    if draw_specs["edge"]["visible"]:
        wire = edge_visual(sheet, coords, **draw_specs["edge"])
        view.add(wire)

    canvas.show()
    view.camera.set_range()
    if interactive:
        app.run()
    return canvas, view
