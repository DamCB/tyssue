
import numpy as np
import pandas as pd

import vispy as vp
from vispy import app, scene
from ..config.json_parser import load_default
from ..utils.utils import spec_updater


def draw_tyssue(sheet, coords=None, **draw_specs_kw):

    draw_specs = load_default('draw', 'sheet')
    spec_updater(draw_specs, draw_specs_kw)

    if coords is None:
        coords = ['x', 'y', 'z']
    vertices, faces, _ = sheet.triangular_mesh(coords)
    canvas = scene.SceneCanvas(keys='interactive', show=True)
    grid = canvas.central_widget.add_grid()
    view = grid.add_view(0, 1)
    view.camera =  'turntable'
    view.camera.aspect = 1
    view.bgcolor = vp.color.Color('#aaaaaa')

    if draw_specs['face']['visible']:
        color = None
        if isinstance(draw_specs['face']['color'], str):
            face_colors = None
            color = draw_specs['face']['color']

        colors = np.asarray(draw_specs['face']['color'])
        if colors.shape == (3,):
            face_colors = pd.DataFrame(index=sheet.je_df.index,
                                       columns=['R', 'G', 'B'])
            for channel, val in zip('RGB', colors):
                face_colors[channel] = val

        elif colors.shape == (4,):
            face_colors = pd.DataFrame(index=sheet.je_df.index,
                                       columns=['R', 'G', 'B', 'A'])
            for channel, val in zip('RGBA', colors):
                face_colors[channel] = val

        elif colors.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
            face_colors = pd.DataFrame(index=sheet.face_df.index, data=colors,
                                       columns=['R', 'G', 'B', 'A'][:colors.shape[1]])
            face_colors = sheet.upcast_face(face_colors)

        elif colors.shape in [(3, sheet.Ne), (4, sheet.Ne)]:
            face_colors = pd.DataFrame(index=sheet.face_df.index, data=colors,
                                       columns=['R', 'G', 'B', 'A'][:colors.shape[1]])

        mesh = scene.visuals.Mesh(vertices=vertices,
                                  faces=faces,
                                  face_colors=face_colors,
                                  color=color)
        view.add(mesh)

    if draw_specs['je']['visible']:

        color = None
        if isinstance(draw_specs['je']['color'], str):
            color = draw_specs['je']['color']

        colors = np.asarray(draw_specs['je']['color'])
        if colors.shape == (3,):
            color = pd.DataFrame(index=sheet.je_df.index,
                                 columns=['R', 'G', 'B', 'A'])
            for channel, val in zip('RGB', colors):
                color[channel] = val
            color['A'] = 1.

        elif colors.shape == (4,):
            color = pd.DataFrame(index=sheet.je_df.index,
                                 columns=['R', 'G', 'B', 'A'])
            for channel, val in zip('RGBA', color):
                color[channel] = val

        elif colors.shape == (3, sheet.Ne):
            color = pd.DataFrame(index=sheet.je_df.index, data=colors,
                                 columns=['R', 'G', 'B'])
            color['A'] = 1.

        elif colors.shape == (4, sheet.Ne):
            color = pd.DataFrame(index=sheet.je_df.index, data=colors,
                                 columns=['R', 'G', 'B', 'A'])


        wire_pos = vertices[sheet.Nc:].copy()
        wire = vp.scene.visuals.Line(pos=wire_pos,
                                     connect=faces[:, :2] - sheet.Nc,
                                     color=color,
                                     width=draw_specs['je']['width'])
        view.add(wire)

    view.camera.set_range()
    canvas.show()
    app.run()
