
import numpy as np
import pandas as pd

import vispy as vp
from vispy import app, scene
from ..config.json_parser import load_default
from ..utils.utils import spec_updater


def vp_view(sheet, coords=None, **draw_specs_kw):

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
            face_colors = pd.DataFrame(index=sheet.edge_df.index,
                                       columns=['R', 'G', 'B'])
            for channel, val in zip('RGB', colors):
                face_colors[channel] = val

        elif colors.shape == (4,):
            face_colors = pd.DataFrame(index=sheet.edge_df.index,
                                       columns=['R', 'G', 'B', 'A'])
            for channel, val in zip('RGBA', colors):
                face_colors[channel] = val

        elif colors.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
            face_colors = pd.DataFrame(index=sheet.face_df.index, data=colors,
                                       columns=['R', 'G', 'B', 'A'][:colors.shape[1]])
            face_colors = sheet.upcast_face(face_colors)

        elif colors.shape in [(sheet.Ne, 3), (sheet.Ne, 4)]:
            face_colors = pd.DataFrame(index=sheet.face_df.index, data=colors,
                                       columns=['R', 'G', 'B', 'A'][:colors.shape[1]])

        mesh = scene.visuals.Mesh(vertices=vertices,
                                  faces=faces,
                                  face_colors=face_colors,
                                  color=color)
        view.add(mesh)

    if draw_specs['edge']['visible']:

        color = None
        if isinstance(draw_specs['edge']['color'], str):
            color = draw_specs['edge']['color']

        else:
            colors = np.asarray(draw_specs['edge']['color'])
            if colors.shape == (3,):
                color = pd.DataFrame(index=sheet.vert_df.index,
                                     columns=['R', 'G', 'B', 'A'])
                for channel, val in zip('RGB', colors):
                    color[channel] = val
                color['A'] = draw_specs['edge'].get('alpha', 1.)

            elif colors.shape == (4,):
                color = pd.DataFrame(index=sheet.vert_df.index,
                                     columns=['R', 'G', 'B', 'A'])
                for channel, val in zip('RGBA', colors):
                    color[channel] = val

            elif colors.shape == (sheet.Ne, 3):
                color = pd.DataFrame(index=sheet.edge_df.index, data=colors,
                                     columns=['R', 'G', 'B'])
                color['A'] = draw_specs['edge'].get('alpha', 1.)
                # Strangely, color spec is on a vertex, not segment, basis
                color['srce'] = sheet.edge_df['srce']
                color = color.groupby('srce').mean()

            elif colors.shape == (sheet.Ne, 4):
                color = pd.DataFrame(index=sheet.edge_df.index, data=colors,
                                     columns=['R', 'G', 'B', 'A'])
                # Strangely, color spec is on a vertex, not segment, basis
                color['srce'] = sheet.edge_df['srce']
                color = color.groupby('srce').mean()

            elif colors.shape == (sheet.Nv, 3):
                color = pd.DataFrame(index=sheet.vert_df.index, data=colors,
                                     columns=['R', 'G', 'B'])
                color['A'] = draw_specs['edge'].get('alpha', 1.)

            elif colors.shape == (sheet.Nv, 4):
                color = pd.DataFrame(index=sheet.vert_df.index, data=colors,
                                     columns=['R', 'G', 'B', 'A'])


            else:
                raise ValueError('''Shape of the color argument doesn't'''
                                 ''' mach the number of edges ''')



        wire_pos = vertices[sheet.Nc:].copy()
        wire = vp.scene.visuals.Line(pos=wire_pos,
                                     connect=faces[:, :2] - sheet.Nc,
                                     color=color,
                                     width=draw_specs['edge']['width'])
        view.add(wire)

    canvas.show()
    view.camera.set_range()
    app.run()
