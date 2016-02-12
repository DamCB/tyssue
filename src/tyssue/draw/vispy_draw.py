
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
        mesh = scene.visuals.Mesh(vertices=vertices,
                                  faces=faces)
        view.add(mesh)

    if draw_specs['je']['visible']:

        wire_pos = vertices[sheet.Nc:].copy()
        wire = vp.scene.visuals.Line(pos=wire_pos,
                                     connect=faces[:, :2] - sheet.Nc,
                                     color=[0.1, 0.1, 0.3, 0.8],
                                     width=1)
        view.add(wire)

    view.camera.set_range()
    canvas.show()
    app.run()
