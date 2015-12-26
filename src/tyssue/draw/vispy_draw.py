
import vispy as vp
from vispy import app, gloo, visuals, scene
from vispy.geometry import MeshData

def draw_tyssue(eptm):

    vertices, faces, _ = eptm.triangular_mesh(['z', 'x', 'y'])

    canvas = scene.SceneCanvas(keys='interactive', show=True)

    grid = canvas.central_widget.add_grid()
    view = grid.add_view(0, 1)
    #view = canvas.central_widget.add_view()
    view.camera =  'turntable'
    view.camera.aspect = 1


    view.bgcolor = vp.color.Color('#aaaaaa')



    mesh = vp.scene.visuals.Mesh(vertices=vertices,
                                 faces=faces)

    wire_pos = vertices[eptm.Nc:].copy()


    wire = vp.scene.visuals.Line(pos=wire_pos,
                                 connect=faces[:, :2] - eptm.Nc,
                                 color=[0.1, 0.1, 0.3, 0.8],
                                 width=1)
    fcenters = vp.scene.visuals.Markers(
        pos=eptm.face_df[eptm.coords].values,
        face_color=[1, 1, 1])

    ccenters = vp.scene.visuals.Markers(
        pos=eptm.cell_df[eptm.coords].values,
        face_color=[1, 1, 1])

    view.add(mesh)
    view.add(wire)
    view.add(fcenters)
    canvas.show()

    app.run()
