import vispy as vp
#vp.use('ipynb_webgl')

from vispy import app, gloo, visuals
from vispy.geometry import MeshData
from vispy import plot


def draw_tyssue(eptm, positions, faces, **kwargs):

    mdata = MeshData(vertices=positions,
                     faces=faces)

    canvas = plot.mesh(meshdata=mdata, **kwargs)

    canvas.show()
    canvas.app.run()
