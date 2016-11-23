import numpy as np

import matplotlib.colors as mpl_colors
from ..config.draw import sheet_spec

try:
    import pythreejs as py3js
except ImportError:
    print('You need pythreejs to use this module'
          'use conda install -c conda-forge pythreejs')


def view_3js(sheet, **draw_specs):
    spec = sheet_spec()
    spec.update(**draw_specs)
    up_srce = sheet.upcast_srce(sheet.vert_df[['x', 'y', 'z']])
    up_trgt = sheet.upcast_trgt(sheet.vert_df[['x', 'y', 'z']])

    vertices = np.hstack([up_srce.values, up_trgt.values])
    vertices = vertices.reshape(vertices.shape[0]*2, 3)
    colors = spec['vert']['color']
    if isinstance(colors, str):
        colors = [colors for v in vertices]
    else:
        colors = np.asarray(colors)
        if (colors.shape == (sheet.Nv, 3)) or (colors.shape == (sheet.Nv, 4)):
            sheet.vert_df['hex_c'] = [mpl_colors.rgb2hex(c)
                                      for c in colors]
            srce_c = sheet.upcast_srce(sheet.vert_df['hex_c'])
            trgt_c = sheet.upcast_trgt(sheet.vert_df['hex_c'])
            colors = np.vstack([srce_c.values,
                                trgt_c.values]).T.reshape(vertices.shape[0])
            colors = list(colors)
        else:
            raise ValueError

    linesgeom = py3js.PlainGeometry(vertices=[list(v) for v in vertices],
                                    colors=colors)
    lines = py3js.Line(geometry=linesgeom,
                       material=py3js.LineBasicMaterial(
                           linewidth=spec['edge']['width'],
                           vertexColors='VertexColors'),
                       type='LinePieces')
    scene = py3js.Scene(
        children=[lines,
                  py3js.DirectionalLight(color='#ccaabb',
                                         position=[0, 5, 0]),
                  py3js.AmbientLight(color='#cccccc')])

    c = py3js.PerspectiveCamera(position=[0, 5, 5])
    renderer = py3js.Renderer(camera=c,
                              scene=scene,
                              controls=[py3js.OrbitControls(controlling=c)])
    return renderer, lines
