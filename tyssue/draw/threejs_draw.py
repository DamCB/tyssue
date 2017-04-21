import numpy as np

import matplotlib.colors as mpl_colors
from ..config.draw import sheet_spec

try:
    import pythreejs as py3js
except ImportError:
    print('You need pythreejs to use this module'
          ' use conda install -c conda-forge pythreejs')


def highlight_cells(eptm, *cells, reset_visible=False):

    if reset_visible:
        eptm.face_df['visible'] = False

    for cell in cells:
        cell_faces = eptm.edge_df[eptm.edge_df['cell']==cell]['face']
        highlight_faces(eptm.face_df, cell_faces,
                        reset_visible=False)


def highlight_faces(face_df, faces,
                    reset_visible=False):
    """
    Sets the faces visibility to 1

    If `reset_visible` is `True`, sets all the other faces
    to `visible = False`
    """
    if ('visible' not in face_df.columns) or reset_visible:
        face_df['visible'] = False

    face_df.loc[faces, 'visible'] = True


def triangular_faces(sheet, coords, **draw_specs):
    spec = sheet_spec()
    spec.update(**draw_specs)
    if 'visible' in sheet.face_df.columns:
        vis_e = sheet.upcast_face(sheet.face_df['visible'])
        faces = sheet.edge_df[vis_e][['srce', 'trgt', 'face']]
    else:
        faces = sheet.edge_df[['srce', 'trgt', 'face']].copy()


    vertices = np.vstack([sheet.vert_df[coords],
                          sheet.face_df[coords]])
    vertices = vertices.reshape((-1, 3))
    faces['face'] += sheet.Nv

    facesgeom = py3js.PlainGeometry(vertices=[list(v) for v in vertices],
                                    faces=[list(f) for f in faces.values])

    return py3js.Mesh(geometry=facesgeom, material=py3js.LambertMaterial())


def edge_lines(sheet, coords, **draw_specs):

    spec = sheet_spec()
    spec.update(**draw_specs)

    up_srce = sheet.upcast_srce(sheet.vert_df[sheet.coords])
    up_trgt = sheet.upcast_trgt(sheet.vert_df[sheet.coords])

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
    return py3js.Line(geometry=linesgeom,
                      material=py3js.LineBasicMaterial(
                          linewidth=spec['edge']['width'],
                          vertexColors='VertexColors'),
                      type='LinePieces')


def view_3js(sheet, coords=['x', 'y', 'z'], **draw_specs):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    renderer: a :class:`pythreejs.pythreejs.Renderer` instance
    lines: a :class:`pythreejs.pythreejs.Line` object

    Example
    -------
    >>> from IPython import display
    >>> renderer, lines = view_3js(eptm)
    >>> display(renderer)
    """

    spec = sheet_spec()
    spec.update(**draw_specs)
    children = [py3js.DirectionalLight(color='#ccaabb',
                                       position=[0, 5, 0]),
                py3js.AmbientLight(color='#cccccc')]

    if spec['edge']['visible']:
        lines = edge_lines(sheet, coords, **spec)
        children.append(lines)
    if spec['face']['visible']:
        faces = triangular_faces(sheet, coords, **spec)
        children.append(faces)


    scene = py3js.Scene(children=children)
    cam = py3js.PerspectiveCamera(position=[0, 5, 5])
    renderer = py3js.Renderer(camera=cam,
                              scene=scene,
                              controls=[py3js.OrbitControls(controlling=cam)])
    return renderer, scene
