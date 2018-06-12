"""3D visualisation inside the notebook.
"""
import warnings
import numpy as np
import pandas as pd
from matplotlib import cm

from ..config.draw import sheet_spec
from ..utils.utils import spec_updater

try:
    import ipyvolume as ipv
except ImportError:
    print('''
This module needs ipyvolume to work.
You can install it with:
$ conda install -c conda-forge ipyvolume
''')


def sheet_view(sheet, coords=['x', 'y', 'z'], **draw_specs_kw):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    fig: a :class:`ipyvolume.widgets.Figure` widget
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    draw_specs = sheet_spec()
    spec_updater(draw_specs, draw_specs_kw)
    edge_spec = draw_specs['edge']
    if edge_spec['visible']:
        edge_mesh = draw_edge(sheet, coords, **edge_spec)
    else:
        edge_mesh = None

    face_spec = draw_specs['face']
    if face_spec['visible']:
        face_mesh = draw_face(sheet, coords, **face_spec)
    else:
        face_mesh = None

    box_size = max(*(sheet.vert_df[u].ptp()
                     for u in sheet.coords))
    border = 0.05 * box_size
    lim_inf = sheet.vert_df[sheet.coords].min().min() - border
    lim_sup = sheet.vert_df[sheet.coords].max().max() + border
    ipv.xyzlim(lim_inf, lim_sup)
    fig = ipv.gcf()
    return fig, (edge_mesh, face_mesh)


def view_ipv(sheet, coords=['x', 'y', 'z'], **edge_specs):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    fig: a :class:`ipyvolume.widgets.Figure` widget
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    warnings.warn('`view_ipv` is deprecated, use the more generic `sheet_view`')
    mesh = draw_edge(sheet, coords, **edge_specs)
    box_size = max(*(sheet.vert_df[u].ptp()
                     for u in sheet.coords))
    border = 0.05 * box_size
    lim_inf = sheet.vert_df[sheet.coords].min().min() - border
    lim_sup = sheet.vert_df[sheet.coords].max().max() + border
    ipv.xyzlim(lim_inf, lim_sup)
    fig = ipv.gcf()
    return fig, mesh


def draw_edge(sheet, coords, **edge_specs):
    """
    Creates a javascript renderer of the edge lines to be displayed
    in Jupyter Notebooks

    Returns
    -------

    fig: a :class:`ipyvolume.widgets.Figure` widget
    mesh: a :class:`ipyvolume.widgets.Mesh` mesh widget

    """
    ipv.style.use(['dark', 'minimal'])
    spec = sheet_spec()['edge']
    spec.update(**edge_specs)
    if isinstance(spec['color'], str):
        color = spec['color']
    elif hasattr(spec['color'], '__len__'):
        color = _wire_color_from_sequence(spec, sheet)[:, :3]

    u, v, w = coords
    mesh = ipv.plot_trisurf(sheet.vert_df[u],
                            sheet.vert_df[v],
                            sheet.vert_df[w],
                            lines=sheet.edge_df[['srce', 'trgt']],
                            color=color)
    return mesh


def draw_face(sheet, coords, **face_draw_specs):

    epsilon = face_draw_specs.get('epsilon', 0)
    up_srce = sheet.upcast_srce(sheet.vert_df[coords])
    up_trgt = sheet.upcast_trgt(sheet.vert_df[coords])

    if epsilon > 0:
        up_face = sheet.upcast_face(sheet.face_df[coords])
        up_srce = (up_srce - up_face) * (1 - epsilon) + up_face
        up_trgt = (up_trgt - up_face) * (1 - epsilon) + up_face

    mesh = np.concatenate([sheet.face_df[coords].values,
                           up_srce.values, up_trgt.values])

    Ne, Nf = sheet.Ne, sheet.Nf
    triangles = np.vstack([sheet.edge_df['face'],
                           np.arange(Ne)+Nf,
                           np.arange(Ne)+Ne+Nf]).T

    color = _face_color_from_sequence(face_draw_specs, sheet, )
    mesh = ipv.plot_trisurf(mesh[:, 0], mesh[:, 1], mesh[:, 2],
                            triangles=triangles, color=color[:, :3])
    return mesh



def _wire_color_from_sequence(edge_spec, sheet):
    """
    """
    color_ = edge_spec['color']
    cmap = cm.get_cmap(edge_spec.get('colormap', 'viridis'))
    if color_.shape in [(sheet.Nv, 3), (sheet.Nv, 4)]:
        return np.asarray(color_)
    elif color_.shape == (sheet.Nv,):
        if color_.ptp() < 1e-10:
            warnings.warn('Attempting to draw a colormap '
                          'with a uniform value')
            return np.ones((sheet.Nv, 3))*0.7
        return cmap((color_ - color_.min())/color_.ptp())

    elif color_.shape in [(sheet.Ne, 3), (sheet.Ne, 4)]:
        color_ = pd.DataFrame(color_.values,
                              index=sheet.edge_df.index)
        color_['srce'] = sheet.edge_df['srce']
        color_ = color_.groupby('srce').mean().values
        return color_
    elif color_.shape == (sheet.Ne,):
        color_ = pd.DataFrame(color_.values,
                              index=sheet.edge_df.index)
        color_['srce'] = sheet.edge_df['srce']
        color_ = color_.groupby('srce').mean().values.ravel()
        if color_.ptp() < 1e-10:
            warnings.warn('Attempting to draw a colormap '
                          'with a uniform value')
            return np.ones((sheet.Nv, 3))*0.7
        return cmap((color_ - color_.min())/color_.ptp())



def _face_color_from_sequence(face_spec, sheet):
    color_ = face_spec['color']
    cmap = cm.get_cmap(face_spec.get('colormap', 'viridis'))
    Nf, Ne = sheet.Nf, sheet.Ne
    color_min, color_max = face_spec.get(
        'color_range', (color_.min(), color_.max()))

    face_mesh_shape = Nf + 2*Ne

    if color_.shape in [(sheet.Nf, 3), (sheet.Nf, 4)]:
        return np.concatenate([color_, color_, color_])

    elif color_.shape == (sheet.Nf,):
        if color_.ptp() < 1e-10:
            warnings.warn('Attempting to draw a colormap '
                          'with a uniform value')
            return np.ones((face_mesh_shape, 3))*0.5

        normed = (color_ - color_min)/(color_max - color_min)
        up_color = sheet.upcast_face(normed).values
        return cmap(np.concatenate([normed, up_color, up_color]))

    else:
        warnings.warn("shape of `face_spec['color']` must be either (Nf, 3), (Nf, 4) or (Nf,)")
        return face_spec["color"]
