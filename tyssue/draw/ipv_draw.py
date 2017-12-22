"""3D visualisation inside the notebook.
"""
import numpy as np
import pandas as pd
import warnings

from matplotlib import cm

import ipyvolume as ipv

from ..config.draw import sheet_spec

try:
    import ipyvolume as ipv
except ImportError:
    print(
    '''This module needs ipyvolume to work.
You can install it with:
$ conda install -c conda-forge ipyvolume
    ''')

def view_ipv(sheet, coords=['x', 'y', 'z'], **edge_specs):
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
        color = _color_from_sequence(spec, sheet)[:, :3]

    u, v, w = coords
    mesh = ipv.plot_trisurf(sheet.vert_df[u],
                            sheet.vert_df[v],
                            sheet.vert_df[w],
                            lines=sheet.edge_df[['srce', 'trgt']],
                            color=color)
    fig = ipv.gcf()
    box_size = max(*(sheet.vert_df[u].ptp()
                     for u in sheet.coords))
    border = 0.05 * box_size
    lim_inf = sheet.vert_df[sheet.coords].min().min() - border
    lim_sup = sheet.vert_df[sheet.coords].max().max() + border
    ipv.xyzlim(lim_inf, lim_sup)
    ipv.show()
    return fig, mesh


def _color_from_sequence(edge_spec, sheet):
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
