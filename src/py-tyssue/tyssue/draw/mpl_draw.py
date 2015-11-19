"""
Matplotlib based plotting
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd

from ..utils.utils import update_default

def get_default_draw_specs():
    default_draw_specs = {
        "je": {
            'visible': True,
            'width': 0.01,
            'head_width': 0.02,
            'length_includes_head': True,
            'shape': 'right',
            'color': '#2b5d0a',
            'alpha': 0.8,
            'zorder': 1
            },
        "jv": {
            'visible': True,
            's': 100,
            'c': '#000a4b',
            'alpha': 0.3,
            'zorder': 2,
            },
        'grad': {
            'color':'b',
            'alpha':0.5,
            'width':0.01,
            },
        "cell": {
            'visible': False,
            'color':'#8aa678',
            'alpha': 1,
            'zorder': -1,
            }
        }
    return default_draw_specs



COORDS = ['x', 'y']

def sheet_view(sheet, coords=COORDS, **draw_specs):
    """ Base view function, parametrizable
    through draw_secs
    """
    draw_specs.update(get_default_draw_specs())
    fig, ax = plt.subplots()
    jv_spec = draw_specs['jv']
    if jv_spec['visible']:
        ax = draw_jv(sheet, coords, ax, **jv_spec)

    je_spec = draw_specs['je']
    if je_spec['visible']:
        ax = draw_je(sheet, coords, ax, **je_spec)

    cell_spec = draw_specs['cell']
    if cell_spec['visible']:
        ax = draw_cell(sheet, coords, ax, **cell_spec)

    ax.set_aspect('equal')
    ax.grid()
    return fig, ax

def draw_cell(sheet, coords, ax, **draw_spec):
    """Draws epithelial sheet polygonal cells in matplotlib
    """
    draw_spec.update(get_default_draw_specs()['cell'])
    polys = sheet.cell_polygons(coords).groupby(level='cell')
    for _, poly in polys:
        patch = Polygon(poly,
                        fill=True,
                        closed=True,
                        **draw_spec)
        ax.add_patch(patch)
        return ax

def draw_jv(sheet, coords, ax, **draw_spec):
    """Draw junction vertices in matplotlib
    """
    draw_spec.update(get_default_draw_specs()['jv'])
    x, y = coords
    ax.scatter(sheet.jv_df[x], sheet.jv_df[y], **draw_spec)
    return ax

def draw_je(sheet, coords, ax, **draw_spec):
    draw_spec.update(get_default_draw_specs()['je'])

    x, y = coords
    dx, dy = ('d'+c for c in coords)
    for e in sheet.je_idx:
        s, t, c = e
        ax.arrow(sheet.jv_df[x].loc[s], sheet.jv_df[y].loc[s],
                 sheet.je_df[dx].loc[e], sheet.je_df[dy].loc[e],
                 **draw_spec)
    return ax


def plot_forces(sheet, model,
                coords,
                norm_factor,
                ax=None,
                approx_grad=None,
                **draw_specs):
    """Plot the net forces at each vertex, with their amplitudes divided
    by norm_factor
    """
    draw_specs.update(get_default_draw_specs())
    gcoords = ['g'+c for c in coords]
    if approx_grad is not None:
        app_grad = approx_grad(sheet, sheet.coords)
        grad_i = pd.DataFrame(index=sheet.jv_idx,
                              data=app_grad.reshape((-1, 3)),
                              columns=sheet.coords) / norm_factor

    else:
        grad_i = model.compute_gradient(sheet, components=False) / norm_factor

    arrows = pd.DataFrame(columns=coords + gcoords,
                          index=sheet.jv_df.index)
    arrows[coords] = sheet.jv_df[coords]
    arrows[gcoords] = - grad_i[coords] # F = -grad E

    if ax is None:
        fig, ax = sheet_view(sheet, coords, **draw_specs)
    else:
        fig = ax.get_figure()

    for _, arrow in arrows.iterrows():
        ax.arrow(*arrow,
                 **draw_specs['grad'])
    return fig, ax
