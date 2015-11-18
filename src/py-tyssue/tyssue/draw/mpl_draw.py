"""
Matplotlib based plotting
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from ..utils.utils import update_default

default_draw_specs = {
    "je": {'visible': True,
           'width': 0.01,
           'head_width': 0.02,
           'length_includes_head': True,
           'shape': 'right',
           'color': '#2b5d0a',
           'alpha': 0.8,
           'zorder': 1},
    "jv": {'visible': True,
           's': 100,
           'c': '#000a4b',
           'alpha': 0.3,
           'zorder': 2},
    "cell": {'visible': False,
             'color':'#8aa678',
             'alpha': 1,
             'zorder': -1}
    }


COORDS = ['x', 'y']

def sheet_view(sheet, coords, draw_specs=None):
    """ Base view function, parametrizable
    through draw_secs
    """

    fig, ax = plt.subplots()
    try:
        dcoords = ['d'+c for c in coords]
        x, y = coords
        dx, dy = dcoords
    except (ValueError, TypeError):
        raise ValueError('The `coords` argument must be'
                         ' a pair of column names')

    update_default(default_draw_specs, draw_specs)
    jv_spec = draw_specs['jv']
    if jv_spec['visible']:
        ax = draw_jv(sheet, coords, ax, jv_spec)

    je_spec = draw_specs['je']
    if jv_spec['visible']:
        ax = draw_je(sheet, coords, ax, je_spec)

    cell_spec = draw_specs['cell']
    if cell_spec['visible']:
        ax = draw_cell(sheet, coords, ax, cell_spec)

    ax.set_aspect('equal')
    ax.grid()
    return fig, ax

def draw_cell(sheet, coords, ax, draw_spec):
    """Draws epithelial sheet polygonal cells in matplotlib
    """
    polys = sheet.cell_polygons(coords).groupby(level='cell')
    for _, poly in polys:
        patch = Polygon(poly,
                        fill=True,
                        closed=True,
                        **draw_spec)
        ax.add_patch(patch)
        return ax

def draw_jv(sheet, coords, ax, draw_spec):
    """Draw junction vertices in matplotlib
    """
    x, y = coords
    ax.scatter(sheet.jv_df[x], sheet.jv_df[y], **draw_spec)
    return ax

def draw_je(sheet, coords, ax, draw_spec):
    x, y = coords
    dx, dy = ('d'+c for c in coords)
    for e in sheet.je_idx:
        s, t, c = e
        ax.arrow(sheet.jv_df[x].loc[s], sheet.jv_df[y].loc[s],
                 sheet.je_df[dx].loc[e], sheet.je_df[dy].loc[e],
                 **draw_spec)
    return ax
