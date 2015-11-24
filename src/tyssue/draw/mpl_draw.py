"""
Matplotlib based plotting
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
import numpy as np


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

def sheet_view(sheet, coords=COORDS, **draw_specs_kw):
    """ Base view function, parametrizable
    through draw_secs
    """
    draw_specs = get_default_draw_specs()
    draw_specs.update(**draw_specs_kw)

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

def draw_cell(sheet, coords, ax, **draw_spec_kw):
    """Draws epithelial sheet polygonal cells in matplotlib
    """
    draw_spec = get_default_draw_specs()['cell']
    draw_spec.update(**draw_spec_kw)
    polys = sheet.cell_polygons(coords).groupby(level='cell')
    for _, poly in polys:
        patch = Polygon(poly,
                        fill=True,
                        closed=True,
                        **draw_spec)
        ax.add_patch(patch)
        return ax

def draw_jv(sheet, coords, ax, **draw_spec_kw):
    """Draw junction vertices in matplotlib
    """
    draw_spec = get_default_draw_specs()['jv']
    draw_spec.update(**draw_spec_kw)
    x, y = coords
    ax.scatter(sheet.jv_df[x], sheet.jv_df[y], **draw_spec_kw)
    return ax

def draw_je(sheet, coords, ax, **draw_spec_kw):
    """
    """
    draw_spec = get_default_draw_specs()['je']
    draw_spec.update(**draw_spec_kw)

    x, y = coords
    dx, dy = ('d'+c for c in coords)
    for e in sheet.je_idx:
        s, t, c = e
        ax.arrow(sheet.jv_df[x].loc[s], sheet.jv_df[y].loc[s],
                 sheet.je_df[dx].loc[e], sheet.je_df[dy].loc[e],
                 **draw_spec)
    return ax


def plot_forces(sheet, model,
                coords, scaling,
                ax=None,
                approx_grad=None,
                **draw_spec_kws):
    """Plot the net forces at each vertex, with their amplitudes multiplied
    by `scaling`
    """
    draw_specs = get_default_draw_specs()
    draw_specs.update(**draw_spec_kws)
    gcoords = ['g'+c for c in coords]
    if approx_grad is not None:
        app_grad = approx_grad(sheet, sheet.coords)
        grad_i = pd.DataFrame(index=sheet.jv_idx,
                              data=app_grad.reshape((-1, 3)),
                              columns=sheet.coords) * scaling

    else:
        grad_i = model.compute_gradient(sheet, components=False) * scaling

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


def plot_analytical_to_numeric_comp(sheet, model, geom,
                                    dim_mod_specs, mod_specs):
    import tyssue.dynamics.sheet_isotropic_model as iso

    fig, ax = plt.subplots(figsize=(8, 8))
    deltas = np.linspace(0.1, 1.8, 50)

    lbda = mod_specs['je']['line_tension'][0]
    gamma = mod_specs['cell']['contractility'][0]

    ax.plot(deltas, iso.isotropic_energy(deltas, mod_specs), 'k-',
            label='Analytical total')
    ax.plot(sheet.delta_o, iso.isotropic_energy(sheet.delta_o, mod_specs), 'ro')
    ax.plot(deltas, iso.elasticity(deltas), 'b-',
            label='Analytical volume elasticity')
    ax.plot(deltas, iso.contractility(deltas, gamma), color='orange', ls='-',
            label='Analytical contractility')
    ax.plot(deltas, iso.tension(deltas, lbda), 'g-',
            label='Analytical line tension')

    ax.set_xlabel(r'Isotropic scaling $\delta$')
    ax.set_ylabel(r'Isotropic energie $\bar E$')

    energies = iso.isotropic_energies(sheet, model, geom,
                                      deltas, dim_mod_specs)
    # energies = energies / norm
    ax.plot(deltas, energies[:, 2], 'bo:', lw=2, alpha=0.8,
            label='Computed volume elasticity')
    ax.plot(deltas, energies[:, 0], 'go:', lw=2, alpha=0.8,
            label='Computed line tension')
    ax.plot(deltas, energies[:, 1], ls=':',
            marker='o', color='orange', label='Computed contractility')
    ax.plot(deltas, energies.sum(axis=1), 'ko:', lw=2, alpha=0.8,
            label='Computed total')

    ax.legend(loc='upper left', fontsize=10)
    ax.set_ylim(0, 1.2)


    print(sheet.delta_o, deltas[energies.sum(axis=1).argmin()])

    return fig, ax
