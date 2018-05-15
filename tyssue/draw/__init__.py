from .plt_draw import quick_edge_draw
from .plt_draw import sheet_view as sheet_view_2d
from .ipv_draw import sheet_view as sheet_view_3d


def sheet_view(sheet, coords=['x', 'y', 'z'],
               ax=None, mode='2D', **draw_specs_kw):
    """


    Parameters
    ----------
    sheet: :class:`Epithelium` instance
    coords: list of strings,
     the coordinates over which to do the plot
    ax: :class:matplotlib.Axes instance, default None
     axis over which to plot the sheet, for quick and
    mode: str, {'2D'|'quick'|'3D'}, default '2D'
     the type of graph to plot (see bellow)

    Returns
    -------
    fig, {ax|meshes}:


    """

    if mode == '2D':
        return sheet_view_2d(sheet, coords, ax, **draw_specs_kw)
    elif mode == 'quick':
        edge_kw = draw_specs_kw.get('edge', {})
        return quick_edge_draw(sheet, coords, ax, **edge_kw)
    elif mode == '3D':
        return sheet_view_3d(sheet, coords, **draw_specs_kw)

    else :
        return ValueError("""
Argument `mode` not understood,
should be either '2D', '3D' or 'quick', got %s""", mode)
