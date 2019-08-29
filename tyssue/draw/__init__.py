import warnings
from .plt_draw import quick_edge_draw, plot_forces
from .plt_draw import sheet_view as sheet_view_2d
from .ipv_draw import browse_history

from .plt_draw import sheet_view as sheet_view_2d
from .ipv_draw import sheet_view as sheet_view_3d
from .plt_draw import create_gif


def sheet_view(sheet, coords=["x", "y", "z"], ax=None, mode="2D", **draw_specs_kw):
    """Main plotting function in 2D or 3D.


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

    if mode == "2D":
        return sheet_view_2d(sheet, coords[:2], ax, **draw_specs_kw)
    elif mode == "quick":
        edge_kw = draw_specs_kw.get("edge", {})
        return quick_edge_draw(sheet, coords[:2], ax, **edge_kw)
    elif mode == "3D":
        return sheet_view_3d(sheet, coords, **draw_specs_kw)

    return ValueError(
        """
Argument `mode` not understood,
should be either '2D', '3D' or 'quick', got %s""",
        mode,
    )


def highlight_cells(eptm, cells, reset_visible=False):
    """Sets a column 'visible' to True for all the faces of the
    cells passed as argument (for a 3D tyssue).

    If no such column exists in eptm.face_df, creates it.

    """
    if reset_visible:
        eptm.face_df["visible"] = False

    if not hasattr(cells, "__iter__"):
        cells = [cells]

    for cell in cells:
        cell_faces = eptm.edge_df[eptm.edge_df["cell"] == cell]["face"]
        highlight_faces(eptm.face_df, cell_faces, reset_visible=False)


def highlight_faces(face_df, faces, reset_visible=False):
    """
    Sets the faces visibility to 1

    If `reset_visible` is `True`, sets all the other faces
    to `visible = False`
    """
    if ("visible" not in face_df.columns) or reset_visible:
        face_df["visible"] = False

    face_df.loc[faces, "visible"] = True
