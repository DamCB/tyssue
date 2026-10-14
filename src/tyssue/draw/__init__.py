import warnings

from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import matplotlib.collections as mcollections
import numpy as np

from .ipv_draw import browse_history  # noqa
from .ipv_draw import sheet_view as sheet_view_3d  # noqa
from .plt_draw import create_gif, create_gif_3d, plot_forces, quick_edge_draw  # noqa
from .plt_draw import sheet_view as sheet_view_2d  # noqa

# vispy is in the optional `viz` extra
try:
    from .vispy_draw import sheet_view as sheet_view_vispy  # noqa
except (ImportError, OSError):
    warnings.warn(
        "vispy is unavailable, so sheet_view(mode='vispy') is disabled. "
        "Install it with: pip install 'tyssue[viz]'",
        stacklevel=2,
    )
    sheet_view_vispy = None

def sheet_view(sheet, coords=["x", "y", "z"], ax=None, mode="2D", **draw_specs_kw):
    """Main plotting function in 2D or 3D.


    Parameters
    ----------
    sheet: :class:`Epithelium` instance
    coords: list of strings,
     the coordinates over which to do the plot
    ax: :class:matplotlib.Axes instance, default None
     axis over which to plot the sheet, for quick and
    mode: str, {'2D'|'quick'|'3D'|'vispy'}, default '2D'
     the type of graph to plot (see bellow)

    Returns
    -------
    fig, {ax|meshes}:


    """

    if mode == "2D":
        return sheet_view_2d(sheet, coords[:2], ax, **draw_specs_kw)
    if mode == "quick":
        edge_kw = draw_specs_kw.get("edge", {})
        return quick_edge_draw(sheet, coords[:2], ax, **edge_kw)
    if mode == "3D":
        return sheet_view_3d(sheet, coords, **draw_specs_kw)
    if mode == "vispy":
        if sheet_view_vispy is None:
            msg = (
                "sheet_view(mode='vispy') needs vispy. "
                "Install it with: pip install 'tyssue[viz]'"
            )
            raise RuntimeError(msg)
        return sheet_view_vispy(sheet, coords, **draw_specs_kw)
    return ValueError(
        """
Argument `mode` not understood,
should be either '2D', '3D', 'quick' or 'vispy', got %s""",
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
