import logging
import numpy as np

from ..geometry.bulk_geometry import MonolayerGeometry
from ..core.sheet import Sheet
from ..geometry.sheet_geometry import SheetGeometry
from ..geometry.utils import rotation_matrix

from .bulk_topology import get_division_vertices
from .bulk_topology import cell_division as bulk_division
from .sheet_topology import type1_transition as sheet_t1
from .sheet_topology import get_division_edges as sheet_division_edges


logger = logging.getLogger(name=__name__)


def cell_division(monolayer, mother, orientation="vertical", psi=None):
    """
    Divides the cell mother in the monolayer.

    Parameters
    ----------
    * monolayer: a :class:`Monolayer` instance
    * mother: int, the index of the cell to devide
    * orientation: str, {"vertical" | "horizontal" | "apical"}
      if "horizontal", performs a division in the equatorial
      plane of the cell. If "vertical" (the default), performs
      a division along the basal-apical axis of the cell.
      If "apical", performs a division cutting the apical face
      perpendicularly to its principal axis
    * psi: float, default 0
      extra rotation angle of the division plane
      around the basal-apical plane

    Returns
    -------
    * daughter: int, the index of the daughter cell
    """

    ab_axis = MonolayerGeometry.basal_apical_axis(monolayer, mother)

    if orientation == "horizontal":
        plane_normal = np.asarray(ab_axis)
    elif orientation == "vertical":
        plane_normal = _vertical_plane_normal(ab_axis, psi=psi)
    elif orientation == "apical":
        rcoords = ['r'+c for c in monolayer.coords]
        apical_pos = monolayer.edge_df.loc[
            (monolayer.edge_df['cell'] == mother)
            & (monolayer.edge_df['segment'] == "apical"),
            rcoords
        ]
        _, _, vh = np.linalg.svd(apical_pos)
        plane_normal = vh[0, :]

    else:
        raise ValueError(
            f"""orientation argument not understood, should be either "horizontal",
"vertical" or "apical", not {orientation}"""
        )

    vertices, mother_verts, daughter_verts = get_division_vertices(
        monolayer, mother=mother, plane_normal=plane_normal, return_all=True
    )
    daughter = bulk_division(
        monolayer, mother, MonolayerGeometry, vertices, mother_verts, daughter_verts
    )

    # Correct segment assignations for the septum
    septum = monolayer.face_df.index[-2:]
    septum_edges = monolayer.edge_df.index[-2 * len(vertices) :]
    if orientation == "vertical":
        monolayer.face_df.loc[septum, "segment"] = "lateral"
        monolayer.edge_df.loc[septum_edges, "segment"] = "lateral"
        _assign_vert_segment(monolayer, vertices)

    elif orientation == "horizontal":
        monolayer.face_df.loc[septum[0], "segment"] = "apical"
        monolayer.face_df.loc[septum[1], "segment"] = "basal"
        monolayer.edge_df.loc[septum_edges[: len(vertices)], "segment"] = "apical"
        monolayer.edge_df.loc[septum_edges[len(vertices) :], "segment"] = "basal"
        monolayer.vert_df.loc[vertices, "segment"] = "apical"

    return daughter


def _vertical_plane_normal(ab_axis, psi=None):

    # Find the simplest vector perpendicular to the ab_axis
    perp_axis = np.array([-ab_axis[1], ab_axis[0], 0])

    if psi is None:
        psi = np.random.uniform(0, np.pi)

    # rotate of an arbitrary angle
    return np.dot(rotation_matrix(psi, ab_axis), perp_axis)


def _assign_vert_segment(monolayer, vertices):

    for v in vertices:
        segs = set(monolayer.edge_df[monolayer.edge_df["srce"] == v]["segment"])
        if "apical" in segs:
            monolayer.vert_df.loc[v, "segment"] = "apical"
        elif "basal" in segs:
            monolayer.vert_df.loc[v, "segment"] = "basal"
        else:
            monolayer.vert_df.loc[v, "segment"] = "lateral"


def find_basal_edge(monolayer, apical_edge):
    """Returns the basal edge parallel to the apical edge passed
    in argument.

    Parameters
    ----------
    monolayer: a :class:`Monolayer` instance

    """
    srce, trgt, cell = monolayer.edge_df.loc[apical_edge, ["srce", "trgt", "cell"]]
    cell_edges = monolayer.edge_df[monolayer.edge_df["cell"] == cell]
    srce_segment = monolayer.vert_df.loc[cell_edges["srce"].values, "segment"]
    srce_segment.index = cell_edges.index
    trgt_segment = monolayer.vert_df.loc[cell_edges["trgt"].values, "segment"]
    trgt_segment.index = cell_edges.index
    try:
        (b_trgt,) = cell_edges[
            (srce_segment == "apical")
            & (trgt_segment == "basal")
            & (cell_edges["srce"] == srce)
        ]["trgt"]
        (b_srce,) = cell_edges[
            (srce_segment == "basal")
            & (trgt_segment == "apical")
            & (cell_edges["trgt"] == trgt)
        ]["srce"]
        (b_edge,) = cell_edges[
            (cell_edges["srce"] == b_srce) & (cell_edges["trgt"] == b_trgt)
        ].index
    except ValueError:
        b_edge = None

    return b_edge
