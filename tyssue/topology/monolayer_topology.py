import logging
import numpy as np

from ..geometry.bulk_geometry import MonolayerGeometry
from ..core.sheet import Sheet
from ..geometry.sheet_geometry import SheetGeometry
from .bulk_topology import get_division_vertices
from .bulk_topology import cell_division as bulk_division
from .sheet_topology import type1_transition as sheet_t1
from .sheet_topology import get_division_edges as sheet_division_edges


logger = logging.getLogger(name=__name__)


def cell_division(monolayer, mother, orientation="vertical", psi=0):
    """
    Divides the cell mother in the monolayer.

    Parameters
    ----------
    * monolayer: a :class:`Monolayer` instance
    * mother: int, the index of the cell to devide
    * orientation: str, {"vertical" | "horizontal"}
      if "horizontal", performs a division in the equatorial
      plane of the cell. If "vertical" (the default), performs
      a division along the basal-apical axis of the cell
    * psi: float, default 0
      extra rotation angle of the division plane
      around the basal-apical plane

    Returns
    -------
    * daughter: int, the index of the daughter cell
    """

    ab_axis = MonolayerGeometry.basal_apical_axis(monolayer, mother)
    plane_normal = np.asarray(ab_axis)

    if orientation == "horizontal":
        vertices = get_division_vertices(
            monolayer, mother=mother, plane_normal=plane_normal
        )
    elif orientation == "vertical":
        apical_sheet = monolayer.get_sub_sheet("apical")
        m_apical_face = monolayer.edge_df[
            (monolayer.edge_df["cell"] == mother)
            & (monolayer.edge_df["segment"] == "apical")
        ]["face"].iloc[0]
        apical_edges = sheet_division_edges(apical_sheet, m_apical_face, SheetGeometry)
        basal_edges = []
        for ae in apical_edges[::-1]:
            basal_edges.append(find_basal_edge(monolayer, ae))
        division_edges = list(apical_edges) + basal_edges
        vertices = get_division_vertices(monolayer, division_edges=division_edges)
    else:
        raise ValueError(
            """orientation argument not understood,
should be either "horizontal" or "vertical", not {}""".format(
                orientation
            )
        )

    daughter = bulk_division(monolayer, mother, MonolayerGeometry, vertices)

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
    b_trgt, = cell_edges[
        (srce_segment == "apical")
        & (trgt_segment == "basal")
        & (cell_edges["srce"] == srce)
    ]["trgt"]
    b_srce, = cell_edges[
        (srce_segment == "basal")
        & (trgt_segment == "apical")
        & (cell_edges["trgt"] == trgt)
    ]["srce"]
    b_edge, = cell_edges[
        (cell_edges["srce"] == b_srce) & (cell_edges["trgt"] == b_trgt)
    ].index
    return b_edge


def type1_transition(monolayer, apical_edge, epsilon=0.1):
    """Performs a type 1 transition on the apical and basal meshes
    """
    v0_a, v1_a, fb_a, cb_a = monolayer.edge_df.loc[
        apical_edge, ["srce", "trgt", "face", "cell"]
    ]
    basal_edge = find_basal_edge(monolayer, apical_edge)
    v0_b, v1_b, fb_b, cb_b = monolayer.edge_df.loc[
        basal_edge, ["srce", "trgt", "face", "cell"]
    ]
    if monolayer.face_df.loc[fb_a, "num_sides"] < 4:
        logger.warning(
            """Face %s has 3 sides,
type 1 transition is not allowed""",
            fb_a,
        )
        return
