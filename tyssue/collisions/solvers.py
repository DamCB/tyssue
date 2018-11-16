import numpy as np
import logging

from tyssue.solvers.sheet_vertex_solver import Solver
from . import self_intersections


log = logging.getLogger(__name__)


class CollisionSolver(Solver):
    @classmethod
    def opt_energy(cls, pos, pos_idx, sheet, geom, model):
        # Keep old position safe
        position_buffer = sheet.vert_df[sheet.coords].copy()

        cls.set_pos(pos, pos_idx, sheet)
        geom.update_all(sheet)

        intersecting_edges = self_intersections(sheet)
        if intersecting_edges.shape[0]:
            log.info("%d intersections where detected", intersecting_edges.shape[0])
            revert_positions(sheet, intersecting_edges, position_buffer)

        geom.update_all(sheet)
        return model.compute_energy(sheet, full_output=False)


def revert_positions(sheet, intersecting_edges, position_buffer):

    unique_edges = np.unique(intersecting_edges)
    unique_verts = np.unique(sheet.edge_df.loc[unique_edges, ["srce", "trgt"]])
    sheet.vert_df.loc[unique_verts, sheet.coords] = position_buffer.loc[unique_verts]
