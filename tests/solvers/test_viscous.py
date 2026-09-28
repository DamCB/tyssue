from tyssue import History, Sheet, SheetGeometry, config
from tyssue.dynamics import PlanarModel
from tyssue.generation import three_faces_sheet
from tyssue.solvers.viscous import EulerSolver


def test_euler():
    geom = SheetGeometry
    model = PlanarModel
    sheet = Sheet("3", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.settings["threshold_length"] = 0.1

    sheet.update_specs(config.dynamics.quasistatic_plane_spec())
    sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
    history = History(sheet)
    solver = EulerSolver(sheet, geom, model, history=history, auto_reconnect=True)
    sheet.vert_df["viscosity"] = 0.1

    sheet.edge_df.loc[[0, 17], "line_tension"] *= 2
    sheet.edge_df.loc[[1], "line_tension"] *= 8
    l0 = sheet.edge_df.loc[0, "length"]
    _ = solver.solve(0.2, dt=0.05)
    assert sheet.edge_df.loc[0, "length"] < l0
    assert len(solver.history) == 5


def _counting_solver(idempotent=True):
    import numpy as np

    from tyssue.behaviors.event_manager import EventManager

    class CountingGeometry(SheetGeometry):
        calls = 0
        idempotent_update = idempotent

        @classmethod
        def update_all(cls, eptm):
            cls.calls += 1
            super().update_all(eptm)

    sheet = Sheet.planar_sheet_3d("flat", 6, 6, 1, 1)
    sheet.sanitize()
    CountingGeometry.update_all(sheet)
    sheet.vert_df["viscosity"] = 1.0
    solver = EulerSolver(sheet, CountingGeometry, PlanarModel, manager=EventManager())
    return solver, CountingGeometry, np


def _run_event(solver, geom, action, name="reconnect"):
    """Runs `action` as the only queued event, returns the update_all calls."""

    def event(eptm, manager):
        action(eptm)

    event.__name__ = name
    solver.manager.current.clear()
    solver.manager.current.append((event, {}))
    geom.calls = 0
    solver.execute_events()
    return geom.calls


def test_execute_events_detects_changes():
    from tyssue.topology.base_topology import collapse_edge
    from tyssue.topology.sheet_topology import split_vert

    solver, geom, np = _counting_solver()
    sheet = solver.eptm

    # nothing changed: the geometry from set_pos is still current
    assert _run_event(solver, geom, lambda eptm: None) == 0

    # a single vertex moved, without the event flagging it
    def nudge(eptm):
        eptm.vert_df.loc[eptm.vert_df.index[3], "x"] += 1e-9

    assert _run_event(solver, geom, nudge) == 1

    # two vertices merged
    def merge(eptm):
        collapse_edge(eptm, eptm.edge_df.index[eptm.edge_df["opposite"] >= 0][0])
        eptm.network_changed = False  # even if the event forgot to flag it

    nv = sheet.Nv
    assert _run_event(solver, geom, merge) == 1
    assert sheet.Nv == nv - 1

    # a vertex split
    def split(eptm):
        rank = np.bincount(eptm.edge_df["srce"], minlength=eptm.Nv)
        split_vert(eptm, eptm.vert_df.index[np.argmax(rank)])
        eptm.network_changed = False

    nv = sheet.Nv
    assert _run_event(solver, geom, split) == 1
    assert sheet.Nv == nv + 1

    # any other event keeps the unconditional update
    assert _run_event(solver, geom, lambda eptm: None, name="division") == 1


def test_execute_events_non_idempotent_geometry():
    solver, geom, _ = _counting_solver(idempotent=False)
    assert _run_event(solver, geom, lambda eptm: None) == 1
