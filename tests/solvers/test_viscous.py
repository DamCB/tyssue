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
