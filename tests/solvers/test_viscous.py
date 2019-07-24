import numpy as np
import pandas as pd

from tyssue import config, Sheet, SheetGeometry, History
from tyssue.generation import three_faces_sheet
from tyssue.dynamics import PlanarModel
from tyssue.solvers.viscous import EulerSolver, IVPSolver


def test_euler():
    geom = SheetGeometry
    model = PlanarModel
    sheet = Sheet("3", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.settings["threshold_length"] = 1e-3

    sheet.update_specs(config.dynamics.quasistatic_plane_spec())
    sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
    history = History(sheet)
    solver = EulerSolver(sheet, geom, model, with_t1=True, with_t3=True)

    sheet.vert_df["viscosity"] = 1.0
    sheet.edge_df.loc[[0, 17], "line_tension"] *= 4
    l0 = sheet.edge_df.loc[0, "length"]
    res = solver.solve(0.2, dt=0.05)
    assert sheet.edge_df.loc[0, "length"] < l0
    assert len(solver.history) == 5
