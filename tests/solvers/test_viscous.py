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


def test_euler_withT1():

    geom = SheetGeometry
    model = PlanarModel
    sheet = Sheet("3", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.settings["threshold_length"] = 1e-2

    sheet.update_specs(config.dynamics.quasistatic_plane_spec())
    sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
    history = History(sheet)
    solver = EulerSolver(sheet, geom, model, with_t1=True, with_t3=True)
    sheet.was_changed = False

    def on_topo_change(sheet):
        sheet.was_changed = True
        sheet.edge_df["line_tension"] = sheet.specs["edge"]["line_tension"]

    sheet.vert_df["viscosity"] = 1.0
    sheet.vert_df.loc[1, "x"] -= 0.9

    sheet.edge_df.loc[[0, 17], "line_tension"] *= 4
    l0 = sheet.edge_df.loc[0, "length"]

    res = solver.solve(
        0.4, dt=0.05, on_topo_change=on_topo_change, topo_change_args=(solver.eptm,)
    )
    assert sheet.was_changed


def test_ivp_withT1():

    geom = SheetGeometry
    model = PlanarModel
    sheet = Sheet("3", *three_faces_sheet())
    geom.update_all(sheet)
    sheet.settings["threshold_length"] = 1e-2

    sheet.update_specs(config.dynamics.quasistatic_plane_spec())
    sheet.face_df["prefered_area"] = sheet.face_df["area"].mean()
    history = History(sheet)
    solver = IVPSolver(sheet, geom, model, with_t1=True, with_t3=True)
    sheet.was_changed = False

    def on_topo_change(sheet):
        sheet.was_changed = True
        sheet.edge_df["line_tension"] = sheet.specs["edge"]["line_tension"]

    sheet.vert_df["viscosity"] = 1.0
    sheet.vert_df.loc[1, "x"] -= 0.9

    sheet.edge_df.loc[[0, 17], "line_tension"] *= 4
    l0 = sheet.edge_df.loc[0, "length"]

    res = solver.solve(
        0.4, on_topo_change=on_topo_change, topo_change_args=(solver.eptm,)
    )
    assert sheet.was_changed
