import os

import numpy as np

from tyssue import config
from tyssue.core.sheet import Sheet
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.io.hdf5 import load_datasets
from tyssue.solvers import QSSolver
from tyssue.solvers.sheet_vertex_solver import Solver as solver
from tyssue.stores import stores_dir

TOL = 1e-5
DECIMAL = 5


def test_solver():

    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()

    sheet = Sheet("emin", datasets, specs)
    nondim_specs = config.dynamics.quasistatic_sheet_spec()
    dim_model_specs = model.dimensionalize(nondim_specs)

    sheet.update_specs(dim_model_specs)
    # sheet.vert_df.is_active = 1
    # grad_err = solver.check_grad(sheet, geom, model)
    # grad_err /= sheet.vert_df.size
    # assert_almost_equal(grad_err, 0.0, DECIMAL)

    settings = {"minimize": {"options": {"disp": False, "ftol": 1e-4, "gtol": 1e-4}}}

    res = solver.find_energy_min(sheet, geom, model, **settings)
    assert res["success"]

    # Restarting from the minimum just found, L-BFGS may stop at once with an
    # "ABNORMAL" line search: `success` says nothing about the physics there,
    # so check the energy instead.
    energy = model.compute_energy(sheet)
    new_solver = QSSolver(with_t1=True, with_t2=True)
    res = new_solver.find_energy_min(sheet, geom, model, **settings["minimize"])
    assert np.isfinite(res["fun"])
    assert res["fun"] <= energy + 1e-9


def test_qssolver_with_t3_alias():
    import warnings

    import pytest

    from tyssue.topology import auto_t2

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        new = QSSolver(with_t1=False, with_t2=True)
    with pytest.warns(FutureWarning, match="with_t2"):
        old = QSSolver(with_t1=False, with_t3=True)
    assert old.rearange and new.rearange
    assert old.set_pos.__code__ is new.set_pos.__code__
    assert new.set_pos.__code__ is auto_t2(lambda eptm, geom: None).__code__
    assert not QSSolver().rearange
