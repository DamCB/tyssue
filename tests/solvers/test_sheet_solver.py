import os
from tyssue.core.sheet import Sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue import config
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir
from tyssue.solvers.sheet_vertex_solver import Solver as solver


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
