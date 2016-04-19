from numpy.testing import assert_almost_equal

from tyssue.core.sheet import Sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.config.json_parser import load_default
from tyssue.stores import load_datasets
from tyssue.dynamics.sheet_isotropic_model import isotropic_relax
from tyssue.solvers.sheet_vertex_solver import Solver as solver


TOL = 1e-5
DECIMAL = 5

def test_solver():

    h5store = 'small_hexagonal.hf5'
    datasets = load_datasets(h5store,
                             data_names=['face', 'vert', 'edge'])
    sheet = Sheet('emin', datasets)
    sheet.set_geom('sheet')
    nondim_specs = load_default('dynamics', 'sheet')
    dim_model_specs = model.dimentionalize(nondim_specs)

    sheet.set_model('sheet', dim_model_specs)
    sheet.grad_norm_factor = sheet.specs['settings']['grad_norm_factor']
    sheet.nrj_norm_factor = sheet.specs['settings']['nrj_norm_factor']
    geom.update_all(sheet)
    isotropic_relax(sheet, nondim_specs)
    # sheet.vert_df.is_active = 1
    grad_err = solver.check_grad(sheet, geom, model)
    grad_err /= sheet.vert_df.size
    assert_almost_equal(grad_err, 0.0, DECIMAL)

    settings = {
        'minimize': {
            'options': {
                'disp': False,
                'ftol': 1e-4,
                'gtol': 1e-4},
            }
        }

    res = solver.find_energy_min(sheet, geom, model, **settings)
    assert res['success']
