import numpy as np

from numpy.testing import assert_almost_equal
from copy import deepcopy

from tyssue.core.sheet import Sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.config.json_parser import load_default
from tyssue.stores import load_datasets
from tyssue.dynamics.sheet_isotropic_model import isotropic_relax


TOL = 1e-5
DECIMAL = 5


def test_adim():

    default_mod_specs = {
        "face": {
            "contractility": 0.12,
            "vol_elasticity": 1.,
            "prefered_height": 10.,
            "prefered_area": 24.,
            "prefered_vol": 240.,
            },
        "edge": {
            "line_tension": 0.04,
            },
        "vert": {
            "radial_tension": 0.,
            },
        "settings": {
            "grad_norm_factor": 1.,
            "nrj_norm_factor": 1.,
            }
        }
    new_mod_specs = deepcopy(default_mod_specs)
    dim_mod_specs = model.dimentionalize(new_mod_specs)
    new_mod_specs['edge']['line_tension'] = 0.
    assert new_mod_specs['edge']['line_tension'] == 0.
    assert default_mod_specs['edge']['line_tension'] == 0.04
    assert dim_mod_specs['edge']['line_tension'] == 0.04 * 1 * 24**1.5 * 10**2


def test_compute_energy():
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

    Et, Ec, Ev = model.compute_energy(sheet, full_output=True)
    assert_almost_equal(Et.mean(), 0.03314790, decimal=DECIMAL)
    assert_almost_equal(Ec.mean(), 0.21975665, decimal=DECIMAL)
    assert_almost_equal(Ev.mean(), 0.04593385, decimal=DECIMAL)

    energy = model.compute_energy(sheet, full_output=False)
    assert_almost_equal(energy, 18.583115963, decimal=DECIMAL)
    assert_almost_equal(energy/sheet.face_df.is_alive.sum(),
                        0.464, decimal=2)

def test_compute_gradient():
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

    grad_t, grad_c, grad_v_srce, grad_v_trgt = model.compute_gradient(sheet,
                                                                    components=True)
    grad_t_norm = np.linalg.norm(grad_t, axis=0).sum() / sheet.nrj_norm_factor
    assert_almost_equal(grad_t_norm, 0.647621287, decimal=DECIMAL)

    grad_c_norm = np.linalg.norm(grad_c, axis=0).sum() / sheet.nrj_norm_factor
    assert_almost_equal(grad_c_norm, 0.715576186, decimal=DECIMAL)

    grad_vs_norm = np.linalg.norm(grad_v_srce.dropna(), axis=0).sum() / sheet.nrj_norm_factor
    assert_almost_equal(grad_vs_norm, 0.436051688, decimal=DECIMAL)

    grad_vt_norm = np.linalg.norm(grad_v_trgt.dropna(), axis=0).sum() / sheet.nrj_norm_factor
    assert_almost_equal(grad_vt_norm, 0.399341306, decimal=DECIMAL)
