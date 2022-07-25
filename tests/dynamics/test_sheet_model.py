import os
from copy import deepcopy

import numpy as np
from numpy.testing import assert_almost_equal

from tyssue import config
from tyssue.core.sheet import Sheet
from tyssue.dynamics.planar_vertex_model import PlanarModel
from tyssue.dynamics.sheet_vertex_model import SheetModel as model
from tyssue.geometry.planar_geometry import PlanarGeometry
from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir
from tyssue.utils.testing import model_tester

TOL = 1e-5
DECIMAL = 5


def test_model():

    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet("emin", datasets, specs)
    model_tester(sheet, model)
    model_tester(sheet, PlanarModel)
    flat = Sheet.planar_sheet_2d("flat", 5, 5, 1, 1)
    flat.sanitize()
    PlanarGeometry.update_all(flat)
    model_tester(flat, PlanarModel)


def test_adim():

    default_mod_specs = {
        "face": {
            "contractility": 0.12,
            "vol_elasticity": 1.0,
            "prefered_height": 10.0,
            "prefered_area": 24.0,
            "prefered_vol": 240.0,
        },
        "edge": {"line_tension": 0.04},
        "vert": {"radial_tension": 0.0},
        "settings": {"grad_norm_factor": 1.0, "nrj_norm_factor": 1.0},
    }
    new_mod_specs = deepcopy(default_mod_specs)
    dim_mod_specs = model.dimensionalize(new_mod_specs)
    new_mod_specs["edge"]["line_tension"] = 0.0
    assert new_mod_specs["edge"]["line_tension"] == 0.0
    assert default_mod_specs["edge"]["line_tension"] == 0.04
    assert dim_mod_specs["edge"]["line_tension"] == 0.04 * 1 * (24 * 10) ** (5 / 3)


def test_compute_energy():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()

    sheet = Sheet("emin", datasets, specs)
    nondim_specs = config.dynamics.quasistatic_sheet_spec()
    dim_model_specs = model.dimensionalize(nondim_specs)
    sheet.update_specs(dim_model_specs, reset=True)

    geom.update_all(sheet)

    Et, Ec, Ev = model.compute_energy(sheet, full_output=True)
    assert_almost_equal(Et.mean(), 0.02458566846202479, decimal=DECIMAL)
    assert_almost_equal(Ec.mean(), 0.12093674686179141, decimal=DECIMAL)
    assert_almost_equal(Ev.mean(), 0.08788417666060594, decimal=DECIMAL)

    energy = model.compute_energy(sheet, full_output=False)
    assert_almost_equal(energy, 14.254513236339077, decimal=DECIMAL)


def test_compute_gradient():
    h5store = os.path.join(stores_dir, "small_hexagonal.hf5")
    datasets = load_datasets(h5store, data_names=["face", "vert", "edge"])
    specs = config.geometry.cylindrical_sheet()

    sheet = Sheet("emin", datasets, specs)
    nondim_specs = config.dynamics.quasistatic_sheet_spec()
    dim_model_specs = model.dimensionalize(nondim_specs)
    sheet.update_specs(dim_model_specs)

    geom.update_all(sheet)

    sheet.edge_df["is_active"] = sheet.upcast_srce("is_active") * sheet.upcast_face(
        "is_alive"
    )

    nrj_norm_factor = sheet.specs["settings"]["nrj_norm_factor"]
    print("Norm factor: ", nrj_norm_factor)
    ((grad_t, _), (grad_c, _), (grad_v_srce, grad_v_trgt)) = model.compute_gradient(
        sheet, components=True
    )
    grad_t_norm = np.linalg.norm(grad_t, axis=0).sum() / nrj_norm_factor
    assert_almost_equal(grad_t_norm, 0.22486850242320636, decimal=DECIMAL)

    grad_c_norm = np.linalg.norm(grad_c, axis=0).sum() / nrj_norm_factor
    assert_almost_equal(grad_c_norm, 0.49692791, decimal=DECIMAL)

    grad_vs_norm = np.linalg.norm(grad_v_srce.dropna(), axis=0).sum() / nrj_norm_factor
    assert_almost_equal(grad_vs_norm, 0.30281367249952407, decimal=DECIMAL)

    grad_vt_norm = np.linalg.norm(grad_v_trgt.dropna(), axis=0).sum() / nrj_norm_factor
    assert_almost_equal(grad_vt_norm, 0.27732035134768285, decimal=DECIMAL)
