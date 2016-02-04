import tyssue.dynamics.sheet_vertex_model as model
from copy import deepcopy


def test_adim():

    default_mod_specs = {
    "face": {
        "contractility": 0.12,
        "vol_elasticity": 1.,
        "prefered_height": 10.,
        "prefered_area": 24.,
        "prefered_vol": 240.,
        },
    "je": {
        "line_tension": 0.04,
        },
    "jv": {
        "radial_tension": 0.,
        },
    "settings": {
        "grad_norm_factor": 1.,
        "nrj_norm_factor": 1.,
        }
    }
    new_mod_specs = deepcopy(default_mod_specs)
    dim_mod_specs = model.dimentionalize(new_mod_specs)
    new_mod_specs['je']['line_tension'] = 0.
    assert new_mod_specs['je']['line_tension'] == 0.
    assert default_mod_specs['je']['line_tension'] == 0.04
    assert dim_mod_specs['je']['line_tension'] == 0.04 * 1 * 24**1.5 * 10**2
