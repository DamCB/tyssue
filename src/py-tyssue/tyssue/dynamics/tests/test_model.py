import tyssue.dynamics.sheet_vertex_model as model
from copy import deepcopy


def test_adim():

    default_mod_specs = {
    "cell": {
        "contractility": (0.12, None),
        "vol_elasticity": (1., None),
        "prefered_height": (10., None),
        "prefered_area": (24., None),
        "prefered_vol": (0., None),
        },
    "je": {
        "line_tension": (0.04, None),
        },
    "jv": {
        "radial_tension": (0., None),
        }
    }
    new_mod_specs = deepcopy(default_mod_specs)
    new_mod_specs['je']['line_tension'] = (0., None)
    model.dimentionalize(new_mod_specs)
    assert new_mod_specs['je']['line_tension'] == (0., None)
    assert default_mod_specs['je']['line_tension'] == (0.04, None)
