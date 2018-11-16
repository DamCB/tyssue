from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def quasistatic_bulk_spec():
    specfile = os.path.join(CURRENT_DIR, "bulk_qs.json")
    return load_spec(specfile)


def quasistatic_plane_spec():
    """Default specification for the dynamics
    of a sheet vertex model - also suitable for general faceted tissue

    .. code-block::

        {
        "face": {
            "contractility": 0.04,
            "vol_elasticity": 1.0,
            "prefered_height": 10.0,
            "prefered_area": 24.0,
            "prefered_vol": 0.0
            },
        "vert": {
             "radial_tension": 0.0
             },
        "edge": {
             "is_active": 1,
             "line_tension": 0.12
             },
        "settings": {
            "grad_norm_factor": 1.0,
            "nrj_norm_factor": 1.0
            }
        }

    """
    specfile = os.path.join(CURRENT_DIR, "plane_qs.json")
    return load_spec(specfile)


def quasistatic_sheet_spec():
    """Default specification for the dynamics
    of a sheet vertex model - also suitable for general faceted tissue

    .. code-block::

        {
        "face": {
            "contractility": 0.04,
            "vol_elasticity": 1.0,
            "prefered_height": 10.0,
            "prefered_area": 24.0,
            "prefered_vol": 0.0
            },
        "vert": {
             "radial_tension": 0.0
             },
        "edge": {
             "line_tension": 0.12
             },
        "settings": {
            "grad_norm_factor": 1.0,
            "nrj_norm_factor": 1.0
            }
        }

    """
    specfile = os.path.join(CURRENT_DIR, "sheet_qs.json")
    return load_spec(specfile)


def quasistatic_sheet_spec_anchors():
    """Default specification for the dynamics
    of a sheet vertex model with anchor vertices

    .. code-block::

        {
        "face": {
            "contractility": 0.04,
            "vol_elasticity": 1.0,
            "prefered_height": 10.0,
            "prefered_area": 24.0,
            "prefered_vol": 0.0
            },
        "vert": {
            "radial_tension": 0.0,
            "at_border": 0,
            "is_anchor": 0
             },
        "edge": {
            "anchor_elasticity": 0.0,
            "line_tension": 0.12,
            "at_border": 0,
            "is_anchor": 0
             },
        "settings": {
            "grad_norm_factor": 1.0,
            "nrj_norm_factor": 1.0
            }
        }

    """
    specfile = os.path.join(CURRENT_DIR, "sheet_qs_anchor.json")
    return load_spec(specfile)
