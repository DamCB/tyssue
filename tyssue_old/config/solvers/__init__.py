from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def minimize_spec():
    """Solver default specification for scipy.optimize.minimize gradient
    descent

    .. code-block::

        {
            "norm_factor": 1,
            "minimize": {
                "method": "L-BFGS-B",
                "options": {"disp": false,
                            "ftol": 1e-6,
                            "gtol": 1e-3}
            }
        }

    """
    spec = {
        "norm_factor": 1,
        "minimize": {
            "method": "L-BFGS-B",
            "options": {"disp": False, "ftol": 1e-6, "gtol": 1e-3},
        },
    }
    return spec


def quasistatic():
    """Solver default specification for scipy.optimize.minimize gradient
    descent

    """
    spec = {
        "method": "L-BFGS-B",
        "options": {"disp": False, "ftol": 1e-6, "gtol": 1e-3},
    }
    return spec
