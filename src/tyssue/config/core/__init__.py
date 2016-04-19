from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

def core_spec():
    """Solver default specification for scipy.optimize.minimize gradient
    descent

    .. code-block::

        {
            'face': {
                'is_alive': true,
                 },
            'edge': {
                'face': 0,
                'srce': 0,
                'trgt': 0
                 },
            'vert': {
                'is_active': true,
                 }
        }

    """
    specfile = os.path.join(CURRENT_DIR, 'core.json')
    return load_spec(specfile)
