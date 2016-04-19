from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))



def sheet_spec():
    """Geometry specification of a sheet in a 3D space

    .. code-block::
    {
    "face": {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "num_sides": 6,
        "area": 1.0,
        "perimeter": 1.0,
        "is_alive": true
        },
     "vert": {
         "x": 0.0,
         "y": 0.0,
         "z": 0.0,
         "is_active": true,
         "rho": 0.0,
         "basal_shift": 4.0
         },
    "edge": {
        "srce": 0,
        "trgt": 0,
        "face": 0,
        "dx": 0.0,
        "dy": 0.0,
        "dz": 0.0,
        "nx": 0.0,
        "ny": 0.0,
        "nz": 1.0,
        "length": 0.0
        },
    "settings": {
        "geometry": "cylindrical",
        "height_axis": "z"
        }
    }

    """
    sheet_json = os.path.join(CURRENT_DIR, 'sheet.json')
    return load_spec(sheet_json)
