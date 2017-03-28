from ..json_parser import load_spec
import os
import warnings

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def rod_sheet():
    """Geomtetry specs of a rod sheet in 3D space

    """

    sheet_json = os.path.join(CURRENT_DIR, 'rod_sheet.json')
    return load_spec(sheet_json)


def cylindrical_sheet():
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
    sheet_json = os.path.join(CURRENT_DIR, 'cylindrical_sheet.json')
    return load_spec(sheet_json)

def flat_sheet():
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
        "geometry": "flat",
        "height_axis": "z"
        }
    }

    """
    sheet_json = os.path.join(CURRENT_DIR, 'flat_sheet.json')
    return load_spec(sheet_json)

def spherical_sheet():
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
        "geometry": "spherical",
        "height_axis": "z"
        }
    }

    """
    sheet_json = os.path.join(CURRENT_DIR, 'spherical_sheet.json')
    return load_spec(sheet_json)


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
    warnings.warn("Deprecated, use spherical_sheet(), "
                  "cylindrical_sheet() or flat_sheet() instead")

    sheet_json = os.path.join(CURRENT_DIR, 'sheet.json')
    return load_spec(sheet_json)


def planar_sheet():
    """
    {"face": {
        "num_sides": 6,
        "x": 0.0,
        "area": 0.0,
        "perimeter": 0.0,
        "is_alive": true,
        "y": 0.0
        },
    "vert": {
        "x": 0.0,
        "is_active": true,
        "y": 0.0
        },
    "edge": {
        "dy": 0.0,
        "dx": 0.0,
        "srce": 0,
        "face": 0,
        "length": 0.0,
        "nz": 0.0,
        "trgt": 0
        }
    }
    """
    planar_json = os.path.join(CURRENT_DIR, 'planar.json')
    return load_spec(planar_json)


def planar_spec():
    """
    {"face": {
        "num_sides": 6,
        "x": 0.0,
        "area": 0.0,
        "perimeter": 0.0,
        "is_alive": true,
        "y": 0.0
        },
    "vert": {
        "x": 0.0,
        "is_active": true,
        "y": 0.0
        },
    "edge": {
        "dy": 0.0,
        "dx": 0.0,
        "srce": 0,
        "face": 0,
        "length": 0.0,
        "nz": 0.0,
        "trgt": 0
        }
    }
    """
    planar_json = os.path.join(CURRENT_DIR, 'planar.json')
    return load_spec(planar_json)


def bulk_spec():
    """{
        "vert": {
            "z": 0.0,
            "x": 0.0,
            "is_active": true,
            "y": 0.0
            },
        "face": {
            "z": 0.0,
            "x": 0.0,
            "num_sides": 6,
            "area": 0.0,
            "perimeter": 0.0,
            "is_alive": true,
            "y": 0.0
            },
        "cell": {
            "z": 0.0,
            "x": 0.0,
            "vol": 0.0,
            "num_faces": 6,
            "is_alive": true,
            "y": 0.0
            },
        "edge": {
            "dz": 0.0,
            "ny": 0.0,
            "dx": 0.0,
            "nx": 0.0,
            "length": 0.0,
            "srce": 0,
            "face": 0,
            "cell": 0,
            "dy": 0.0,
            "trgt": 0,
            "nz": 0.0
            }
        }

    """
    bulk_json = os.path.join(CURRENT_DIR, 'bulk.json')
    return load_spec(bulk_json)
