def planar_sheet():
    """
    Specification for a 2D sheet in the plane
    """
    spec = {
        "edge": {
            "trgt": 0,
            "nz": 1.0,
            "length": 1.0,
            "face": 0,
            "srce": 0,
            "dx": 0.0,
            "dy": 0.0,
            "sx": 0.0,
            "sy": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "fx": 0.0,
            "fy": 0.0,
        },
        "vert": {"y": 0.0, "is_active": 1, "x": 0.0},
        "face": {
            "y": 0.0,
            "is_alive": 1,
            "perimeter": 0.0,
            "area": 0.0,
            "x": 0.0,
            "num_sides": 6,
        },
        "settings": {"geometry": "planar"},
    }
    return spec


def planar_spec():
    return planar_sheet()


def planar_periodic_sheet():
    spec = planar_sheet()
    spec["settings"] = {"boundaries": {"x": [-1, 1], "y": [-1, 1]}}
    return spec


def sheet_spec():

    spec = {
        "face": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "num_sides": 6,
            "area": 1.0,
            "perimeter": 1.0,
            "is_alive": 1,
        },
        "vert": {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "is_active": 1,
            "rho": 0.0,
            "height": 0.0,
            "basal_shift": 4.0,
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
            "sx": 0.0,
            "sy": 0.0,
            "sz": 0.0,
            "tx": 0.0,
            "ty": 0.0,
            "tz": 0.0,
            "fx": 0.0,
            "fy": 0.0,
            "fz": 0.0,
            "length": 1.0,
            "is_active": 1,
        },
        "settings": {"geometry": "cylindrical", "height_axis": "z"},
    }
    return spec


def periodic_sheet():
    spec = flat_sheet()
    spec["settings"].update({"boundaries": {"x": [-1, 1], "y": [-1, 1]}})
    return spec


def cylindrical_sheet():
    """Geometry specification of a sheet in a 3D space
    """
    spec = sheet_spec()
    return spec


def rod_sheet():
    """Geomtetry specs of a rod sheet in 3D space

    """
    spec = sheet_spec()
    spec["settings"].update({"geometry": "rod", "height_axis": "z", "ab": [0.0, 0.0]})
    spec["vert"].update({"left_tip": False, "right_tip": False})

    return spec


def flat_sheet():
    """Geometry specification of a sheet in a 3D space

    """
    spec = sheet_spec()
    spec["settings"].update({"geometry": "flat"})
    return spec


def spherical_sheet():
    """Geometry specification of a sheet in a 3D space.

    Height is computed with respect to the distance to
    the coordinate systems center

    """

    spec = sheet_spec()
    spec["settings"].update({"geometry": "spherical"})
    return spec


def bulk_spec():
    """ Geometry specification for bulk tissues

    """
    spec = sheet_spec()

    spec["edge"].update({"cx": 0.0, "cy": 0.0, "cz": 0.0, "cell": 0, "sub_vol": 0.0})

    spec["cell"] = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "area": 0.0,
        "vol": 0.0,
        "num_faces": 6,
        "is_alive": 1,
    }
    return spec
