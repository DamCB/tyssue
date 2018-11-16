from ..geometry import planar_periodic_sheet


def planar_periodic8x8():

    spec = planar_periodic_sheet()
    spec["settings"].update({"boundaries": {"x": [-0.1, 8], "y": [-0.1, 8]}})
    return spec
