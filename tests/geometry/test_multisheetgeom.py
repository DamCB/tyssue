import numpy as np

from tyssue.core.multisheet import MultiSheet
from tyssue.geometry.multisheetgeometry import MultiSheetGeometry
from tyssue.generation import hexa_grid2d, from_2d_voronoi
from scipy.spatial import Voronoi

import tyssue


def test_multisheet():

    base_specs = tyssue.config.geometry.flat_sheet()
    specs = base_specs.copy()
    specs["face"]["layer"] = 0
    specs["vert"]["layer"] = 0
    specs["vert"]["depth"] = 0.0
    specs["edge"]["layer"] = 0
    specs["settings"]["geometry"] = "flat"
    specs["settings"]["interpolate"] = {"function": "multiquadric", "smooth": 0}
    layer_args = [
        (24, 24, 1, 1, 0.4),
        (16, 16, 2, 2, 1),
        (24, 24, 1, 1, 0.4),
        (24, 24, 1, 1, 0.4),
    ]
    dz = 1.0

    layer_datasets = []
    for i, args in enumerate(layer_args):
        centers = hexa_grid2d(*args)
        data = from_2d_voronoi(Voronoi(centers))
        data["vert"]["z"] = i * dz
        layer_datasets.append(data)

    msheet = MultiSheet("more", layer_datasets, specs)
    bbox = [[0, 25], [0, 25]]
    for sheet in msheet:
        edge_out = sheet.cut_out(bbox, coords=["x", "y"])
        sheet.remove(edge_out)

    MultiSheetGeometry.update_all(msheet)
    assert np.all(np.isfinite(msheet[0].face_df["area"]))
