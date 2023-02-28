import numpy as np
import pandas as pd
from numpy import pi

from tyssue import config
from tyssue.core import Epithelium
from tyssue.geometry.planar_geometry import PlanarGeometry


def test_face_projected_pos():
    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 0.75], [-0.5, -0.75]]

    tri_edges = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 3, 1],
        [3, 1, 1],
        [1, 0, 1],
        [0, 2, 2],
        [2, 3, 2],
        [3, 0, 2],
    ]

    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"
    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"
    specs = config.geometry.planar_spec()
    eptm = Epithelium("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)

    res_rot_pos_pi2 = PlanarGeometry.face_projected_pos(eptm, 0, pi / 2.0)
    res_rot_pos_face1_2pi = PlanarGeometry.face_projected_pos(eptm, 1, 2.0 * pi)

    expected_rot_pos_pi2 = pd.DataFrame.from_dict(
        {
            "vert": [0, 1, 2, 3],
            "x": [0.25, 0.25, -0.5, 1.0],
            "y": [-0.166667, 0.8333333, -0.666667, -0.666667],
        }
    ).set_index("vert")

    expected_rot_pos_face1_2pi = pd.DataFrame.from_dict(
        {
            "vert": [0, 1, 2, 3],
            "x": [-0.166667, 0.833333, -0.666667, -0.666667],
            "y": [0.25, 0.25, 1.00, -0.5],
        }
    ).set_index("vert")
    tolerance = 1e-16
    assert all((expected_rot_pos_pi2 - res_rot_pos_pi2) ** 2 < tolerance)
    assert all((expected_rot_pos_face1_2pi - res_rot_pos_face1_2pi) ** 2 < tolerance)


def test_update_repulsion():
    datasets = {}
    tri_verts = [[0, 0], [1, 0], [-0.5, 0.75], [-0.5, -0.75]]

    tri_edges = [
        [0, 1, 0],
        [1, 2, 0],
        [2, 0, 0],
        [0, 3, 1],
        [3, 1, 1],
        [1, 0, 1],
        [0, 2, 2],
        [2, 3, 2],
        [3, 0, 2],
    ]

    datasets["edge"] = pd.DataFrame(
        data=np.array(tri_edges), columns=["srce", "trgt", "face"]
    )
    datasets["edge"].index.name = "edge"
    datasets["face"] = pd.DataFrame(data=np.zeros((3, 2)), columns=["x", "y"])
    datasets["face"].index.name = "face"

    datasets["vert"] = pd.DataFrame(data=np.array(tri_verts), columns=["x", "y"])
    datasets["vert"].index.name = "vert"
    specs = config.geometry.planar_spec()
    specs['vert']['force_repulsion'] = 10
    eptm = Epithelium("extra", datasets, specs, coords=["x", "y"])
    PlanarGeometry.update_all(eptm)
    PlanarGeometry.update_repulsion(eptm)

    assert 'grid' in eptm.vert_df.columns
    assert 'v_repulsion' in eptm.vert_df.columns
    assert len(np.unique(eptm.vert_df.loc[1, 'v_repulsion'])) > 1

