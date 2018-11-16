import numpy as np

from math import sin
from numpy.testing import assert_almost_equal
from tyssue.generation.shapes import generate_ring
from tyssue import PlanarGeometry


def test_annular():

    sheet = generate_ring(10, 1, 2, R_vit=None, apical="in")
    PlanarGeometry.update_all(sheet)
    apical_length = sheet.edge_df.loc[sheet.apical_edges, "length"]
    basal_length = sheet.edge_df.loc[sheet.basal_edges, "length"]
    lateral_length = sheet.edge_df.loc[sheet.lateral_edges, "length"]

    assert sheet.Nf == 10
    assert sheet.Ne == 40
    assert sheet.Nv == 20
    assert_almost_equal(apical_length.mean(), 2 * sin(np.pi / 10))
    assert_almost_equal(basal_length.mean(), 4 * sin(np.pi / 10))
    assert_almost_equal(lateral_length.mean(), 1)
    assert (
        np.linalg.norm(sheet.vert_df.loc[sheet.apical_verts, ["x", "y"]], axis=1).mean()
        == 1
    )
    assert (
        np.linalg.norm(sheet.vert_df.loc[sheet.basal_verts, ["x", "y"]], axis=1).mean()
        == 2
    )

    sheet = generate_ring(10, 1, 2, R_vit=None, apical="out")
    PlanarGeometry.update_all(sheet)
    apical_length = sheet.edge_df.loc[sheet.apical_edges, "length"]
    basal_length = sheet.edge_df.loc[sheet.basal_edges, "length"]
    lateral_length = sheet.edge_df.loc[sheet.lateral_edges, "length"]

    assert_almost_equal(apical_length.mean(), 4 * sin(np.pi / 10))
    assert_almost_equal(basal_length.mean(), 2 * sin(np.pi / 10))
    assert (
        np.linalg.norm(sheet.vert_df.loc[sheet.apical_verts, ["x", "y"]], axis=1).mean()
        == 2
    )
    assert (
        np.linalg.norm(sheet.vert_df.loc[sheet.basal_verts, ["x", "y"]], axis=1).mean()
        == 1
    )
