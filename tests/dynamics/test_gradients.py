import numpy as np

from tyssue.dynamics.base_gradients import length_grad
from tyssue import generation, Sheet, SheetGeometry
from tyssue.geometry.sheet_geometry import EllipsoidGeometry
from tyssue.dynamics.sheet_gradients import height_grad


def test_length_grad():
    sheet = Sheet("etst", *generation.three_faces_sheet())
    SheetGeometry.update_all(sheet)

    lg = length_grad(sheet)

    assert np.all(lg.loc[0].values == np.array([-1, 0, 0]))


def test_spherical_grad():
    sheet = generation.ellipsoid_sheet(1, 1, 1, 10)
    sheet.settings["geometry"] = "spherical"
    EllipsoidGeometry.update_all(sheet)
    np.testing.assert_approx_equal(
        np.linalg.norm(height_grad(sheet), axis=1).mean(), 1, 2
    )
