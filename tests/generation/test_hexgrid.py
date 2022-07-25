from math import tau

import numpy as np

from tyssue.generation.hexagonal_grids import circle, hexa_cylinder, hexa_disk


def test_circle():

    points = circle(6, 3, tau / 3)
    assert points.shape == (6, 2)
    rho = np.linalg.norm(points, axis=1)
    np.testing.assert_array_almost_equal(rho, np.ones(6) * 3)
    np.testing.assert_array_almost_equal(circle(0), [[0.0, 0.0]])


def test_disk():

    points = hexa_disk(4, 2)
    assert points.shape == (5, 2)
    rho = np.linalg.norm(points, axis=1)
    np.testing.assert_almost_equal(rho.max(), 2.0)
    np.testing.assert_almost_equal(rho.min(), 0.0)

    points = hexa_disk(24)
    rho = np.linalg.norm(points, axis=1)
    rho.sort()
    np.testing.assert_array_almost_equal(rho[-24:], np.ones(24))


def test_hexa_cylinder():

    points = hexa_cylinder(12, 5, 1)
    assert points.shape == (60, 3)
    rho = np.linalg.norm(points[:, :2], axis=1)
    np.testing.assert_array_almost_equal(rho, np.ones(60))
    points = hexa_cylinder(12, 5, 2, capped=True)
    rho = np.linalg.norm(points[:, :2], axis=1)
    np.testing.assert_almost_equal(rho.max(), 2.0)
    np.testing.assert_almost_equal(rho.min(), 0.0)
