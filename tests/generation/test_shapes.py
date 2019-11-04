import warnings
import pytest

from math import sin

import numpy as np
from numpy.testing import assert_almost_equal
from tyssue.generation.shapes import (
    generate_ring,
    ellipsoid_sheet,
    spherical_sheet,
    spherical_monolayer,
)
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


def test_ellipsoid():
    with pytest.warns(UserWarning):
        ell = ellipsoid_sheet(6, 7, 10, 10)
    assert ell.settings["abc"] == [6, 7, 10]
    rho = np.linalg.norm(ell.vert_df[ell.coords], axis=1)
    np.testing.assert_almost_equal(rho.max(), 10.0, decimal=1)
    np.testing.assert_almost_equal(rho.min(), 6.0, decimal=1)


def test_spherical():
    sph = spherical_sheet(1.0, 40)
    rho = np.linalg.norm(sph.vert_df[sph.coords], axis=1)
    np.testing.assert_almost_equal(rho.mean(), 1.0, decimal=1)


def test_spherical_mono():
    radius = 1  # µm
    height = 3  # the height of the cells
    Nc = 400  # the (approximate) number of cells
    R_in = radius
    R_out = radius + height

    organo = spherical_monolayer(R_in, R_out, Nc, apical="in")

    rho = np.linalg.norm(organo.vert_df[organo.coords], axis=1)

    mean_apical = rho[organo.vert_df["segment"] == "apical"].mean()
    mean_basal = rho[organo.vert_df["segment"] == "basal"].mean()
    np.testing.assert_almost_equal(mean_apical, radius, decimal=1)
    np.testing.assert_almost_equal(mean_basal, radius + height, decimal=1)
    assert np.all(organo.cell_df["vol"] > 0)
