import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from tyssue import PlanarGeometry
from tyssue.generation.shapes import (
    generate_lateral_tissue
)

def test_lateralsheet():

    sheet = generate_lateral_tissue(15, 15, 2)
    PlanarGeometry.update_all(sheet)
    apical_length = sheet.edge_df.loc[sheet.apical_edges, "length"]
    basal_length = sheet.edge_df.loc[sheet.basal_edges, "length"]
    lateral_length = sheet.edge_df.loc[sheet.lateral_edges, "length"]

    assert sheet.Nf == 15
    assert sheet.Ne == 60
    assert sheet.Nv == 32
    assert_almost_equal(apical_length.mean(), 1)
    assert_almost_equal(basal_length.mean(), 1)
    assert_almost_equal(lateral_length.mean(), 2)
