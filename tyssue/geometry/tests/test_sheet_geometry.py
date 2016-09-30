import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from tyssue import config
from tyssue.core import Epithelium
from tyssue.core.generation import three_faces_sheet, extrude, hexa_grid3d, hexa_grid2d
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.config.geometry import spherical_sheet
from numpy import pi



def test_spherical_update_height():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    specs = spherical_sheet()
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)

    expected_rho = pd.Series([0.0, 1.0, 1.7320381058163818,\
                              1.9999559995159892, 1.732, 0.99997799975799462,\
                              1.7320381058163818, 2.0, 1.7320381058163818,\
                              0.99997799975799462, 1.732, 1.9999559995159892,\
                              1.7320381058163818, 0.0, 0.33333333333333331,\
                              0.57734603527212724, 0.66665199983866308,\
                              0.57733333333333325, 0.33332599991933154, \
                              0.57734603527212724, 0.66666666666666663,\
                              0.57734603527212724, 0.33332599991933154,\
                              0.57733333333333325, 0.66665199983866308, \
                              0.57734603527212724])

    expected_height = pd.Series([-4.0, -3.0, -2.2679618941836184, \
                                 -2.000044000484011, -2.2679999999999998, \
                                 -3.0000220002420055, -2.2679618941836184, \
                                 -2.0, -2.2679618941836184,\
                                 -3.0000220002420055, -2.2679999999999998, \
                                 -2.000044000484011, -2.2679618941836184,\
                                 -4.0, -3.6666666666666665, \
                                 -3.4226539647278726, -3.3333480001613367,\
                                 -3.4226666666666667, -3.6666740000806683,\
                                 -3.4226539647278726, -3.3333333333333335,\
                                 -3.4226539647278726, -3.6666740000806683,\
                                 -3.4226666666666667, -3.3333480001613367,\
                                 -3.4226539647278726])

    SheetGeometry.update_all(eptm)

    tolerance = 1e-16

    assert all((eptm.vert_df['rho'] - expected_rho)**2 < tolerance)
    assert all((eptm.vert_df['height'] - expected_height)**2 < tolerance)
