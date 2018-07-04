import numpy as np

import pandas as pd
from numpy.testing import assert_array_equal
from scipy.spatial import Voronoi

from tyssue.core import Epithelium
from tyssue import config
from tyssue.generation import three_faces_sheet, extrude
from tyssue.geometry.bulk_geometry import BulkGeometry
from tyssue.geometry.cellcell_geometry import scale, update_dcoords, update_length, update_all

def test_scale():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, method='translation')

    initial_cell_df = pd.DataFrame.from_dict({'cell':[0,1,2],\
                                              'x':[0.5, -1.0, 0.5],\
                                              'y':[8.660000e-01,-6.167906e-18,-8.6600000e-01],\
                                              'z':[-0.5, -0.5, -0.5],\
                                              'is_alive':[True, True, True],
                                              'num_faces':[8,8,8],
                                              'vol':[2.598,2.598,2.598]}).set_index('cell')

    x10_cell_df = pd.DataFrame.from_dict({'cell':[0,1,2],\
                                          'x':[5., -10.0, 5.],\
                                          'y':[8.660000e+00,-6.167906e-17,-8.6600000e+00],\
                                          'z':[-5., -5., -5.],\
                                          'is_alive':[True, True, True],
                                          'num_faces':[8,8,8],
                                        'vol':[2.598,2.598,2.598]}).set_index('cell')



    eptm = Epithelium('test_volume',datasets, config.geometry.bulk_spec(), coords=['x','y','z'])

    BulkGeometry.update_all(eptm)

    scale(eptm, delta = 10.0, coords= ['x','y','z'])

    tolerance = 1e-16

    assert all( (x10_cell_df[['x','y','z']] - eptm.cell_df[['x', 'y', 'z']])**2 < 1e-16)


def test_update():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d, method='translation')

    initial_cell_df = pd.DataFrame.from_dict({'cell':[0, 1, 2],
                                              'x':[0.5, -1.0, 0.5],
                                              'y':[8.660000e-01, -6.167906e-18, -8.6600000e-01],
                                              'z':[-0.5, -0.5, -0.5],
                                              'is_alive':[True, True, True],
                                              'num_faces':[8, 8, 8],
                                              'vol':[2.598, 2.598, 2.598]}).set_index('cell')

    eptm = Epithelium('test_volume',datasets, config.geometry.bulk_spec(), coords=['x','y','z'])

    BulkGeometry.update_all(eptm)


    # TO DO : make sure this code is not actually deprecated
    #update_all(eptm)

    # check dcoords

    # check length
