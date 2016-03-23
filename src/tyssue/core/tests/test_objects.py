import numpy as np
from numpy.testing import assert_array_equal


from tyssue.core import Epithelium
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet
from tyssue.core.objects import get_opposite



def test_3faces():

    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, specs)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)


def test_triangular_mesh():
    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, specs)
    vertices, faces, face_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)


def test_opposite():
    datasets, data_dicts = three_faces_sheet()
    opposites = get_opposite(datasets['edge'])
    true_opp =  np.array([17., -1., -1.,
                          -1., -1., 6., 5.,
                          -1., -1., -1., -1.,
                          12., 11., -1., -1.,
                          -1., -1., 0.])
    assert_array_equal(true_opp, opposites)
