from tyssue.core import Epithelium
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet


def test_3faces():

    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, specs)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)


def test_triangular_mesh():
    datasets, data_dicts = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, data_dicts)
    vertices, faces, face_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)
