from tyssue.core import Epithelium, Face, JunctionVertex, JunctionEdge
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet


def test_3faces():

    datasets, data_dicts = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, data_dicts)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)

    face = Face(eptm, 0)
    assert len(face.vert_orbit()) == 6

    vert = JunctionVertex(eptm, 0)
    assert len(vert.face_orbit()) == 3
    assert len(vert.vert_orbit()) == 3


    edge = JunctionEdge(eptm, 5)
    assert edge.source_idx == 5
    assert edge.target_idx == 0
    assert edge.face_idx == 0


def test_triangular_mesh():
    datasets, data_dicts = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, data_dicts)
    vertices, faces, face_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)
