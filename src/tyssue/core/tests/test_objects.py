from tyssue.core import Epithelium, Face, JunctionVertex, JunctionEdge
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet


def test_3faces():

    datasets = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)

    face = Face(eptm, 0)
    assert len(face.jv_orbit()) == 6

    jv = JunctionVertex(eptm, 0)
    assert len(jv.face_orbit()) == 3
    assert len(jv.jv_orbit()) == 3


    je = JunctionEdge(eptm, 5)
    assert je.source_idx == 5
    assert je.target_idx == 0
    assert je.face_idx == 0


def test_triangular_mesh():
    datasets = three_faces_sheet()
    eptm = Sheet('3faces_2D', datasets)
    vertices, faces, face_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)
