from tyssue.core import Epithelium, Cell, JunctionVertex, JunctionEdge
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_cells_sheet


def test_3cells():

    datasets = three_cells_sheet()
    eptm = Epithelium('3cells_2D', datasets)
    assert (eptm.Nc, eptm.Nv, eptm.Nf) == (3, 13, 18)

    cell = Cell(eptm, 0)
    assert len(cell.jv_orbit()) == 6

    jv = JunctionVertex(eptm, 0)
    assert len(jv.cell_orbit()) == 3
    assert len(jv.jv_orbit()) == 3


    je = JunctionEdge(eptm, (0, 1, 0))
    assert je.source_idx == 0
    assert je.target_idx == 1
    assert je.cell_idx == 0
    assert je.oposite_idx == (1, 0, 2)

def test_triangular_mesh():
    datasets = three_cells_sheet()
    eptm = Sheet('3cells_2D', datasets)
    vertices, faces, cell_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)
