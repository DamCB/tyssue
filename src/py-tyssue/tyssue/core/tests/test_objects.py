from tyssue.core import Epithelium, Cell, JunctionVertex, JunctionEdge
from tyssue.core.generation import three_cells_sheet

def test_world_import():
    import tyssue
    assert tyssue.core.test_import() == "howdy"


def test_3cells():

    cell_df, jv_df, je_df = three_cells_sheet()
    eptm = Epithelium('3cells_2D', cell_df, jv_df, je_df)
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
    assert je.opposite_idx == (1, 0, 2)
