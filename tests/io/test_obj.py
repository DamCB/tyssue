import tempfile

from tyssue import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.io import obj


def test_write_junction_mesh():
    sheet = Sheet("test", *three_faces_sheet())

    fh = tempfile.mktemp(suffix=".obj")
    obj.save_junction_mesh(fh, sheet)
    with open(fh) as fb:
        lines = fb.readlines()
    assert len(lines) == 35
    assert "# 13 vertices" in lines[4]


def test_save_splitted():

    sheet = Sheet("test", *three_faces_sheet())

    fh = tempfile.mktemp(suffix=".obj")
    obj.save_splitted_cells(fh, sheet)
    with open(fh) as fb:
        lines = fb.readlines()
    assert len(lines) == 63
    assert "# 39 vertices" in lines[4]


def test_save_triangulated():

    sheet = Sheet("test", *three_faces_sheet())

    fh = tempfile.mktemp(suffix=".obj")
    obj.save_triangulated(fh, sheet)
    with open(fh) as fb:
        lines = fb.readlines()
    assert len(lines) == 40
    assert "# 16 vertices" in lines[4]
