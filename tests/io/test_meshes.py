import tempfile
from tyssue.generation import three_faces_sheet
from tyssue import Sheet
from tyssue.io import meshes
from tyssue import SheetGeometry as geom


def test_save_import_triangular_mesh():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".obj")
    meshio.save_triangular_mesh(fh, sheet)
    data = meshio.import_triangular_mesh(fh)
    sheet = Sheet('test', data)
    geom.update_all(sheet)
    assert sheet.Nf == 18
    assert sheet.Ne == 54
    assert sheet.Nv == 16


def test_save_import_mesh():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".ply")
    meshio.save_mesh(fh, sheet)
    data = meshio.import_mesh(fh)
    sheet = Sheet('test', data)
    geom.update_all(sheet)
    assert sheet.Nf == 3
    assert sheet.Ne == 18
    assert sheet.Nv == 13
