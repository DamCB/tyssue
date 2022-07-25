import tempfile

from tyssue import Sheet
from tyssue import SheetGeometry as geom
from tyssue.generation import three_faces_sheet
from tyssue.io import meshes


def test_save_import_triangular_mesh():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".obj")
    meshes.save_triangular_mesh(fh, sheet)
    data = meshes.import_triangular_mesh(fh)
    sheet = Sheet("test", data)
    geom.update_all(sheet)
    assert sheet.Nf == 18
    assert sheet.Ne == 54
    assert sheet.Nv == 16


def test_save_import_mesh():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".ply")
    meshes.save_mesh(fh, sheet)
    data = meshes.import_mesh(fh)
    sheet = Sheet("test", data)
    geom.update_all(sheet)
    assert sheet.Nf == 3
    assert sheet.Ne == 18
    assert sheet.Nv == 13
