import numpy as np

from tyssue import Sheet, Monolayer
from tyssue.generation import three_faces_sheet, extrude
from tyssue.utils import connectivity
from tyssue.config.geometry import bulk_spec


def test_ef_connect():
    data, specs = three_faces_sheet()
    sheet = Sheet("test", data, specs)
    ef_connect = connectivity.edge_in_face_connectivity(sheet)
    idx = sheet.edge_df.query(f"face == {sheet.Nf-1}").index
    assert ef_connect[idx[0], idx[1]]


def test_face_face_connectivity():
    data, specs = three_faces_sheet()
    sheet = Sheet("test", data, specs)
    ffc = connectivity.face_face_connectivity(sheet, exclude_opposites=False)
    expected = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    np.testing.assert_array_equal(ffc, expected)

    ffc = connectivity.face_face_connectivity(sheet, exclude_opposites=True)
    expected = np.array([[0, 2, 2], [2, 0, 2], [2, 2, 0]])
    np.testing.assert_array_equal(ffc, expected)

    mono = Monolayer("test", extrude(data), bulk_spec())
    ffc = connectivity.face_face_connectivity(mono, exclude_opposites=False)
    assert ffc[0][ffc[0] == 2].shape == (10,)
    assert ffc[0][ffc[0] == 1].shape == (4,)
    assert ffc.max() == 4

    ffc = connectivity.face_face_connectivity(mono, exclude_opposites=True)
    assert ffc[0][ffc[0] == 2].shape == (10,)
    assert ffc[0][ffc[0] == 1].shape == (4,)
    assert ffc.max() == 2


def test_cell_cell_connectivity():

    data, _ = three_faces_sheet()
    mono = Monolayer("test", extrude(data), bulk_spec())
    ccc = connectivity.cell_cell_connectivity(mono)
    expected = np.array([[0, 36, 36], [36, 0, 36], [36, 36, 0]])
    np.testing.assert_array_equal(ccc, expected)


def test_srce_trgt_connectivity():
    data, specs = three_faces_sheet()
    sheet = Sheet("test", data, specs)
    stc = connectivity.srce_trgt_connectivity(sheet)
    expected = np.array([3, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1])
    np.testing.assert_array_equal(stc.sum(axis=0), expected)


def test_verts_in_face_connectivity():
    data, specs = three_faces_sheet()
    sheet = Sheet("test", data, specs)
    vfc = connectivity.verts_in_face_connectivity(sheet)
    assert vfc[0][vfc[0] == 2].shape == (3,)


def test_verts_in_cell_connectivity():
    data, specs = three_faces_sheet()
    mono = Monolayer("test", extrude(data), bulk_spec())
    ccc = connectivity.verts_in_cell_connectivity(mono)
    assert ccc[0][ccc[0] == 9].shape == (18,)
    assert ccc[0][ccc[0] == 18].shape == (6,)
    assert ccc[0][ccc[0] == 27].shape == (1,)
