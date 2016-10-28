import numpy as np

from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet

from tyssue.geometry.sheet_geometry import SheetGeometry as sgeom
from tyssue.stores import load_datasets
from tyssue import config


def test_3faces():
    datasets, specs = three_faces_sheet()
    sheet = Sheet('3faces_2D', datasets, specs)
    sgeom.update_dcoords(sheet)
    sgeom.update_length(sheet)
    np.testing.assert_allclose(sheet.edge_df['length'], 1,
                               rtol=1e-3)

    sgeom.update_centroid(sheet)
    np.testing.assert_array_almost_equal([0.5, -1., 0.5],
                                         sheet.face_df['x'],
                                         decimal=3)

    sgeom.update_normals(sheet)
    norms = np.linalg.norm(sheet.edge_df[['nx', 'ny', 'nz']], axis=1)
    np.testing.assert_allclose(norms, 0.866, rtol=1e-3)

    sgeom.update_areas(sheet)
    np.testing.assert_allclose(sheet.face_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)

    sgeom.update_all(sheet)
    np.testing.assert_allclose(sheet.face_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)


def test_face_rotation():

    h5store = 'small_hexagonal.hf5'
    datasets = load_datasets(h5store,
                             data_names=['face', 'vert', 'edge'])
    specs = config.geometry.sheet_spec()

    sheet = Sheet('emin', datasets, specs)
    sgeom.update_all(sheet)
    face = 17
    normal = sheet.edge_df[sheet.edge_df['face'] == face][sheet.ncoords].mean()
    rot = sgeom.face_rotation(sheet, face, 0)
                              # np.random.random()*2*np.pi)
    rotated = np.dot(rot, normal)
    np.testing.assert_allclose(rotated[:2], np.zeros(2), atol=1e-7)
