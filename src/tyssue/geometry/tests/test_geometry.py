import numpy as np

from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet

from tyssue.geometry import sheet_geometry as sgeom
from tyssue.utils.utils import set_data_columns

def test_3faces():
    datasets, data_dicts = three_faces_sheet()
    sheet = Sheet('3faces_2D', datasets, data_dicts)
    sheet.set_geom('sheet')

    #sgeom.update_all(sheet, coords=sheet.coords)

    sgeom.update_dcoords(sheet)
    sgeom.update_length(sheet)
    np.testing.assert_allclose(sheet.je_df['length'], 1,
                               rtol=1e-3)

    sgeom.update_centroid(sheet)
    np.testing.assert_array_almost_equal([0.5, -1., 0.5],
                                         sheet.face_df['x'],
                                         decimal=3)

    sgeom.update_normals(sheet)
    norms = np.linalg.norm(sheet.je_df[['nx', 'ny', 'nz']], axis=1)
    np.testing.assert_allclose(norms, 0.866, rtol=1e-3)

    sgeom.update_areas(sheet)
    np.testing.assert_allclose(sheet.face_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)

    sgeom.update_all(sheet)
    np.testing.assert_allclose(sheet.face_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)
