import numpy as np

from tyssue.core import Epithelium
from tyssue.core.generation import three_cells_sheet

from tyssue.geometry import sheet_geometry as sgeom


def test_3cells():
    cell_df, jv_df, je_df = three_cells_sheet()
    sheet = Epithelium('3cells_2D', cell_df, jv_df, je_df)
    sgeom.update_dcoords(sheet)
    sgeom.update_length(sheet)
    np.testing.assert_array_almost_equal(np.ones(sheet.nf),
                                         sheet.edge_df['length'],
                                         rtol=1e-3)

    sgeom.update_centroid(sheet)
    np.testing.assert_array_almost_equal([0.5, -1., 0.5],
                                         sheet.cell_df['x'],
                                         rtol=1e-3)

    sgeom.update_normals(sheet)
    norms = np.linalg.norm(eptm.je_df[dcoords],
                           axis=1)
    np.testing.assert_array_almost_equal(norms, 0.866, rtol=1e-3)
    sgeom.update_areas(sheet)
    np.testing.assert_allclose(sheet.cell_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)

    sgeom.update_all(sheet)
    np.testing.assert_allclose(sheet.cell_df['area'],
                               np.sqrt(3)*1.5, rtol=1e-3)
