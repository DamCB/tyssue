from scipy.spatial import Voronoi
from tyssue.utils import utils
from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet
from tyssue.generation import hexa_grid2d, from_2d_voronoi
from tyssue.generation import hexa_grid3d, from_3d_voronoi
from numpy.testing import assert_almost_equal


def test_scaled_unscaled():

    sheet = Sheet('3faces_3D', *three_faces_sheet())
    SheetGeometry.update_all(sheet)

    def mean_area():
        return sheet.face_df.area.mean()

    prev_area = sheet.face_df.area.mean()

    sc_area = utils.scaled_unscaled(mean_area, 2,
                                    sheet, SheetGeometry)
    post_area = sheet.face_df.area.mean()
    assert post_area == prev_area
    assert_almost_equal(sc_area / post_area, 4.)


def test_edge_modify_original_value():
    grid = hexa_grid2d(6, 4, 3, 3)
    datasets = from_2d_voronoi(Voronoi(grid))
    sheet = Sheet('test', datasets)

    column = 'length'
    new_value = sheet.edge_df[column][0] + 1
    sheet_modify = utils.edge_modify_original_value(sheet, column, new_value)
    for elmt in sheet_modify.edge_df[column]:
        assert elmt == new_value


def test_vert_modify_original_value():
    grid = hexa_grid2d(6, 4, 3, 3)
    datasets = from_2d_voronoi(Voronoi(grid))
    sheet = Sheet('test', datasets)

    column = 'rho'
    new_value = sheet.vert_df[column][0] + 10
    sheet_modify = utils.vert_modify_original_value(sheet, column, new_value)
    for elmt in sheet_modify.vert_df[column]:
        assert elmt == new_value


def test_face_modify_original_value():
    grid = hexa_grid2d(6, 4, 3, 3)
    datasets = from_2d_voronoi(Voronoi(grid))
    sheet = Sheet('test', datasets)

    column = 'area'
    new_value = sheet.face_df[column][0] + 5
    sheet_modify = utils.face_modify_original_value(sheet, column, new_value)
    for elmt in sheet_modify.face_df[column]:
        assert elmt == new_value


def test_cell_modify_original_value():
    grid = hexa_grid3d(6, 4, 3)
    datasets = from_3d_voronoi(Voronoi(grid))
    sheet = Sheet('test', datasets)

    column = 'area'
    new_value = sheet.cell_df[column][0] + 0.5
    sheet_modify = utils.cell_modify_original_value(sheet, column, new_value)
    for elmt in sheet_modify.cell_df[column]:
        assert elmt == new_value
