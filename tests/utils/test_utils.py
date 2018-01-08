from tyssue.utils import utils
from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet
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
