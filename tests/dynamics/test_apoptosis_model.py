from tyssue import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.utils.testing import model_tester, effector_tester
from tyssue.dynamics.apoptosis_model import ApicoBasalTension, SheetApoptosisModel


def test_effector():
    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet('test', sheet_dsets, specs)
    effector_tester(sheet, ApicoBasalTension)

def test_model():
    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet('test', sheet_dsets, specs)
    model_tester(sheet, SheetApoptosisModel)
