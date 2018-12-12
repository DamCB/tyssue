from tyssue.generation import extrude, three_faces_sheet
from tyssue import Monolayer, config, Sheet
from tyssue.utils import testing
from tyssue.dynamics.bulk_model import BulkModel, BulkModelwithFreeBorders

from tyssue.dynamics.effectors import BorderElasticity


def test_effector():

    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    mono = Monolayer.from_flat_sheet("test", sheet, config.geometry.bulk_spec())
    testing.effector_tester(mono, BorderElasticity)


def test_models():

    sheet_dsets, specs = three_faces_sheet()
    sheet = Sheet("test", sheet_dsets, specs)
    mono = Monolayer.from_flat_sheet("test", sheet, config.geometry.bulk_spec())

    testing.model_tester(mono, BulkModel)
    testing.model_tester(mono, BulkModelwithFreeBorders)
