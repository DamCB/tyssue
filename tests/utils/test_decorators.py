import pytest

from tyssue import Epithelium
from tyssue.generation import three_faces_sheet
from tyssue.utils.decorators import do_undo, validate


def test_do_undo():
    eptm = Epithelium("t", *three_faces_sheet())
    max_srce = eptm.edge_df.srce.max()

    @do_undo
    def bad_action(eptm):
        eptm.edge_df["srce"] = 0
        raise Exception

    try:
        bad_action(eptm)
    except Exception:
        assert eptm.edge_df.srce.max() == max_srce


def test_validate():
    eptm = Epithelium("t", *three_faces_sheet())

    @validate
    def drop(eptm):
        eptm.edge_df.drop(3, inplace=True)

    with pytest.raises(ValueError):
        drop(eptm)
