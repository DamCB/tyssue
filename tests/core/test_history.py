import warnings
import pytest

from tyssue import Sheet, History, config, Epithelium
from tyssue.generation import three_faces_sheet, extrude


def test_simple_history():
    """
    """
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet, {"edge": ["dx"]})
    assert "dx" in history.datasets["edge"].columns

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    history.record()
    assert sheet.datasets["vert"].shape[0] * 2 == history.datasets["vert"].shape[0]
    history.record(["vert", "face"])
    assert sheet.datasets["vert"].shape[0] * 3 == history.datasets["vert"].shape[0]
    assert sheet.datasets["face"].shape[0] * 2 == history.datasets["face"].shape[0]
    mono = Epithelium("eptm", extrude(sheet.datasets))
    histo2 = History(mono)
    for element in mono.datasets:
        assert mono.datasets[element].shape[0] == histo2.datasets[element].shape[0]


def test_warning():

    sheet = Sheet("3", *three_faces_sheet())
    with pytest.warns(UserWarning):
        history = History(sheet, extra_cols={"vert": ["invalid_column"]})


def test_retrieve():
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet, {"face": ["area"]})
    dsets = history.retrieve(0)
    for elem, dset in dsets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
    assert "area" in dsets["face"].columns
    dsets = history.retrieve(1)
    for elem, dset in dsets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]

    sheet.vert_df.loc[0, "x"] = 100
    sheet.face_df["area"] = 100
    history.record()
    dsets = history.retrieve(1)
    for elem, dset in dsets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        print(dset)
    assert dsets["vert"].loc[0, "x"] == 100
    assert dsets["face"].loc[0, "area"] != 100
    history.record(["vert", "face"])
    dsets = history.retrieve(2)
    assert dsets["face"].loc[0, "area"] == 100
    dsets = history.retrieve(1)
    assert dsets["face"].loc[0, "area"] != 100
