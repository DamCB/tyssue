import pytest
import os
from pathlib import Path


from tyssue import Sheet, History, Epithelium, RNRGeometry
from tyssue.core.history import HistoryHdf5
from tyssue.generation import three_faces_sheet, extrude


def test_simple_history():
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet)
    assert "dx" in history.datasets["edge"].columns

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    history.record()
    assert sheet.datasets["vert"].shape[0] * 2 == history.datasets["vert"].shape[0]
    history.record()
    assert sheet.datasets["vert"].shape[0] * 3 == history.datasets["vert"].shape[0]
    assert sheet.datasets["face"].shape[0] * 3 == history.datasets["face"].shape[0]
    mono = Epithelium("eptm", extrude(sheet.datasets))
    histo2 = History(mono)
    for element in mono.datasets:
        assert mono.datasets[element].shape[0] == histo2.datasets[element].shape[0]


def test_warning():

    sheet = Sheet("3", *three_faces_sheet())
    with pytest.warns(UserWarning):
        History(
            sheet, extra_cols={"edge": ["dx"], "face": ["area"], "vert": ["segment"]}
        )


def test_retrieve():
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet)
    sheet_ = history.retrieve(0)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
    assert "area" in sheet_.datasets["face"].columns
    with pytest.warns(UserWarning):
        sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]

    sheet.vert_df.loc[0, "x"] = 100.0
    sheet.face_df["area"] = 100.0
    history.record()
    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        print(dset)
    assert sheet_.datasets["vert"].loc[0, "x"] == 100.0
    assert sheet_.datasets["face"].loc[0, "area"] == 100.0
    history.record()
    sheet_ = history.retrieve(2)
    assert sheet_.datasets["face"].loc[0, "area"] == 100.0
    sheet_ = history.retrieve(1)
    assert sheet_.datasets["face"].loc[0, "area"] == 100.0


def test_overwrite_time():
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet)
    history.record(time_stamp=1)
    history.record(time_stamp=1)
    sheet_ = history.retrieve(1)
    assert sheet_.Nv == sheet.Nv


def test_overwrite_tim_hdf5e():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(sheet, hf5file="out.hf5")
    history.record(time_stamp=1)
    history.record(time_stamp=1)
    sheet_ = history.retrieve(1)
    os.remove("out.hf5")
    assert sheet_.Nv == sheet.Nv


def test_retrieve_bulk():
    eptm = Epithelium("3", extrude(three_faces_sheet()[0]))
    RNRGeometry.update_all(eptm)

    history = History(eptm)
    eptm_ = history.retrieve(0)
    RNRGeometry.update_all(eptm_)


def test_historyHDF5_path_warning():

    sheet = Sheet("3", *three_faces_sheet())
    with pytest.warns(UserWarning):
        history = HistoryHdf5(sheet)
        history.record(time_stamp=0)

    with pytest.warns(UserWarning):
        history = HistoryHdf5(sheet, hf5file="out.hf5")
        history.record(time_stamp=0)

    for p in Path(".").glob("out*.hf5"):
        p.unlink()


def test_historyHDF5_retrieve():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(sheet, hf5file="out.hf5")

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    history.record(time_stamp=0)
    history.record(time_stamp=1)
    sheet_ = history.retrieve(0)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 0

    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 1
    for p in Path(".").glob("out*.hf5"):
        p.unlink()


def test_historyHDF5_save_every():
    sheet = Sheet("3", *three_faces_sheet())

    history = HistoryHdf5(
        sheet,
        save_every=2,
        dt=1,
        hf5file="out.hf5",
    )

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    for i in range(6):
        history.record(time_stamp=i)
    sheet_ = history.retrieve(0)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 0

    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 0

    sheet_ = history.retrieve(2)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 2

    for p in Path(".").glob("out*.hf5"):
        p.unlink()


def test_historyHDF5_itemsize():
    sheet = Sheet("3", *three_faces_sheet())
    sheet.vert_df["segment"] = "apical"
    history = HistoryHdf5(
        sheet,
        hf5file="out.hf5",
    )

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    sheet.vert_df.loc[0, "segment"] = ""
    history.record(time_stamp=1)

    sheet.vert_df.loc[0, "segment"] = "lateral"
    history.record(time_stamp=2)

    sheet.face_df.loc[0, "area"] = 12.0
    history.record(time_stamp=3, sheet=sheet)

    sheet1_ = history.retrieve(1)
    assert sheet1_.vert_df.loc[0, "segment"] == ""

    sheet2_ = history.retrieve(2)
    assert sheet2_.vert_df.loc[0, "segment"] == "lateral"

    sheet3_ = history.retrieve(3)
    assert sheet3_.face_df.loc[0, "area"] == 12.0

    for p in Path(".").glob("out*.hf5"):
        p.unlink()


def test_historyHDF5_save_other_sheet():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(
        sheet,
        save_only={"edge": ["dx"], "face": ["area"], "vert": ["segment"]},
        hf5file="out.hf5",
    )

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    sheet.face_df.loc[0, "area"] = 1.0
    history.record(time_stamp=1)

    sheet.face_df.loc[0, "area"] = 12.0
    history.record(time_stamp=2, sheet=sheet)

    sheet1_ = history.retrieve(1)
    assert sheet1_.face_df.loc[0, "area"] == 1.0
    sheet2_ = history.retrieve(2)
    assert sheet2_.face_df.loc[0, "area"] == 12.0

    for p in Path(".").glob("out*.hf5"):
        p.unlink()


def test_historyHDF5_from_archive():

    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(sheet, hf5file="test.hf5")
    history.record()
    history.record()
    history.record()

    retrieved = HistoryHdf5.from_archive("test.hf5")
    try:
        assert isinstance(retrieved.sheet, type(sheet))
    finally:
        os.remove("test.hf5")


def test_to_and_from_archive():

    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet)
    history.record()
    history.record()
    history.record()
    history.to_archive("test.hf5")
    history_h = HistoryHdf5.from_archive("test.hf5")
    sheet_ = history_h.retrieve(2)
    try:
        assert sheet_.Nv == sheet.Nv
    finally:
        os.remove("test.hf5")


def test_unsaved_col():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(
        sheet,
        hf5file="test.hf5",
    )
    history.record()
    history.record()
    sheet.face_df["new_col"] = 0
    with pytest.warns(UserWarning):
        history.record()
    os.remove("test.hf5")


def test_change_col_types():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(
        sheet,
        hf5file="test.hf5",
    )
    history.record()
    history.record()
    sheet.face_df["z"] = "abc"
    with pytest.raises(ValueError):
        history.record()
    os.remove("test.hf5")
