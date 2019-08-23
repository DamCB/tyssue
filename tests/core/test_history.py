import warnings
import pytest
import os

import pandas as pd

from tyssue import Sheet, History, config, Epithelium, RNRGeometry
from tyssue.core.history import HistoryHdf5
from tyssue.generation import three_faces_sheet, extrude


def test_simple_history():
    """
    """
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet, extra_cols={"edge": ["dx"]})
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
    history = History(sheet, extra_cols={"face": ["area"]})
    sheet_ = history.retrieve(0)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
    assert "area" in sheet_.datasets["face"].columns
    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]

    sheet.vert_df.loc[0, "x"] = 100
    sheet.face_df["area"] = 100
    history.record()
    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        print(dset)
    assert sheet_.datasets["vert"].loc[0, "x"] == 100
    assert sheet_.datasets["face"].loc[0, "area"] != 100
    history.record(["vert", "face"])
    sheet_ = history.retrieve(2)
    assert sheet_.datasets["face"].loc[0, "area"] == 100
    sheet_ = history.retrieve(1)
    assert sheet_.datasets["face"].loc[0, "area"] != 100


def test_overwrite_time():
    sheet = Sheet("3", *three_faces_sheet())
    history = History(sheet)
    history.record(time_stamp=1)
    history.record(time_stamp=1)
    sheet_ = history.retrieve(1)
    assert sheet_.Nv == sheet.Nv


def test_retrieve_bulk():
    eptm = Epithelium("3", extrude(three_faces_sheet()[0]))
    RNRGeometry.update_all(eptm)

    history = History(eptm)
    eptm_ = history.retrieve(0)
    RNRGeometry.update_all(eptm_)


def test_historyHDF5_path_warning():
    """
    """
    sheet = Sheet("3", *three_faces_sheet())
    with pytest.warns(UserWarning):
        history = HistoryHdf5(sheet, extra_cols={"edge": ["dx"]}, path=None)


def test_historyHDF5_file_warning():
    pd.HDFStore('out.hf5', 'w').close()
    sheet = Sheet("3", *three_faces_sheet())
    with pytest.warns(UserWarning):
        history = HistoryHdf5(sheet, extra_cols={"edge": ["dx"]}, path=os.getcwd())

    os.remove('out.hf5')
    os.remove('out2.hf5')


def test_historyHDF5_retrieve():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(sheet, extra_cols={"edge": ["dx"]}, path=os.getcwd())

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    history.record(["vert", "face", "edge"], time_stamp=0)
    history.record(["vert", "face", "edge"], time_stamp=1)
    sheet_ = history.retrieve(0)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 0

    sheet_ = history.retrieve(1)
    for elem, dset in sheet_.datasets.items():
        assert dset.shape[0] == sheet.datasets[elem].shape[0]
        assert dset.time.unique()[0] == 1
    os.remove('out.hf5')


def test_historyHDF5_save_every():
    sheet = Sheet("3", *three_faces_sheet())
    history = HistoryHdf5(sheet, extra_cols={"edge": ["dx"]}, path=os.getcwd(), save_every=2, dt=1)

    for element in sheet.datasets:
        assert sheet.datasets[element].shape[0] == history.datasets[element].shape[0]
    for i in range(6):
        history.record(["vert", "face", "edge"], time_stamp=i)
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

    os.remove('out.hf5')