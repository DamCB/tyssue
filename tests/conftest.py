from __future__ import annotations

import shutil
import tempfile

import pytest
from pathlib import Path

DATA_PATH = Path.cwd() / "tests"


@pytest.fixture()
def data_path():
    if not DATA_PATH.exists():
        DATA_PATH.mkdir(parents=True)
    return DATA_PATH


@pytest.fixture()
def tmp_data_path():
    path = Path(tempfile.mkdtemp(prefix="tyssue_"))
    if not path.exists():
        path.mkdir(parents=True)
    return path


@pytest.fixture()
def copy_to_mount_path(data_path, tmp_data_path):
    shutil.copytree(data_path, tmp_data_path, dirs_exist_ok=True)
    return tmp_data_path


@pytest.fixture()
def transient_mount_path(data_path, tmp_data_path):
    shutil.copytree(data_path, tmp_data_path, dirs_exist_ok=True)
    yield tmp_data_path
    # shutil.rmtree(tmp_data_path)


@pytest.fixture()
def test_hf5(transient_mount_path):
    return transient_mount_path / "test.hf5"


@pytest.fixture()
def out_hf5(transient_mount_path):
    return transient_mount_path / "out.hf5"
