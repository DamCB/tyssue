import tempfile

import numpy as np
import zarr as zr

from tyssue import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.io import zarr


def test_save_datasets():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".zarr")
    zarr.save_datasets(fh, sheet)
    with zr.open(fh) as st:
        for key in sheet.datasets:
            assert key in st


def test_load_datasets():
    sheet = Sheet("test", *three_faces_sheet())
    sheet.settings["test"] = 3
    fh = tempfile.mktemp(suffix=".zarr")
    zarr.save_datasets(fh, sheet)
    datasets, settings = zarr.load_datasets(fh)
    assert np.all(datasets["vert"][sheet.vert_df.columns] == sheet.vert_df)
    assert settings == sheet.settings
    sheet.settings["test"] = np.zeros(4)
    zarr.save_datasets(fh, sheet)
    datasets, settings = zarr.load_datasets(fh)
    assert settings["test"] == [0.0, 0.0, 0.0, 0.0]
