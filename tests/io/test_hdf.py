import tempfile
import numpy as np
import pandas as pd

from tyssue.generation import three_faces_sheet
from tyssue import Sheet
from tyssue.io import hdf5


def test_save_datasets():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".hdf5")
    hdf5.save_datasets(fh, sheet)
    with pd.HDFStore(fh) as st:
        for key in sheet.datasets:
            assert key in st


def test_load_datasets():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".hdf5")
    hdf5.save_datasets(fh, sheet)

    datasets = hdf5.load_datasets(fh)
    assert np.all(datasets["vert"][sheet.vert_df.columns] == sheet.vert_df)
