import numpy as np
import pandas as pd

from tyssue.core import generation


def test_3faces():
    datasets, data_dicts = generation.three_faces_sheet()
    assert 'je' in datasets
    assert 'face' in datasets
    assert 'jv' in datasets
