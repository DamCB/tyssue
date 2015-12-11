import numpy as np
import pandas as pd

from tyssue.core import generation


def test_3cells():
    datasets = generation.three_cells_sheet()
    assert 'je' in datasets
    assert 'cell' in datasets
    assert 'jv' in datasets
