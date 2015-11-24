import numpy as np
import pandas as pd

from tyssue.core import generation


def test_3cells():
    datasets = generation.three_cells_sheet()
    je_df = datasets['je']
    edge_idx = pd.MultiIndex.from_tuples([(0, 1, 0),
                                          (1, 2, 0),
                                          (2, 3, 0),
                                          (3, 4, 0),
                                          (4, 5, 0),
                                          (5, 0, 0),
                                          (0, 5, 1),
                                          (5, 6, 1),
                                          (6, 7, 1),
                                          (7, 8, 1),
                                          (8, 9, 1),
                                          (9, 0, 1),
                                          (0, 9, 2),
                                          (9, 10, 2),
                                          (10, 11, 2),
                                          (11, 12, 2),
                                          (12, 1, 2),
                                          (1, 0, 2)])
    np.testing.assert_array_equal(edge_idx.values, je_df.index)
