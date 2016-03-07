import numpy as np
import pandas as pd

from scipy.spatial import Voronoi
from tyssue.core import generation
from tyssue.core.generation import hexa_grid3d, from_3d_voronoi
from tyssue.core.generation import hexa_grid2d, from_2d_voronoi

def test_3faces():

    datasets, _ = generation.three_faces_sheet()
    assert datasets['edge'].shape[0] == 18
    assert datasets['face'].shape[0] == 3
    assert datasets['vert'].shape[0] == 13


def test_from_3d_voronoi():

    grid = hexa_grid3d(6, 4, 3)
    datasets = from_3d_voronoi(Voronoi(grid))
    assert datasets['vert'].shape[0] == 139
    assert datasets['edge'].shape[0] == 1272
    assert datasets['face'].shape[0] == 283
    assert datasets['cell'].shape[0] == 72

def test_from_2d_voronoi():

    grid = hexa_grid2d(6, 4, 1, 1)
    datasets = from_2d_voronoi(Voronoi(grid))
    assert datasets['vert'].shape[0] == 32
    assert datasets['edge'].shape[0] == 82
    assert datasets['face'].shape[0] == 24
