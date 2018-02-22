from scipy.spatial import Voronoi
from tyssue.behaviors.sheet_events import SheetEvents
from tyssue.core.sheet import Sheet
from tyssue.generation import hexa_grid2d, from_2d_voronoi
from tyssue.dynamics.bulk_model import BulkModel
from tyssue.stores import stores_dir
from tyssue.io.hdf5 import load_datasets
from tyssue.config.geometry import cylindrical_sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom

import os
import numpy as np
import pandas as pd


def test_add_events():

    grid = hexa_grid2d(6, 4, 3, 3)
    datasets = from_2d_voronoi(Voronoi(grid))
    sheet = Sheet('test_add_events', datasets)
    behaviors_sheet = SheetEvents(sheet, BulkModel, BulkModel)
    initial_cell_event = [(1, 'contract'),
                          (4, 'contract'),
                          (6, 'contract')]

    behaviors_sheet.add_events(initial_cell_event)

    assert len(behaviors_sheet.current_deque) == 3


def test_execute_behaviors():
    h5store = os.path.join(stores_dir, 'small_hexagonal.hf5')
    datasets = load_datasets(
        h5store,
        data_names=['face', 'vert', 'edge'])
    specs = cylindrical_sheet()
    sheet = Sheet('emin', datasets, specs)
    geom.update_all(sheet)
    sheet.settings['delamination'] = {'contractile_increase': 2.0,
                                      'critical_area': 10}
    sheet.face_df['id'] = sheet.face_df.index.values

    behaviors_sheet = SheetEvents(sheet, geom, geom)
    face = 17
    initial_cell_event = [(face, 'type3')]
    behaviors_sheet.add_events(initial_cell_event)

    initial_nbsides = sheet.face_df.loc[behaviors_sheet.idx_lookup(face)][
        'num_sides']
    behaviors_sheet.execute_behaviors()
    next_nbsides = sheet.face_df.loc[behaviors_sheet.idx_lookup(face)][
        'num_sides']

    assert next_nbsides == initial_nbsides - 1
    if next_nbsides > 3:
        assert behaviors_sheet.current_deque[0][1] == 'type1_at_shorter'
