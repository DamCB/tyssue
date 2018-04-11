import os
import numpy as np
import pandas as pd


from tyssue.behaviors import EventManager, wait
from tyssue.behaviors.sheet_events import division, apoptosis
from tyssue.core.sheet import Sheet
from tyssue.stores import stores_dir
from tyssue.io.hdf5 import load_datasets
from tyssue.config.geometry import cylindrical_sheet
from tyssue.geometry.sheet_geometry import SheetGeometry as geom


def test_add_events():

    manager = EventManager('face')
    initial_cell_event = [(division, 1, (), {'geom': geom}),
                          (wait, 3, (4,), {}),
                          (apoptosis, 5, (), {})]
    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 3


def test_execute_apoptosis():
    h5store = os.path.join(stores_dir, 'small_hexagonal.hf5')
    datasets = load_datasets(
        h5store,
        data_names=['face', 'vert', 'edge'])
    specs = cylindrical_sheet()
    sheet = Sheet('emin', datasets, specs)
    geom.update_all(sheet)
    sheet.settings['apoptosis'] = {'contractile_increase': 2.0,
                                   'critical_area': 0.1}
    sheet.face_df['id'] = sheet.face_df.index.values

    manager = EventManager('face')
    face_id = 17
    face_area = sheet.face_df.loc[face_id, 'area']
    initial_nbsides = sheet.face_df.loc[face_id, 'num_sides']

    initial_cell_event = (apoptosis, face_id, (),
                          sheet.settings['apoptosis'])

    manager.current.append(initial_cell_event)
    manager.execute(sheet)
    manager.update()
    assert len(manager.current) > 0
    manager.execute(sheet)
    manager.update()

    sheet.settings['apoptosis'] = {'contractile_increase': 2.0,
                                   'critical_area': 2*face_area}
    manager.current.clear()

    modified_cell_event = (apoptosis, face_id, (),
                          sheet.settings['apoptosis'])

    manager.current.append(modified_cell_event)
    manager.execute(sheet)
    manager.update()
    next_nbsides = sheet.face_df.loc[
        sheet.idx_lookup(face_id, 'face'), 'num_sides']

    assert next_nbsides == initial_nbsides - 1
    if next_nbsides > 3:
        assert len(manager.current) > 0


def test_execute_division():
    h5store = os.path.join(stores_dir,
                           'small_hexagonal.hf5')
    datasets = load_datasets(
        h5store,
        data_names=['face', 'vert', 'edge'])
    specs = cylindrical_sheet()
    sheet = Sheet('emin', datasets, specs)
    geom.update_all(sheet)
    sheet.face_df['id'] = sheet.face_df.index.values
    manager = EventManager('face')
    face_id = 17
    event = (division, face_id, (), {})
