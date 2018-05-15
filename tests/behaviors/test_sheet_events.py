import os
import numpy as np
import pandas as pd

from tyssue.core.sheet import Sheet
from tyssue.stores import stores_dir
from tyssue.io.hdf5 import load_datasets
from tyssue.generation import three_faces_sheet
from tyssue import config
from tyssue.geometry.sheet_geometry import SheetGeometry as geom

from tyssue.behaviors import EventManager, wait
from tyssue.behaviors.sheet_events import (division,
                                           apoptosis,
                                           type1_at_shorter,
                                           type3,
                                           contract,
                                           ab_pull)


def test_add_events():

    manager = EventManager('face')
    initial_cell_event = [(division, 1, (), {'geom': geom}),
                          (wait, 3, (4,), {}),
                          (apoptosis, 5, (), {})]
    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 3


def test_add_only_once():
    manager = EventManager('face')
    initial_cell_event = [(division, 1, (), {'geom': geom}),
                          (apoptosis, 3, (4,), {}),
                          (apoptosis, 3, (), {})]

    manager.extend(initial_cell_event)
    manager.execute(None)
    manager.update()
    assert len(manager.current) == 2


def test_execute_apoptosis():
    h5store = os.path.join(stores_dir, 'small_hexagonal.hf5')
    datasets = load_datasets(
        h5store,
        data_names=['face', 'vert', 'edge'])
    specs = config.geometry.cylindrical_sheet()
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
                                   'critical_area': 2 * face_area}
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

    sheet = Sheet('emin', *three_faces_sheet())
    sheet.face_df['vol'] = 1.
    sheet.specs['face']['prefered_vol'] = 1.
    sheet.face_df['prefered_vol'] = 1.

    sheet.face_df['id'] = sheet.face_df.index.values
    manager = EventManager('face')
    face_id = 1
    event = (division, face_id, (),
             {'growth_rate': 0.2, 'critical_vol': 1.5})
    manager.current.append(event)
    V0 = sheet.face_df.loc[1, 'prefered_vol']
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, 'prefered_vol'] == V0 * 1.2
    manager.execute(sheet)
    manager.update()
    assert sheet.face_df.loc[1, 'prefered_vol'] == V0 * 1.44
    sheet.face_df.loc[1, 'vol'] *= 1.6
    manager.execute(sheet)
    assert sheet.Nf == 4


def test_type1_at_shorter():

    sheet = Sheet('emin', *three_faces_sheet())
    sheet.vert_df.loc[0, 'x'] += 0.5
    geom.update_all(sheet)
    type1_at_shorter(sheet, 0, geom)
    np.all(sheet.face_df.num_sides.values == [5, 7, 5])


def test_remove_face():

    sheet = Sheet('emin', *three_faces_sheet())
    type3(sheet, 0, geom)
    assert sheet.Nf == 2
    assert sheet.Nv == 8
    assert sheet.Ne == 10


def test_contract():

    sheet = Sheet('emin', *three_faces_sheet())
    sheet.face_df['contractility'] = 1.
    contract(sheet, 0, 0.5, multiple=False)
    assert sheet.face_df.loc[0, 'contractility'] == 1.5

    contract(sheet, 1, 2., multiple=True)
    assert sheet.face_df.loc[1, 'contractility'] == 2.


def test_ab_pull():

    sheet = Sheet('emin', *three_faces_sheet())
    sheet.vert_df['radial_tension'] = 1.
    ab_pull(sheet, 0, 1., distributed=False)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, 'radial_tension'],
        np.ones(6) * 2.)
    ab_pull(sheet, 0, 3., distributed=True)
    np.testing.assert_array_equal(
        sheet.vert_df.loc[0:5, 'radial_tension'],
        np.ones(6) * 2.5)
