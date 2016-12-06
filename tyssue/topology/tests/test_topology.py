from tyssue.core.sheet import Sheet

from tyssue.geometry.sheet_geometry import SheetGeometry as geom
from tyssue.core.generation import three_faces_sheet
from tyssue.stores import load_datasets
from tyssue.topology.sheet_topology import (cell_division,
                                            type1_transition,
                                            split_vert)
from tyssue.config.geometry import sheet_spec


def test_division():

    h5store = 'small_hexagonal.hf5'
    datasets = load_datasets(h5store,
                             data_names=['face',
                                         'vert',
                                         'edge'])
    specs = sheet_spec()
    sheet = Sheet('emin', datasets, specs)
    geom.update_all(sheet)

    Nf, Ne, Nv = sheet.Nf, sheet.Ne, sheet.Nv

    cell_division(sheet, 17, geom)

    assert sheet.Nf - Nf == 1
    assert sheet.Nv - Nv == 2
    assert sheet.Ne - Ne == 6


def test_t1_transition():

    h5store = 'small_hexagonal.hf5'
    datasets = load_datasets(h5store,
                             data_names=['face',
                                         'vert',
                                         'edge'])
    specs = sheet_spec()
    sheet = Sheet('emin', datasets, specs)
    geom.update_all(sheet)
    face = sheet.edge_df.loc[84, 'face']
    type1_transition(sheet, 84)
    assert sheet.edge_df.loc[84, 'face'] != face


def test_split_vert():

    datasets, specs = three_faces_sheet()
    sheet = Sheet('3cells_2D', datasets, specs)
    geom.update_all(sheet)

    split_vert(sheet, 0, epsilon=1e-1)
    geom.update_all(sheet)
    assert sheet.Nv == 15
    assert sheet.Ne == 18

    datasets, specs = three_faces_sheet()
    sheet = Sheet('3cells_2D', datasets, specs)
    geom.update_all(sheet)

    split_vert(sheet, 1, epsilon=1e-1)
    geom.update_all(sheet)
    assert sheet.Nv == 14
    assert sheet.Ne == 18
