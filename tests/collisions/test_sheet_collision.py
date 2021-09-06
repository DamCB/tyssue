import pandas as pd
import pytest
from pathlib import Path

from tyssue import Sheet, SheetGeometry
from tyssue.io import hdf5
from tyssue import collisions
from tyssue.collisions import solvers
from tyssue.stores import stores_dir


def test_detection():
    sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
    sheet.vert_df.z = 5 * sheet.vert_df.x ** 2
    # sheet.vert_df[sheet.coords] += np.random.normal(scale=0.001, size=(sheet.Nv, 3))
    SheetGeometry.update_all(sheet)

    sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
    SheetGeometry.update_all(sheet)
    colliding_edges = set(collisions.self_intersections(sheet).flatten())
    expected = {32, 1, 34, 9, 35}
    assert colliding_edges == expected


def test_solving():

    sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
    sheet.vert_df.z = 5 * sheet.vert_df.x ** 2
    SheetGeometry.update_all(sheet)
    positions_buffer = sheet.vert_df[sheet.coords].copy()

    sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
    SheetGeometry.update_all(sheet)
    colliding_edges = collisions.self_intersections(sheet)
    boxes = solvers.CollidingBoxes(sheet, positions_buffer, colliding_edges)
    boxes.solve_collisions(shyness=0.01)
    assert collisions.self_intersections(sheet).size == 0
    assert sheet.vert_df.loc[[22, 12], "x"].diff().loc[12] == 0.01


def test_already():
    # GH111
    sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
    sheet.vert_df.z = 5 * sheet.vert_df.x ** 2
    SheetGeometry.update_all(sheet)

    sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
    SheetGeometry.update_all(sheet)
    positions_buffer = sheet.vert_df[sheet.coords].copy()
    sheet.vert_df.x -= 0.1 * (sheet.vert_df.x / 2) ** 3
    SheetGeometry.update_all(sheet)
    colliding_edges = collisions.self_intersections(sheet)
    boxes = solvers.CollidingBoxes(sheet, positions_buffer, colliding_edges)
    res = boxes.solve_collisions(shyness=0.01)
    colliding_edges = collisions.self_intersections(sheet)
    assert len(colliding_edges) == 0
