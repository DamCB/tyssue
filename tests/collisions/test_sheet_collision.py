from pathlib import Path

from tyssue import Sheet, SheetGeometry, collisions
from tyssue.generation import three_faces_sheet
from tyssue.collisions import solvers
from tyssue.io import hdf5
from tyssue.stores import stores_dir


def test_detection():
    sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
    sheet.vert_df.z = 5 * sheet.vert_df.x ** 2
    # sheet.vert_df[sheet.coords] += np.random.normal(scale=0.001, size=(sheet.Nv, 3))
    SheetGeometry.update_all(sheet)

    sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
    SheetGeometry.update_all(sheet)
    colliding_edges = set(collisions.self_intersections(sheet).flatten())

    expected = {0, 1, 5, 6, 9, 32, 34, 35, 83, 84, 152, 153}
    assert colliding_edges == expected


def test_solving_2D():
    """
    better test  when 2D lateral geometry will be added in tyssue
    """
    sheet = Sheet("test", *three_faces_sheet())

    sheet.vert_df.loc[4, 'x'] = -1
    sheet.vert_df.loc[4, 'y'] = 0.7
    sheet.vert_df.loc[3, 'x'] = -0.4
    sheet.vert_df.loc[3, 'y'] = 1.5

    SheetGeometry.update_all(sheet)
    sheet.coords = list("xy")

    colliding_edges = collisions.self_intersections(sheet)
    boxes = solvers.CollidingBoxes2D(sheet, sheet.vert_df[sheet.coords].copy(), colliding_edges)
    boxes.solve_collisions(shyness=0.01)
    sheet.coords = list("xyz")
    SheetGeometry.update_all(sheet)
    assert collisions.self_intersections(sheet).size == 0


# def test_already():
#     # GH111
#     sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
#     sheet.vert_df.z = 5 * sheet.vert_df.x**2
#     SheetGeometry.update_all(sheet)

#     sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
#     SheetGeometry.update_all(sheet)
#     positions_buffer = sheet.vert_df[sheet.coords].copy()
#     sheet.vert_df.x -= 0.1 * (sheet.vert_df.x / 2) ** 3
#     SheetGeometry.update_all(sheet)
#     colliding_edges = collisions.self_intersections(sheet)
#     boxes = solvers.CollidingBoxes(sheet, positions_buffer, colliding_edges)
#     boxes.solve_collisions(shyness=0.01)
#     colliding_edges = collisions.self_intersections(sheet)
#     assert len(colliding_edges) == 0

# def test_already():
#     # GH111
#     sheet = Sheet("crossed", hdf5.load_datasets(Path(stores_dir) / "sheet6x5.hf5"))
#     sheet.vert_df.z = 5 * sheet.vert_df.x ** 2
#     SheetGeometry.update_all(sheet)
#
#     sheet.vert_df.x -= 35 * (sheet.vert_df.x / 2) ** 3
#     SheetGeometry.update_all(sheet)
#     positions_buffer = sheet.vert_df[sheet.coords].copy()
#     sheet.vert_df.x -= 0.1 * (sheet.vert_df.x / 2) ** 3
#     SheetGeometry.update_all(sheet)
#     colliding_edges = collisions.self_intersections(sheet)
#     boxes = solvers.CollidingBoxes(sheet, positions_buffer, colliding_edges)
#     boxes.solve_collisions(shyness=0.01)
#     colliding_edges = collisions.self_intersections(sheet)
#     assert len(colliding_edges) == 0
