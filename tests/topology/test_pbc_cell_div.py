import tyssue

from tyssue import stores
from pathlib import Path

from tyssue import config, Sheet
from tyssue import SheetGeometry, PlanarGeometry
from tyssue.io import hdf5

from tyssue.topology.sheet_topology import cell_division


def test_bulk_celldiv():
    dsets = hdf5.load_datasets(Path(stores.stores_dir) / "planar_periodic8x8.hf5")
    specs = config.geometry.planar_sheet()
    specs["settings"]["boundaries"] = {"x": [-0.1, 8.1], "y": [-0.1, 8.1]}
    sheet = Sheet("periodic", dsets, specs)
    coords = ["x", "y"]
    draw_specs = config.draw.sheet_spec()
    PlanarGeometry.update_all(sheet)
    Nf = sheet.Nf
    # arbitrarily choose a cell from a bulk to divide
    div_cell = sheet.face_df.index[
        (sheet.face_df["at_x_boundary"] == False)
        & (sheet.face_df["at_y_boundary"] == False)
    ][0]
    daughter = cell_division(sheet, div_cell, PlanarGeometry, angle=0.6)
    assert sheet.validate()
    assert sheet.Nf == Nf + 1


def test_x_boundary_celldiv():
    dsets = hdf5.load_datasets(Path(stores.stores_dir) / "planar_periodic8x8.hf5")
    specs = config.geometry.planar_sheet()
    specs["settings"]["boundaries"] = {"x": [-0.1, 8.1], "y": [-0.1, 8.1]}
    sheet = Sheet("periodic", dsets, specs)
    coords = ["x", "y"]
    draw_specs = config.draw.sheet_spec()
    PlanarGeometry.update_all(sheet)
    Nf = sheet.Nf
    # arbitrarily choose a cells on x_boundary to divide
    div_cell = sheet.face_df.index[
        (sheet.face_df["at_x_boundary"] == True)
        & (sheet.face_df["at_y_boundary"] == False)
    ][0]
    daughter = cell_division(sheet, div_cell, PlanarGeometry, angle=0.6)
    assert sheet.validate()
    assert sheet.Nf == Nf + 1


def test_y_boundary_celldiv():
    dsets = hdf5.load_datasets(Path(stores.stores_dir) / "planar_periodic8x8.hf5")
    specs = config.geometry.planar_sheet()
    specs["settings"]["boundaries"] = {"x": [-0.1, 8.1], "y": [-0.1, 8.1]}
    sheet = Sheet("periodic", dsets, specs)
    coords = ["x", "y"]
    draw_specs = config.draw.sheet_spec()
    PlanarGeometry.update_all(sheet)
    Nf = sheet.Nf
    # arbitrarily choose a cells on y_boundary to divide
    div_cell = sheet.face_df.index[
        (sheet.face_df["at_x_boundary"] == False)
        & (sheet.face_df["at_y_boundary"] == True)
    ][0]
    daughter = cell_division(sheet, div_cell, PlanarGeometry, angle=0.6)
    assert sheet.validate()
    assert sheet.Nf == Nf + 1


def test_x_and_y_boundary_celldiv():
    dsets = hdf5.load_datasets(Path(stores.stores_dir) / "planar_periodic8x8.hf5")
    specs = config.geometry.planar_sheet()
    specs["settings"]["boundaries"] = {"x": [-0.1, 8.1], "y": [-0.1, 8.1]}
    sheet = Sheet("periodic", dsets, specs)
    coords = ["x", "y"]
    draw_specs = config.draw.sheet_spec()
    PlanarGeometry.update_all(sheet)
    Nf = sheet.Nf
    # arbitrarily choose a cells on x_boundary and y_boundary to divide
    div_cell = sheet.face_df.index[
        (sheet.face_df["at_x_boundary"] == True)
        & (sheet.face_df["at_y_boundary"] == True)
    ][0]
    daughter = cell_division(sheet, div_cell, PlanarGeometry, angle=0.6)
    assert sheet.validate()
    assert sheet.Nf == Nf + 1
