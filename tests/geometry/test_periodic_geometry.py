import os

from tyssue import Sheet, config
from tyssue import PlanarGeometry, SheetGeometry
from tyssue.io import hdf5
from tyssue.stores import stores_dir


def test_periodic_planar():
    hdf_path = hdf5.load_datasets(os.path.join(stores_dir, 'planar_periodic8x8.hf5'))
    specs = config.stores.planar_periodic8x8()
    sheet = Sheet('periodic', hdf_path, specs)
    PlanarGeometry.update_all(sheet)
    assert sheet.edge_df.length.max() < 1.
    fsx = sheet.edge_df['fx'] - sheet.edge_df['sx']
    fsy = sheet.edge_df['fy'] - sheet.edge_df['sy']
    lai = (fsx**2 + fsy**2)**0.5
    assert lai.max() < 0.7
    assert sheet.face_df.area.max() < 1.2
    assert sheet.face_df.area.min() > 0.8

    sheet.update_specs(config.geometry.sheet_spec())
    sheet = Sheet('2.5D', sheet.datasets, sheet.specs)
    SheetGeometry.update_all(sheet)
    assert sheet.edge_df.length.max() < 1.
    fsx = sheet.edge_df['fx'] - sheet.edge_df['sx']
    fsy = sheet.edge_df['fy'] - sheet.edge_df['sy']
    lai = (fsx**2 + fsy**2)**0.5
    assert lai.max() < 0.7
    assert sheet.face_df.area.max() < 1.2
    assert sheet.face_df.area.min() > 0.8
