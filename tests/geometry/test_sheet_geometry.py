import os
import pandas as pd

from tyssue import config, Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet
from tyssue.io.hdf5 import load_datasets
from tyssue.stores import stores_dir

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TOLERANCE = 1e-8



def test_spherical_update_height():

    datasets, _ = three_faces_sheet(zaxis=True)
    specs = config.geometry.spherical_sheet()
    eptm = Sheet('3faces_3D', datasets, specs)

    expected_rho = pd.Series([0.0, 1.0, 1.7320381058163818,
                              1.9999559995159892, 1.732, 0.99997799975799462,
                              1.7320381058163818, 2.0, 1.7320381058163818,
                              0.99997799975799462, 1.732, 1.9999559995159892,
                              1.7320381058163818])

    expected_height = expected_rho - eptm.vert_df['basal_shift']
    SheetGeometry.update_all(eptm)

    assert all((eptm.vert_df['rho'] - expected_rho)**2 < TOLERANCE)
    assert all((eptm.vert_df['height'] - expected_height)**2 < TOLERANCE)


def test_rod_update_height():

    pth = os.path.join(stores_dir, 'rod_sheet.hf5')
    datasets = load_datasets(pth)
    specs = config.geometry.rod_sheet()
    sheet = Sheet('rod', datasets, specs)
    SheetGeometry.update_all(sheet)

    assert (sheet.vert_df.rho.mean() - 0.96074585429756632)**2 < TOLERANCE
