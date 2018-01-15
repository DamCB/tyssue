import numpy as np
from tyssue import Epithelium
from tyssue import BulkGeometry
from tyssue import Sheet
from tyssue.config.geometry import bulk_spec

from tyssue.generation import extrude

from tyssue.topology.bulk_topology import IH_transition, HI_transition


def test_IH_transition():

    sheet = Sheet.planar_sheet_3d('sheet', 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets,
                       method='translation')

    eptm = Epithelium('test_IHt', datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    Nc, Nf, Ne, Nv = eptm.Nc, eptm.Nf, eptm.Ne, eptm.Nv
    eptm.settings['threshold_length'] = 1e-3
    IH_transition(eptm, 26)
    BulkGeometry.update_all(eptm)
    assert eptm.Nc == Nc
    assert eptm.Nf == Nf + 2
    assert eptm.Ne == Ne + 12
    assert eptm.Nv == Nv + 1

    invalid = eptm.get_invalid()
    assert np.alltrue(1 - invalid)
    assert np.alltrue(eptm.edge_df['sub_vol'] > 0)


def test_HI_transition():

    sheet = Sheet.planar_sheet_3d('sheet', 5, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets,
                       method='translation')

    eptm = Epithelium('test_HIt', datasets, bulk_spec())
    BulkGeometry.update_all(eptm)
    Nc, Nf, Ne, Nv = eptm.Nc, eptm.Nf, eptm.Ne, eptm.Nv
    eptm.settings['threshold_length'] = 1e-3
    IH_transition(eptm, 26)
    BulkGeometry.update_all(eptm)
    face = eptm.face_df.index[-1]
    HI_transition(eptm, face)
    assert eptm.Nc == Nc
    assert eptm.Nf == Nf
    assert eptm.Ne == Ne
    assert eptm.Nv == Nv

    invalid = eptm.get_invalid()
    assert np.alltrue(1 - invalid)
    assert np.alltrue(eptm.edge_df['sub_vol'] > 0)
