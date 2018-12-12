import os
import tempfile
import numpy as np
import pandas as pd

from tyssue import config
from tyssue import Monolayer, Sheet
from tyssue import MonoLayerGeometry as geom
from tyssue.generation import three_faces_sheet, extrude


from tyssue.behaviors.event_manager import EventManager, wait
from tyssue.behaviors.monolayer.actions import (grow,
                                                shrink,
                                                contract,
                                                contract_apical_face,
                                                ab_pull)
from tyssue.behaviors.monolayer.apoptosis_events import apoptosis


def test_grow():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d('flat', 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method='translation')
    mono = Monolayer('mono', datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.cell_df['prefered_vol'] = 1.
    mono.face_df['prefered_area'] = 1.

    grow(mono, 0, 0.2)

    assert mono.cell_df.loc[0, 'prefered_vol'] == 1.2
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"]
    for f in faces:
        assert round(mono.face_df.loc[f, "prefered_area"], 4) == 1.1292


def test_shrink():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d('flat', 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method='translation')
    mono = Monolayer('mono', datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.cell_df['prefered_vol'] = 1.
    mono.face_df['prefered_area'] = 1.

    shrink(mono, 0, 0.2)

    assert round(mono.cell_df.loc[0, 'prefered_vol'], 4) == 0.8333
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"]
    for f in faces:
        assert round(mono.face_df.loc[f, "prefered_area"], 4) == 0.8855


def test_ab_pull():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d('flat', 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method='translation')
    mono = Monolayer('mono', datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.edge_df['line_tension'] = 1.
    mono.specs["edge"]["line_tension"] = 1.

    assert len(mono.edge_df.line_tension.unique()) == 1
    ab_pull(mono, 0, 10, False)
    assert len(mono.edge_df.line_tension.unique()) == 2


def test_contract():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d('flat', 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method='translation')
    mono = Monolayer('mono', datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.face_df["contractility"] = 1.
    assert len(mono.face_df.contractility.unique()) == 1
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"].unique()
    for f in faces:
        contract(mono, f, 0.2)

    assert len(mono.face_df.contractility.unique()) == 2
    for f in faces:
        assert mono.face_df.loc[f, "contractility"] == 1.2


def test_contract_apical_face():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d('flat', 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method='translation')
    mono = Monolayer('mono', datasets, specs)
    mono.face_df['id'] = mono.face_df.index.values
    geom.center(mono)
    geom.update_all(mono)
    mono.face_df["contractility"] = 1.
    assert len(mono.face_df.contractility.unique()) == 1
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"].unique()
    for f in faces:
        contract_apical_face(mono, f, 0.2)

    assert len(mono.face_df.contractility.unique()) == 2
    for f in faces:
        if mono.face_df.loc[f, "segment"] == "apical":
            assert mono.face_df.loc[f, "contractility"] == 1.2
        else:
            assert mono.face_df.loc[f, "contractility"] == 1.
