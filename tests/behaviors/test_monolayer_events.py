import os
import tempfile
import numpy as np
import pandas as pd

from tyssue import config
from tyssue import Monolayer, Sheet
from tyssue import MonolayerGeometry as geom
from tyssue.generation import three_faces_sheet, extrude


from tyssue.behaviors.event_manager import EventManager, wait
from tyssue.behaviors.monolayer.actions import (
    grow,
    shrink,
    contract,
    contract_apical_face,
    ab_pull,
    ab_pull_edge,
)
from tyssue.behaviors.monolayer.apoptosis_events import apoptosis
from tyssue.behaviors.monolayer.delamination_events import constriction


def test_grow():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.cell_df["prefered_vol"] = 1.0
    mono.cell_df["prefered_area"] = 1.0

    grow(mono, 0, 0.2)

    assert mono.cell_df.loc[0, "prefered_vol"] == 1.2
    assert round(mono.cell_df.loc[0, "prefered_area"], 4) == 1.1292


def test_shrink():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.cell_df["prefered_vol"] = 1.0
    mono.cell_df["prefered_area"] = 1.0

    shrink(mono, 0, 0.2)

    assert round(mono.cell_df.loc[0, "prefered_vol"], 4) == 0.8333
    assert round(mono.cell_df.loc[0, "prefered_area"], 4) == 0.8855


def test_ab_pull():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.edge_df["line_tension"] = 1.0
    mono.specs["edge"]["line_tension"] = 1.0

    assert len(mono.edge_df.line_tension.unique()) == 1
    ab_pull(mono, 0, 10, False)
    assert len(mono.edge_df.line_tension.unique()) == 2


def test_ab_pull_edge():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.edge_df["line_tension"] = 1.0
    mono.specs["edge"]["line_tension"] = 1.0

    assert len(mono.edge_df.line_tension.unique()) == 1
    ab_pull_edge(mono, 1, 2, False)
    assert len(mono.edge_df.line_tension.unique()) == 2


def test_contract():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)

    geom.center(mono)
    geom.update_all(mono)
    mono.face_df["contractility"] = 1.0
    assert len(mono.face_df.contractility.unique()) == 1
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"].unique()
    for f in faces:
        contract(mono, f, 0.2)

    assert len(mono.face_df.contractility.unique()) == 2
    for f in faces:
        assert mono.face_df.loc[f, "contractility"] == 1.2


def test_contract_apical_face():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)
    mono.face_df["id"] = mono.face_df.index.values
    geom.center(mono)
    geom.update_all(mono)
    mono.face_df["contractility"] = 1.0
    assert len(mono.face_df.contractility.unique()) == 1
    faces = mono.edge_df[mono.edge_df["cell"] == 0]["face"].unique()
    for f in faces:
        contract_apical_face(mono, f, 0.2)

    assert len(mono.face_df.contractility.unique()) == 2
    for f in faces:
        if mono.face_df.loc[f, "segment"] == "apical":
            assert mono.face_df.loc[f, "contractility"] == 1.2
        else:
            assert mono.face_df.loc[f, "contractility"] == 1.0


def test_apoptosis():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 4, 5, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)
    mono.face_df["id"] = mono.face_df.index.values
    geom.center(mono)
    geom.update_all(mono)
    mono.face_df["contractility"] = 1.0
    manager = EventManager("face")
    cell_id = 0
    apical_face = mono.face_df[
        (mono.face_df.index.isin(mono.get_orbits("cell", "face")[cell_id]))
        & (mono.face_df.segment == "apical")
    ].index[0]
    sheet.settings["apoptosis"] = {"cell_id": cell_id}
    initial_cell_event = [(apoptosis, sheet.settings["apoptosis"])]

    manager.extend(initial_cell_event)
    manager.execute(mono)
    manager.update()
    assert len(manager.current) == 1

    i = 0
    while i < 5:
        manager.execute(mono)
        manager.update()
        i = i + 1

    assert mono.face_df.loc[apical_face, "contractility"] > 1.0


def test_constriction():
    specs = config.geometry.bulk_spec()
    sheet = Sheet.planar_sheet_3d("flat", 6, 8, 1, 1)
    sheet.sanitize()
    datasets = extrude(sheet.datasets, method="translation")
    mono = Monolayer("mono", datasets, specs)
    geom.center(mono)
    geom.update_all(mono)
    dyn_specs = config.dynamics.quasistatic_bulk_spec()
    dyn_specs["cell"]["area_elasticity"] = 0.05
    dyn_specs["cell"]["prefered_area"] = 6.0
    dyn_specs["cell"]["vol_elasticity"] = 1.0
    dyn_specs["cell"]["prefered_vol"] = 1.2
    dyn_specs["face"]["contractility"] = 0.0
    dyn_specs["edge"]["line_tension"] = 0.0
    mono.update_specs(dyn_specs, reset=True)
    mono.face_df.loc[mono.apical_faces, "contractility"] = 1.12
    mono.face_df.loc[mono.basal_faces, "contractility"] = 1.12

    manager = EventManager("face")
    sheet.face_df["enter_in_process"] = 0

    mono.settings["constriction"] = {}
    mono.cell_df["is_mesoderm"] = False
    mono.face_df["is_mesoderm"] = False

    cell_to_constrict = [12]
    apical_face = mono.face_df[
        (mono.face_df.index.isin(mono.get_orbits("cell", "face")[cell_to_constrict[0]]))
        & (mono.face_df.segment == "apical")
    ].index[0]
    mono.cell_df.loc[cell_to_constrict, "is_mesoderm"] = True
    mono.cell_df["id"] = mono.cell_df.index.values
    mono.face_df["id"] = mono.face_df.index.values

    list_face_in_cell = mono.get_orbits("cell", "face")
    cell_in_mesoderm = mono.cell_df[mono.cell_df.is_mesoderm].index.values
    for i in cell_in_mesoderm:
        faces_in_cell = mono.face_df.loc[list_face_in_cell[i].unique()]
        for f in faces_in_cell.index.values:
            mono.face_df.loc[f, "is_mesoderm"] = True

    for i in cell_to_constrict:

        delam_kwargs = mono.settings["constriction"].copy()
        delam_kwargs.update(
            {
                "cell_id": i,
                "contract_rate": 2,
                "critical_area": 0.02,
                "shrink_rate": 0.4,
                "critical_volume": 0.1,
                "radial_tension": 3,
                "max_traction": 35,
                "contract_neighbors": True,
                "contract_span": 1,
                "with_rearrangement": True,
                "critical_area_reduction": 5,
            }
        )

        initial_cell_event = [(constriction, delam_kwargs)]
        manager.extend(initial_cell_event)
    manager.execute(mono)
    manager.update()
    assert len(manager.current) == 1

    i = 0
    while i < 10:
        manager.execute(mono)
        manager.update()
        i = i + 1

    assert mono.face_df.loc[apical_face, "contractility"] > 1.0
    for c in cell_to_constrict:
        assert mono.cell_df.loc[c, "num_faces"] <= 4
