import pandas as pd
import numpy as np
import ipyvolume as ipv

from tyssue.generation import three_faces_sheet, extrude
from tyssue import Sheet, config, Epithelium, SheetGeometry
from tyssue.geometry.bulk_geometry import RNRGeometry
from tyssue.draw.ipv_draw import sheet_view
from tyssue.draw import highlight_cells


def test_sheet_view():

    sheet = Sheet("test", *three_faces_sheet())
    SheetGeometry.update_all(sheet)
    face_spec = {
        "color": pd.Series(range(3)),
        "color_range": (0, 3),
        "visible": True,
        "colormap": "Blues",
        "epsilon": 0.1,
    }

    color = pd.DataFrame(
        np.zeros((sheet.Ne, 3)), index=sheet.edge_df.index, columns=["R", "G", "B"]
    )

    color.loc[0, "R"] = 0.8

    edge_spec = {"color": color, "visible": True}
    fig, (edge_mesh, face_mesh) = sheet_view(sheet, face=face_spec, edge=edge_spec)
    assert face_mesh.color.shape == (39, 3)
    assert face_mesh.triangles.shape == (18, 3)
    assert face_mesh.lines is None
    assert edge_mesh.triangles is None
    assert edge_mesh.lines.shape == (18, 2)
    sheet.face_df["visible"] = False
    sheet.face_df.loc[0, "visible"] = True
    ipv.clear()
    fig, (edge_mesh, face_mesh) = sheet_view(sheet, face=face_spec, edge=edge_spec)
    assert face_mesh.triangles.shape == (6, 3)


def test_highlight():
    dsets = extrude(three_faces_sheet()[0])
    mono3 = Epithelium("3", dsets, config.geometry.bulk_spec())
    RNRGeometry.update_all(mono3)
    highlight_cells(mono3, 0)
    assert mono3.face_df.visible.sum() == 8
    highlight_cells(mono3, [0, 1], reset_visible=False)
    assert mono3.face_df.visible.sum() == 16

    highlight_cells(mono3, 2, reset_visible=True)
    assert mono3.face_df.visible.sum() == 8
