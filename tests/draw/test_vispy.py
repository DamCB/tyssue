import pandas as pd
import numpy as np

from tyssue.generation import three_faces_sheet, extrude
from tyssue import Sheet, SheetGeometry
from tyssue.draw.vispy_draw import sheet_view


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
    canvas, view = sheet_view(sheet, face=face_spec, edge=edge_spec, interactive=False)
    content = view.scene.children
    edge_mesh, face_mesh = content[-2:]

    assert face_mesh.mesh_data.get_face_colors().shape == (18, 4)
    assert face_mesh.mesh_data.get_faces().shape == (18, 3)
    assert edge_mesh.pos.shape == (13, 3)
