import pandas as pd
import numpy as np

from tyssue.generation import three_faces_sheet
from tyssue import Sheet, config
from tyssue.draw.ipv_draw import sheet_view



def test_sheet_view():

    sheet = Sheet('test', *three_faces_sheet())
    face_spec = {'color': pd.Series(range(3)),
                 'color_range': (0, 3),
                 'visible': True,
                 'colormap': 'Blues',
                 'epsilon': 0.1}

    color = pd.DataFrame(np.zeros((sheet.Ne, 3)),
                         index=sheet.edge_df.index,
                         columns=['R', 'G', 'B'])

    color.loc[0, 'R'] = 0.8

    edge_spec = {'color': color, 'visible': True}
    fig, (edge_mesh, face_mesh) = sheet_view(sheet, face=face_spec, edge=edge_spec)
    assert face_mesh.color.shape == (39, 3)
    assert face_mesh.triangles.shape == (18, 3)
    assert face_mesh.lines is None
    assert edge_mesh.triangles is None
    assert edge_mesh.lines.shape == (18, 2)
