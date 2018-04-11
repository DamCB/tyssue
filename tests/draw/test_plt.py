import matplotlib
matplotlib.use('agg')
import numpy as np

from tyssue.generation import three_faces_sheet
from tyssue import Sheet, config
from tyssue.draw.plt_draw import quick_edge_draw, sheet_view
import matplotlib.pyplot as plt

draw_specs = config.draw.sheet_spec()

def test_quick_edge_draw():
    sheet = Sheet('test', *three_faces_sheet())
    fig, ax = quick_edge_draw(sheet)
    assert ax.lines[0].get_xydata().shape == (54, 2)

def test_sheet_view():
    sheet = Sheet('test', *three_faces_sheet())
    sheet.vert_df['rand'] = np.linspace(0.0, 1.0, num=sheet.vert_df.shape[0])
    cmap = plt.cm.get_cmap('viridis')
    color_cmap = cmap(sheet.vert_df.rand)
    draw_specs['vert']['color'] = color_cmap
    draw_specs['vert']['alpha'] = 0.5
    draw_specs['vert']['s'] = 500
    sheet.face_df['col'] = np.linspace(0.0, 1.0,
                                        num=sheet.face_df.shape[0])

    cmap = plt.cm.get_cmap('viridis')
    color_cmap = cmap(sheet.face_df.col)

    draw_specs['edge']['visible'] = False

    draw_specs['face']['visible'] = True
    draw_specs['face']['color'] = color_cmap
    draw_specs['face']['alpha'] = 0.5
    edge_color = np.linspace(0.0, 1.0,
                            num=sheet.edge_df.shape[0])

    draw_specs['edge']['visible'] = True
    edge_cmap = cmap(edge_color)
    draw_specs['edge']['color'] = edge_cmap #[0, 0, 0, 1]
    #draw_specs['edge']['color'] = [0, 0, 0, 0.2]

    draw_specs['edge']['width'] = 8. * np.linspace(0.0, 1.0,
                                                   num=sheet.edge_df.shape[0])

    fig, ax = sheet_view(sheet, ['x', 'y'], **draw_specs)
    assert len(ax.collections) == 3
    assert ax.collections[0].get_edgecolors().shape == (13, 4)
    assert ax.collections[1].get_edgecolors().shape == (18, 4)
    assert ax.collections[2].get_edgecolors().shape == (0, 4)
    assert ax.collections[2].get_facecolors().shape == (3, 4)
