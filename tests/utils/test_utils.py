from scipy.spatial import Voronoi
from tyssue.utils import utils
from tyssue import Sheet, SheetGeometry
from tyssue.generation import three_faces_sheet, extrude
from tyssue.generation import hexa_grid2d, from_2d_voronoi
from numpy.testing import assert_almost_equal
from tyssue import Monolayer, config
from tyssue.topology.base_topology import close_face


def test_to_nd():
    grid = hexa_grid2d(6, 4, 3, 3) 
    datasets = from_2d_voronoi(Voronoi(grid)) 
    sheet = Sheet('test_extract_bounding_box', datasets) 
    result = utils._to_3d(sheet.face_df['x'])

    assert result.shape[1] == 3


def test_scaled_unscaled():

    sheet = Sheet('3faces_3D', *three_faces_sheet())
    SheetGeometry.update_all(sheet)

    def mean_area():
        return sheet.face_df.area.mean()

    prev_area = sheet.face_df.area.mean()

    sc_area = utils.scaled_unscaled(mean_area, 2,
                                    sheet, SheetGeometry)
    post_area = sheet.face_df.area.mean()
    assert post_area == prev_area
    assert_almost_equal(sc_area / post_area, 4.)


def test_modify():

    datasets, _ = three_faces_sheet()
    extruded = extrude(datasets, method='translation')
    mono = Monolayer('test', extruded,
                     config.geometry.bulk_spec())
    mono.update_specs(config.dynamics.quasistatic_bulk_spec(),
                      reset=True)
    modifiers = {
        'apical': {
            'edge': {'line_tension': 1.},
            'face': {'contractility': 0.2},
        },
        'basal': {
            'edge': {'line_tension': 3.},
            'face': {'contractility': 0.1},
        }
    }

    utils.modify_segments(mono, modifiers)
    assert mono.edge_df.loc[mono.apical_edges,
                            'line_tension'].unique()[0] == 1.
    assert mono.edge_df.loc[mono.basal_edges,
                            'line_tension'].unique()[0] == 3.


def test_ar_calculation():
    sheet = Sheet('test', *three_faces_sheet())
    e0 = sheet.edge_df.index[0]
    face = sheet.edge_df.loc[e0, 'face']
    sheet.edge_df = sheet.edge_df.loc[sheet.edge_df.index[1:]].copy()
    close_face(sheet, face)

    sheet.face_df['AR'] = utils.ar_calculation(sheet)
    assert 'AR' in sheet.face_df
