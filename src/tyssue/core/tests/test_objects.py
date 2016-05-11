import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal


from tyssue.core import Epithelium
from tyssue.core.sheet import Sheet
from tyssue.core.generation import three_faces_sheet
from tyssue.core.objects import get_opposite
from tyssue import config
from tyssue.geometry.planar_geometry import PlanarGeometry

def test_3faces():

    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, specs)
    assert (eptm.Nc, eptm.Nv, eptm.Ne) == (3, 13, 18)


def test_triangular_mesh():
    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets, specs)
    vertices, faces, face_mask = eptm.triangular_mesh(['x', 'y', 'z'])
    assert vertices.shape == (16, 3)
    assert faces.shape == (18, 3)


def test_opposite():
    datasets, data_dicts = three_faces_sheet()
    opposites = get_opposite(datasets['edge'])
    true_opp = np.array([17., -1., -1.,
                         -1., -1., 6., 5.,
                         -1., -1., -1., -1.,
                         12., 11., -1., -1.,
                         -1., -1., 0.])
    assert_array_equal(true_opp, opposites)


def test_extra_indices():

    datasets = {}
    tri_verts = [[0, 0],
                 [1, 0],
                 [-0.5, 3**0.5/2],
                 [-0.5, -3**0.5/2]]

    tri_edges = [[0, 1, 0],
                 [1, 2, 0],
                 [2, 0, 0],
                 [0, 3, 1],
                 [3, 1, 1],
                 [1, 0, 1],
                 [0, 2, 2],
                 [2, 3, 2],
                 [3, 0, 2]]

    datasets['edge'] = pd.DataFrame(data=np.array(tri_edges),
                                    columns=['srce', 'trgt', 'face'])
    datasets['edge'].index.name = 'edge'
    datasets['face'] = pd.DataFrame(data=np.zeros((3, 2)),
                                    columns=['x', 'y'])
    datasets['face'].index.name = 'face'

    datasets['vert'] = pd.DataFrame(data=np.array(tri_verts),
                                    columns=['x', 'y'])
    datasets['vert'].index.name = 'vert'
    specs = config.geometry.planar_spec()
    eptm = Epithelium('extra', datasets, specs, coords=['x', 'y'])
    PlanarGeometry.update_all(eptm)
    eptm.edge_df['opposite'] = get_opposite(eptm.edge_df)
    eptm.get_extra_indices()

    assert (2*eptm.Ni + eptm.No) == eptm.Ne
    assert eptm.west_edges.size == eptm.Ni
    assert eptm.Nd == 2*eptm.Ni

    for edge in eptm.free_edges:
        opps = eptm.edge_df[eptm.edge_df['srce'] ==
                            eptm.edge_df.loc[edge, 'trgt']]['trgt']
        assert eptm.edge_df.loc[edge, 'srce'] not in opps

    for edge in eptm.east_edges:
        srce, trgt = eptm.edge_df.loc[edge, ['srce', 'trgt']]
        opp = eptm.edge_df[(eptm.edge_df['srce'] == trgt) &
                           (eptm.edge_df['trgt'] == srce)].index
        assert opp[0] in eptm.west_edges

    for edge in eptm.west_edges:
        srce, trgt = eptm.edge_df.loc[edge, ['srce', 'trgt']]
        opp = eptm.edge_df[(eptm.edge_df['srce'] == trgt) &
                           (eptm.edge_df['trgt'] == srce)].index
        assert opp[0] in eptm.east_edges


def test_sort_eastwest():
    datasets = {}
    tri_verts = [[0, 0],
                 [1, 0],
                 [-0.5, 3**0.5/2],
                 [-0.5, -3**0.5/2]]

    tri_edges = [[0, 1, 0],
                 [1, 2, 0],
                 [2, 0, 0],
                 [0, 3, 1],
                 [3, 1, 1],
                 [1, 0, 1],
                 [0, 2, 2],
                 [2, 3, 2],
                 [3, 0, 2]]

    datasets['edge'] = pd.DataFrame(data=np.array(tri_edges),
                                    columns=['srce', 'trgt', 'face'])
    datasets['edge'].index.name = 'edge'
    datasets['face'] = pd.DataFrame(data=np.zeros((3, 2)),
                                    columns=['x', 'y'])
    datasets['face'].index.name = 'face'

    datasets['vert'] = pd.DataFrame(data=np.array(tri_verts),
                                    columns=['x', 'y'])
    datasets['vert'].index.name = 'vert'
    specs = config.geometry.planar_spec()
    eptm = Epithelium('extra', datasets, specs, coords=['x', 'y'])
    PlanarGeometry.update_all(eptm)
    eptm.edge_df['opposite'] = get_opposite(eptm.edge_df)
    eptm.sort_edges_eastwest()
    assert_array_equal(np.asarray(eptm.free_edges),
                       [0, 1, 2])
    assert_array_equal(np.asarray(eptm.east_edges),
                       [3, 4, 5])
    assert_array_equal(np.asarray(eptm.west_edges),
                       [6, 7, 8])
