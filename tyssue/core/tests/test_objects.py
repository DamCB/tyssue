import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from tyssue.core import Epithelium
from tyssue.core.generation import three_faces_sheet
from tyssue.core.objects import get_opposite, _ordered_edges, ordered_vert_idxs
from tyssue import config
from tyssue.geometry.planar_geometry import PlanarGeometry
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.core.generation import extrude, hexa_grid3d, hexa_grid2d
from tyssue.config.dynamics import quasistatic_sheet_spec
from tyssue.config.geometry import spherical_sheet
from pytest import raises





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

    
#### BC ####


def test_wrong_datasets_keys():
    datasets, specs = three_faces_sheet()
    datasets['edges'] = datasets['edge']
    del datasets['edge']
    with raises(ValueError, message='Expecting a ValueError since edge not in datasets'):
        eptm = Epithelium('3faces_2D', datasets, specs)
    
        

def test_optional_args_eptm():
    datasets, specs = three_faces_sheet()
    data_names = set(datasets.keys())
    eptm = Epithelium('3faces_2D', datasets)
    specs_names = set(eptm.specs.keys())
    assert specs_names.issuperset(data_names)
    assert 'settings' in specs_names
    
def test_3d_eptm_cell_getter_setter():
    datasets_2d, specs = three_faces_sheet()
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets)
    assert eptm.cell_df is not None

    datasets_3d = extrude(datasets_2d, method='translation')
    eptm.cell_df = datasets_3d['cell']
    for key in eptm.cell_df:
        assert key in datasets_3d['cell']    


def test_eptm_copy():  
    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets)
    eptm_copy = eptm.copy()
    assert eptm_copy.identifier == eptm.identifier + '_copy'
    assert set(eptm_copy.datasets.keys()).issuperset(eptm.datasets.keys())
    eptm_deepcopy = eptm.copy(deep_copy=True)
    assert eptm_deepcopy is not None
    

def test_settings_getter_setter():
    datasets, specs = three_faces_sheet()
    eptm = Epithelium('3faces_2D', datasets)
    
    eptm.settings['settings1'] = 154
    # not validated in coverage
    # (Actually the 'settings' getter is called
    # and then the dictionary class setter
    # instead of directly the 'settings' setter.)
    # See http://stackoverflow.com/a/3137768
    assert 'settings1' in eptm.settings
    assert eptm.specs['settings']['settings1'] == 154
    assert eptm.settings['settings1'] == 154



def test_idx_getters():
    datasets_2d, specs = three_faces_sheet()
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets)
    #assert len(eptm.edge_idx.difference(datasets['edge'].index)) == 0
    assert len(eptm.face_idx.difference(datasets['face'].index)) == 0
    assert len(eptm.vert_idx.difference(datasets['vert'].index)) == 0
    assert len(eptm.cell_idx.difference(datasets['cell'].index)) == 0
    assert len(eptm.e_cell_idx.index.difference(datasets['edge']['cell'].index)) == 0
    
    edge_idx_array = np.vstack((datasets['edge']['srce'],\
                                datasets['edge']['trgt'],\
                                datasets['edge']['face'])).T
    
    assert_array_equal(eptm.edge_idx_array,edge_idx_array)

    
def test_number_getters():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)
    eptm_2d = Epithelium('3faces_2D',datasets_2d, specs)

    assert eptm.Nc == datasets['cell'].shape[0]
    assert eptm.Nv == datasets['vert'].shape[0]
    assert eptm.Nf == datasets['face'].shape[0]
    assert eptm.Ne == datasets['edge'].shape[0]

    

def test_upcast():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)
    eptm_2d = Epithelium('3faces_2D',datasets_2d, specs)

    expected_res = datasets['vert'].loc[eptm.e_cell_idx]
    expected_res.index = eptm.edge_df.index
    
    assert_array_equal(expected_res, eptm.upcast_cell(datasets['vert']))
                       

def test_summation():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)
    eptm_2d = Epithelium('3faces_2D',datasets_2d, specs)

    edge_copy = datasets['edge'].copy()
    edge_copy.index = eptm.edge_mindex
    
    assert_array_equal(edge_copy.sum(level='cell'),eptm.sum_cell(eptm.edge_df))

    eptm_2d.edge_df['test_sum'] = np.linspace(1,eptm_2d.Ne, eptm_2d.Ne)

    res_sum_srce = eptm_2d.sum_srce(eptm_2d.edge_df['test_sum'])
    expected_sum_srce = pd.Series([21.0, 20.0, 3.0, 4.0, 5.0, 14.0, 9.0, 10.0, 11.0, 26.0, 15.0, 16.0, 17.0])

    res_sum_trgt = eptm_2d.sum_trgt(eptm_2d.edge_df['test_sum'])
    expected_sum_trgt = pd.Series([36.0, 18.0, 2.0, 3.0, 4.0, 12.0, 8.0, 9.0, 10.0, 24.0, 14.0, 15.0, 16.0])

    res_sum_face = eptm_2d.sum_face(eptm_2d.edge_df['test_sum'])
    expected_sum_face = pd.Series([21.0, 57.0, 93.0])
    
    assert (expected_sum_srce == res_sum_srce).all()
    assert (expected_sum_trgt == res_sum_trgt).all()
    assert (expected_sum_face == res_sum_face).all()
    
    
    
    
def test_orbits():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)
    eptm_2d = Epithelium('3faces_2D',datasets_2d, specs)

    
    expected_res_cell = datasets['edge'].groupby('srce').apply(
        lambda df:df['cell'])

    expected_res_face = datasets['edge'].groupby('face').apply(
        lambda df:df['trgt'])
    
    assert_array_equal(expected_res_cell, eptm.get_orbits('srce','cell'))
    assert_array_equal(expected_res_face, eptm.get_orbits('face','trgt'))
    
    
  
def test_polygons():
    datasets_2d, specs = three_faces_sheet(zaxis=True)
    datasets = extrude(datasets_2d)
    eptm = Epithelium('3faces_3D', datasets, specs)
    eptm_2d = Epithelium('3faces_2D',datasets_2d, specs)



    dict_expected_res = {0:[[0.0,0.0,0.0],
                       [1.0,0.0,0.0],
                       [1.5,0.86599999999999999,0.0],
                       [1.0,1.732,0.0],
                       [0.0,1.732,0.0],
                       [-0.5,0.86599999999999999,0.0]],
                    1:[[0.0,0.0,0.0],
                       [-0.5,0.86599999999999999,0.0],
                       [-1.5,0.86599999999999999,0.0],
                       [-2.0,0.0,0.0],
                       [-1.5,-0.86599999999999999,0.0],
                       [-0.5,-0.86599999999999999,0.0]],
                    2:[[0.0,0.0,0.0],
                       [-0.5,-0.86599999999999999,0.0],
                       [0.0,-1.732,0.0],
                       [1.0,-1.732,0.0],
                       [1.5,-0.86599999999999999,0.0],
                       [1.0,0.0,0.0]],
                    3:[[0.33333333333333331,0.0,0.0],
                       [0.0,0.0,0.0],
                       [-0.16666666666666666,0.28866666666666663,0.0],
                       [0.0,0.57733333333333325,0.0],
                       [0.33333333333333331,0.57733333333333325,0.0],
                       [0.5, 0.28866666666666663, 0.0]],
                    4:[[-0.16666666666666666, 0.28866666666666663, 0.0],
                       [0.0, 0.0, 0.0],
                       [-0.16666666666666666, -0.28866666666666663, 0.0],
                       [-0.5, -0.28866666666666663,0.0],
                       [-0.66666666666666663, 0.0, 0.0],
                       [-0.5, 0.28866666666666663, 0.0]],
                    5:[[-0.16666666666666666, -0.28866666666666663, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.33333333333333331, 0.0, 0.0],
                       [0.5, -0.28866666666666663, 0.0],
                       [0.33333333333333331, -0.57733333333333325, 0.0],
                       [0.0, -0.57733333333333325, 0.0]],
                    6:[[1.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.33333333333333331, 0.0, 0.0]],
                    7:[[1.5, 0.86599999999999999, 0.0],
                       [1.0, 0.0, 0.0],
                       [0.33333333333333331, 0.0, 0.0],
                       [0.5, 0.28866666666666663, 0.0]],
                    8:[[1.0, 1.732, 0.0],
                       [1.5, 0.86599999999999999, 0.0],
                       [0.5, 0.28866666666666663, 0.0],
                       [0.33333333333333331, 0.57733333333333325, 0.0]],
                    9:[[0.0, 1.732, 0.0],
                       [1.0, 1.732, 0.0],
                       [0.33333333333333331, 0.57733333333333325, 0.0],
                       [0.0, 0.57733333333333325, 0.0]],
                    10:[[-0.5, 0.86599999999999999, 0.0],
                        [0.0, 1.732, 0.0],
                        [0.0, 0.57733333333333325, 0.0],
                        [-0.16666666666666666, 0.28866666666666663, 0.0]],
                    11:[[0.0, 0.0, 0.0],
                        [-0.5, 0.86599999999999999, 0.0],
                        [-0.16666666666666666, 0.28866666666666663, 0.0],
                        [0.0, 0.0, 0.0]],
                    12:[[-0.5, 0.86599999999999999, 0.0],
                        [0.0,0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [-0.16666666666666666, 0.28866666666666663, 0.0]],
                    13:[[-1.5, 0.86599999999999999, 0.0],
                        [-0.5, 0.86599999999999999, 0.0],
                        [-0.16666666666666666, 0.28866666666666663, 0.0],
                        [-0.5, 0.28866666666666663, 0.0]],
                    14:[[-2.0, 0.0, 0.0],
                        [-1.5, 0.86599999999999999, 0.0],
                        [-0.5, 0.28866666666666663, 0.0],
                        [-0.66666666666666663, 0.0, 0.0]],
                    15:[[-1.5, -0.86599999999999999, 0.0],
                        [-2.0, 0.0, 0.0],
                        [-0.66666666666666663, 0.0, 0.0],
                        [-0.5, -0.28866666666666663, 0.0]],
                    16:[[-0.5, -0.86599999999999999, 0.0],
                        [-1.5, -0.86599999999999999, 0.0],
                        [-0.5, -0.28866666666666663,0.0],
                        [-0.16666666666666666, -0.28866666666666663, 0.0]],
                    17:[[0.0, 0.0, 0.0],
                        [-0.5, -0.86599999999999999, 0.0],
                        [-0.16666666666666666, -0.28866666666666663, 0.0],
                        [0.0, 0.0, 0.0]],
                    18:[[-0.5, -0.86599999999999999, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [-0.16666666666666666, -0.28866666666666663, 0.0]],
                    19:[[0.0, -1.732, 0.0],
                        [-0.5, -0.86599999999999999, 0.0],
                        [-0.16666666666666666, -0.28866666666666663, 0.0],
                        [0.0, -0.57733333333333325, 0.0]],
                    20:[[1.0,-1.732, 0.0],
                        [0.0,-1.732, 0.0],
                        [0.0, -0.57733333333333325, 0.0],
                        [0.33333333333333331, -0.57733333333333325, 0.0]],
                    21:[[1.5, -0.86599999999999999, 0.0],
                        [1.0, -1.732, 0.0],
                        [0.33333333333333331, -0.57733333333333325, 0.0],
                        [0.5, -0.28866666666666663, 0.0]],
                    22:[[1.0, 0.0, 0.0],
                        [1.5, -0.86599999999999999, 0.0],
                        [0.5, -0.28866666666666663, 0.0],
                        [0.33333333333333331, 0.0, 0.0]],
                    23:[[0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.33333333333333331, 0.0, 0.0],
                        [0.0, 0.0, 0.0]]}

    expected_res = pd.Series(dict_expected_res)

    ## test standard on a 3d-epithelium
    
    res=eptm.face_polygons(['x','y','z'])
    assert all([(expected_res[i] ==  res[i]).all() for i in range(res.shape[0])])

def test_face_polygons_exception():

    datasets = {}
    tri_verts = [[0, 0],
                 [1, 0],
                 [-0.5, 3**0.5/2],
                 [-0.5, -3**0.5/2]]
    
    tri_edges_valid = [[0, 1, 0],
                       [1, 2, 0],
                       [2, 0, 0],
                       [0, 3, 1],
                       [3, 1, 1],
                       [1, 0, 1],
                       [0, 2, 2],
                       [2, 3, 2],
                       [3, 0, 2]]
    
    tri_edges_invalid = [[0, 1, 0],
                         [1, 2, 0],
                         [2, 0, 0],
                         [0, 3, 1],
                         [3, 1, 1],
                         [1, 0, 1],
                         [0, 2, 2],
                         [2, 3, 2],
                         [3, 1, 2]] # changed 0 to 1 to create an invalid face


    datasets['edge'] = pd.DataFrame(data=np.array(tri_edges_valid),
                                    columns=['srce', 'trgt', 'face'])
    datasets['edge'].index.name = 'edge'
    
    datasets['face'] = pd.DataFrame(data=np.zeros((3, 2)),
                                    columns=['x', 'y'])
    datasets['face'].index.name = 'face'
    
    datasets['vert'] = pd.DataFrame(data=np.array(tri_verts),
                                    columns=['x', 'y'])
    datasets['vert'].index.name = 'vert'
    
    specs = config.geometry.planar_spec()
    eptm = Epithelium('valid', datasets, specs, coords=['x', 'y'])
    PlanarGeometry.update_all(eptm)
    
    eptm.face_polygons(['x','y'])
    
def test_invalid_valid_sanitize():
    # get_invalid and get_valid
    
   datasets = {}
   tri_verts = [[0, 0],
                [1, 0],
                [-0.5, 3**0.5/2],
                [-0.5, -3**0.5/2]]
   
   tri_edges_valid = [[0, 1, 0],
                    [1, 2, 0],
                      [2, 0, 0],
                      [0, 3, 1],
                      [3, 1, 1],
                      [1, 0, 1],
                      [0, 2, 2],
                      [2, 3, 2],
                      [3, 0, 2]]
   
   tri_edges_invalid = [[0, 1, 0],
                        [1, 2, 0],
                        [2, 0, 0],
                        [0, 3, 1],
                        [3, 1, 1],
                        [1, 0, 1],
                        [0, 2, 2],
                        [2, 3, 2],
                        [3, 1, 2]] # changed 0 to 1 to create an invalid face

   ## Epithelium whose faces are all valid
   ## 
   datasets['edge'] = pd.DataFrame(data=np.array(tri_edges_valid),
                                   columns=['srce', 'trgt', 'face'])
   datasets['edge'].index.name = 'edge'
   
   datasets['face'] = pd.DataFrame(data=np.zeros((3, 2)),
                                   columns=['x', 'y'])
   datasets['face'].index.name = 'face'
   
   datasets['vert'] = pd.DataFrame(data=np.array(tri_verts),
                                   columns=['x', 'y'])
   datasets['vert'].index.name = 'vert'

   specs = config.geometry.planar_spec()
   eptm = Epithelium('valid', datasets, specs, coords=['x', 'y'])
   PlanarGeometry.update_all(eptm)

   ## Epithelium with invalid faces (last 3)
   datasets_invalid = datasets.copy()
   datasets_invalid['edge'] = pd.DataFrame(data=np.array(tri_edges_invalid),
                                           columns=['srce', 'trgt', 'face'])
   datasets_invalid['edge'].index.name = 'edge'
   
   eptm_invalid = Epithelium('invalid', datasets_invalid, specs, coords=['x', 'y'])
   PlanarGeometry.update_all(eptm_invalid)
   

   eptm.get_valid()
   eptm_invalid.get_valid()

   res_invalid_expect_all_false = eptm.get_invalid()
   res_invalid_expect_some_true = eptm_invalid.get_invalid()
   
   assert eptm.edge_df['is_valid'].all()
   assert (not eptm_invalid.edge_df['is_valid'][6:].all())
   assert (not res_invalid_expect_all_false.all() )
   assert res_invalid_expect_some_true[6:].all()

   ### testing sanitize
   ### edges 6 to 9 should be removed.
   edge_df_before = eptm.edge_df.copy()
   edge_df_invalid_before = eptm_invalid.edge_df.copy()
   
   eptm.sanitize()
   eptm_invalid.sanitize()

   assert edge_df_before.equals(eptm.edge_df)
   assert edge_df_invalid_before[:6].equals(eptm_invalid.edge_df)
   
   
   

def test_remove():
    datasets, specs = three_faces_sheet()
    datasets = extrude(datasets, method="translation")

    eptm = Epithelium('3Faces_3D', datasets, specs)



    dict_before = {'srce': {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: \
                            0, 7: 5, 8: 6, 9: 7, 10: 8, 11: 9, 12: \
                            0, 13: 9, 14: 10, 15: 11, 16: 12, 17: \
                            1, 18: 14, 19: 15, 20: 16, 21: 17, 22: \
                            18, 23: 13, 24: 18, 25: 19, 26: 20, 27:\
                            21, 28: 22, 29: 13, 30: 22, 31: 23, 32:\
                            24, 33: 25, 34: 14, 35: 13, 36: 1, 37: 0,\
                            38: 13, 39: 14, 40: 2, 41: 1, 42: 14, 43:\
                            15, 44: 3, 45: 2, 46: 15, 47: 16, 48: 4, \
                            49: 3, 50: 16, 51: 17, 52: 5, 53: 4, 54: \
                            17, 55: 18, 56: 0, 57: 5, 58: 18, 59: 13,\
                            60: 5, 61: 0, 62: 13, 63: 18, 64: 6, 65: 5,\
                            66: 18, 67: 19, 68: 7, 69: 6, 70: 19, 71: \
                            20, 72: 8, 73: 7, 74: 20, 75: 21, 76: 9, 77:\
                            8, 78: 21, 79: 22, 80: 0, 81: 9, 82: 22, 83:\
                            13, 84: 9, 85: 0, 86: 13, 87: 22, 88: 10, \
                            89: 9, 90: 22, 91: 23, 92: 11, 93: 10, 94:\
                            23, 95: 24, 96: 12, 97: 11, 98: 24, 99: 25,\
                            100: 1, 101: 12, 102: 25, 103: 14, 104: 0,\
                            105: 1, 106: 14, 107: 13}, \
                   'trgt': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 0, 6: 5, 7:\
                            6, 8: 7, 9: 8, 10: 9, 11: 0, 12: 9, 13: 10,\
                            14: 11, 15: 12, 16: 1, 17: 0, 18: 13, 19: \
                            14, 20: 15, 21: 16, 22: 17, 23: 18, 24: 13,\
                            25: 18, 26: 19, 27: 20, 28: 21, 29: 22, 30:\
                            13, 31: 22, 32: 23, 33: 24, 34: 25, 35: 14,\
                            36: 0, 37: 13, 38: 14, 39: 1, 40: 1, 41: 14,\
                            42: 15, 43: 2, 44: 2, 45: 15, 46: 16, 47: 3, \
                            48: 3, 49: 16, 50: 17, 51: 4, 52: 4, 53: 17,\
                            54: 18, 55: 5, 56: 5, 57: 18, 58: 13, 59: 0, \
                            60: 0, 61: 13, 62: 18, 63: 5, 64: 5, 65: 18,\
                            66: 19, 67: 6, 68: 6, 69: 19, 70: 20, 71: 7, \
                            72: 7, 73: 20, 74: 21, 75: 8, 76: 8, 77: 21,\
                            78: 22, 79: 9, 80: 9, 81: 22, 82: 13, 83: 0,\
                            84: 0, 85: 13, 86: 22, 87: 9, 88: 9, 89: 22, \
                            90: 23, 91: 10, 92: 10, 93: 23, 94: 24, 95: \
                            11, 96: 11, 97: 24, 98: 25, 99: 12, 100: 12,\
                            101: 25, 102: 14, 103: 1, 104: 1, 105: 14,\
                            106: 13, 107: 0}, \
                   'face': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: \
                            1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: 2, \
                            14: 2, 15: 2, 16: 2, 17: 2, 18: 3, 19: 3, 20:\
                            3, 21: 3, 22: 3, 23: 3, 24: 4, 25: 4, 26: 4, \
                            27: 4, 28: 4, 29: 4, 30: 5, 31: 5, 32: 5, 33:\
                            5, 34: 5, 35: 5, 36: 6, 37: 6, 38: 6, 39: 6, \
                            40: 7, 41: 7, 42: 7, 43: 7, 44: 8, 45: 8, 46:\
                            8, 47: 8, 48: 9, 49: 9, 50: 9, 51: 9, 52: 10,\
                            53: 10, 54: 10, 55: 10, 56: 11, 57: 11, 58: 11,\
                            59: 11, 60: 12, 61: 12, 62: 12, 63: 12, 64: \
                            13, 65: 13, 66: 13, 67: 13, 68: 14, 69: 14,\
                            70: 14, 71: 14, 72: 15, 73: 15, 74: 15, 75: \
                            15, 76: 16, 77: 16, 78: 16, 79: 16, 80: 17, \
                            81: 17, 82: 17, 83: 17, 84: 18, 85: 18, 86: \
                            18, 87: 18, 88: 19, 89: 19, 90: 19, 91: 19, \
                            92: 20, 93: 20, 94: 20, 95: 20, 96: 21, 97: \
                            21, 98: 21, 99: 21, 100: 22, 101: 22, 102: 22,\
                            103: 22, 104: 23, 105: 23, 106: 23, 107: 23}}

    dict_after = {'srce': {0: 0, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0, 7:\
                           6, 8: 7, 9: 8, 10: 9, 11: 1, 12: 12, 13: 13,\
                           14: 14, 15: 15, 16: 16, 17: 10, 18: 16, 19:\
                           17, 20: 18, 21: 19, 22: 11, 23: 10, 24: 2, \
                           25: 0, 26: 10, 27: 12, 28: 3, 29: 2, 30: 12,\
                           31: 13, 32: 4, 33: 3, 34: 13, 35: 14, 36: 5,\
                           37: 4, 38: 14, 39: 15, 40: 6, 41: 5, 42: 15,\
                           43: 16, 44: 0, 45: 6, 46: 16, 47: 10, 48: 6,\
                           49: 0, 50: 10, 51: 16, 52: 7, 53: 6, 54: 16,\
                           55: 17, 56: 8, 57: 7, 58: 17, 59: 18, 60: 9,\
                           61: 8, 62: 18, 63: 19, 64: 1, 65: 9, 66: 19,\
                           67: 11, 68: 0, 69: 1, 70: 11, 71: 10}, \
                  'trgt': {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 0, 6: 6, 7:\
                           7, 8: 8, 9: 9, 10: 1, 11: 0, 12: 10, 13: 12,\
                           14: 13, 15: 14, 16: 15, 17: 16, 18: 10, 19: \
                           16, 20: 17, 21: 18, 22: 19, 23: 11, 24: 0, \
                           25: 10, 26: 12, 27: 2, 28: 2, 29: 12, 30: 13,\
                           31: 3, 32: 3, 33: 13, 34: 14, 35: 4, 36: 4, \
                           37: 14, 38: 15, 39: 5, 40: 5, 41: 15, 42: 16,\
                           43: 6, 44: 6, 45: 16, 46: 10, 47: 0, 48: 0,\
                           49: 10, 50: 16, 51: 6, 52: 6, 53: 16, 54: \
                           17, 55: 7, 56: 7, 57: 17, 58: 18, 59: 8, 60:\
                           8, 61: 18, 62: 19, 63: 9, 64: 9, 65: 19, 66:\
                           11, 67: 1, 68: 1, 69: 11, 70: 10, 71: 0},\
                  'face': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, \
                           7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 2, 13: \
                           2, 14: 2, 15: 2, 16: 2, 17: 2, 18: 3, 19:\
                           3, 20: 3, 21: 3, 22: 3, 23: 3, 24: 4, 25:\
                           4, 26: 4, 27: 4, 28: 5, 29: 5,30: 5, 31: 5,\
                           32: 6, 33: 6, 34: 6, 35: 6, 36: 7, 37: 7, \
                           38: 7, 39: 7, 40: 8, 41: 8, 42: 8, 43: 8, 44:\
                           9, 45: 9, 46: 9, 47: 9, 48: 10, 49: 10,\
                           50: 10, 51: 10, 52: 11, 53: 11, 54: 11, 55:\
                           11, 56: 12, 57: 12, 58: 12, 59: 12, 60: 13,\
                           61: 13, 62: 13, 63: 13, 64: 14, 65: 14, 66:\
                           14, 67: 14, 68: 15, 69: 15, 70: 15, 71: 15}}

    sft_before = pd.DataFrame.from_dict(dict_before)
    sft_after = pd.DataFrame.from_dict(dict_after)

    eptm.remove([0])
    eptm.update_mindex()

    assert eptm.edge_df[['srce','trgt','face']].equals(sft_after[['srce','trgt','face']])

    

def test_cut_out():
   datasets_2d, specs = three_faces_sheet()
   datasets = extrude(datasets_2d,method='translation')
   
   eptm = Epithelium('3faces_3D', datasets)
   eptm_ordered = eptm.copy(deep_copy=True)
   
   bounding_box_xy = np.array([[-1.,10.],[-1.5,1.5]])
   bounding_box_yx = np.array([[-1.5,1.5],[-1.0,10.]])
   bounding_box_xyz = np.array([[-10.,10.],[-1.5,10.],[-2.,1.]])

   expected_index_xy = pd.Index([ 2,  3,  4,  7,  8,  9, 10, 13, 14, 15,\
                                  20, 21, 22, 25, 26, 27, 28, 31, 32, \
                                  33, 44, 46, 47, 48, 49, 50, 51, 52, \
                                  53, 54, 64, 66, 67, 68, 69, 70, 71, \
                                  72, 73, 74, 75, 76, 77, 78, 88, 90, \
                                  91, 92, 93, 94, 95, 96, 97, 98],\
                                name='edge',dtype='int64')
   
   expected_index_xyz = pd.Index([13, 14, 15, 31, 32, 33, 88, 90, 91, \
                                  92, 93, 94, 95, 96, 97, 98],\
                                 name='edge',dtype='int64') 
   
   # test 2-coords, ordered
   res=eptm.cut_out(bbox=bounding_box_xy,coords=['x','y'])
   assert len(res) == len(expected_index_xy)
   assert (res == expected_index_xy).all()

   # test 2-coords, inverse order
   res = eptm.cut_out(bbox=bounding_box_yx,coords=['y','x'])
   assert len(res) == len(expected_index_xy)
   assert (res == expected_index_xy).all()
   
   # test 3-coords
   res = eptm.cut_out(bbox=bounding_box_xyz,coords=['x','y','z'])
   assert len(res) == len(expected_index_xyz) 
   assert (res == expected_index_xyz ).all()
   
   # test default coords argument
   res = eptm.cut_out(bbox=bounding_box_xy)
   assert len(res) == len(expected_index_xy)
   assert (res == expected_index_xy).all()
   
   

def test_vertex_mesh():
    datasets = {}
    tri_verts = [[0, 0, 0],
                 [1, 0, 0],
                 [-0.5, 0.86, 1.],
                 [-0.5, -0.86, 1.]]
    
    tri_edges = [[0, 1, 0, 0],
                 [1, 2, 0, 0],
                 [2, 0, 0, 0],
                 [0, 3, 1, 0],
                 [3, 1, 1, 0],
                 [1, 0, 1, 0],
                 [0, 2, 2, 0],
                 [2, 3, 2, 0],
                 [3, 0, 2, 0]]
    
    datasets['edge'] = pd.DataFrame(data=np.array(tri_edges),
                                    columns=['srce', 'trgt', 'face','cell'])
    datasets['edge'].index.name = 'edge'
    
    datasets['face'] = pd.DataFrame(data=np.zeros((3, 3)),
                                    columns=['x', 'y', 'z'])
    datasets['face'].index.name = 'face'
    
    datasets['vert'] = pd.DataFrame(data=np.array(tri_verts),
                                    columns=['x', 'y', 'z'])
    datasets['vert'].index.name = 'vert'
    
    specs = config.geometry.flat_sheet()
    
    eptm = Epithelium('vertex_mesh', datasets, specs, coords=['x', 'y', 'z'])
    SheetGeometry.update_all(eptm)

    ## tested method
    res_verts, res_faces, res_normals = eptm.vertex_mesh(['x','y','z'])
    res_xy_verts, res_xy_faces = eptm.vertex_mesh(['x','y','z'],vertex_normals=False)
    res_faces = list(res_faces)

        
    expected_faces = [[0, 1, 2], [0, 3, 1], [0, 2, 3]]
    

    ## floating point precision might causes issues here
    ## when comparing arrays ... there seems to be
    ## a built-in 1e-10 tolerance in
    ## the assert_array_equal function.
    
    expected_normals = np.array([[1.911111111e-01, 9.25185854e-18, 2.866666667e-01],
                                 [-4.16333634e-17, 0.0, 2.866666667e-01],
                                 [2.866666667e-01, -1.666666667e-01, 2.866666667e-01],
                                 [2.866666667e-01, 1.666666667e-01, 2.866666667e-01]])
    
    assert_array_equal(res_verts, np.array(tri_verts))
    assert all([res_faces[i] == expected_faces[i] for i in range(len(expected_faces))])
    assert_array_equal(np.round(res_normals, decimals=6), np.round(expected_normals, decimals=6))
    
    

def test_ordered_edges():
    # test _ordered_edges
    # also test ordered_vert_idxs
    datasets, specs = three_faces_sheet(zaxis=True)
    eptm = Epithelium('ordered_index', datasets, specs)    
    
    res_edges_2d = _ordered_edges(eptm.edge_df.loc[eptm.edge_df['face'] == 0])
    expected_edges_2d = [[0, 1, 0], [1, 2, 0], [2, 3, 0], [3, 4, 0], [4, 5, 0], [5, 0, 0]]
    expected_vert_idxs = [idxs[0] for idxs in expected_edges_2d]
    assert res_edges_2d == expected_edges_2d
    assert expected_vert_idxs == ordered_vert_idxs(eptm.edge_df.loc[eptm.edge_df['face'] == 0])
    res_invalid_face = ordered_vert_idxs(eptm.edge_df.loc[eptm.edge_df['face'] == 98765])
    
    ## testing the exception case in ordered_vert_idxs :
    res_invalid_face = ordered_vert_idxs(eptm.edge_df.loc[eptm.edge_df['face'] == 98765])
    assert np.isnan(res_invalid_face)


