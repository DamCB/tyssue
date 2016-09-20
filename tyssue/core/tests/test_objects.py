import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from tyssue.core import Epithelium
from tyssue.core.generation import three_faces_sheet
from tyssue.core.objects import get_opposite
from tyssue import config
from tyssue.geometry.planar_geometry import PlanarGeometry
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
    
    ## TODO : test with an open face
    ## should raise an exception   

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
    pass

def test_cut_out():
    pass

def test_reset_index():
    pass

def test_vertex_mesh():
    pass

def test_ordered_edges():
    # test _ordered_edges
    # also test orderd_vert_idxs line 718
    pass


