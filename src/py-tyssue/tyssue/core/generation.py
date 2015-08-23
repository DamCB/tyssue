import pandas as pd
import numpy as np

###
cell_data = {
    ## Cell Geometry
    'perimeter': (0., np.float),
    'area': (0., np.float),
    ## Coordinates
    'x': (0., np.float),
    'y': (0., np.float),
    'z': (0., np.float),
    ## Topology
    'num_sides': (1, np.int),
    ## Masks
    'is_alive': (1, np.bool)}

jv_data = {
    ## Coordinates
    'x': (0., np.float),
    'y': (0., np.float),
    'z': (0., np.float),
    ## Masks
    'is_active': (1, np.bool)}

je_data = {
    ## Coordinates
    'dx': (0., np.float),
    'dy': (0., np.float),
    'dz': (0., np.float),
    'length': (0., np.float),
    ### Normal
    'n': (1., np.float)}



def three_cells_sheet_array(zaxis=False):
    '''
    Creates the apical junctions mesh of three packed hexagonal cells.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates, with `z = 0`.

    Cells have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    points: (13, ndim) np.array of floats
      the positions, where ndim is 2 or 3 depending on `zaxis`
    edges: (15, 2)  np.array of ints
      indices of the edges
    (Nc, Nv, Ne): triple of ints
      number of cells, vertices and edges (3, 13, 15)

    '''


    Nc = 3 # Number of cells

    points = np.array([[0., 0.],
                       [1.0, 0.0],
                       [1.5, 0.866],
                       [1.0, 1.732],
                       [0.0, 1.732],
                       [-0.5, 0.866],
                       [-1.5, 0.866],
                       [-2, 0.],
                       [-1.5, -0.866],
                       [-0.5, -0.866],
                       [0, -1.732],
                       [1, -1.732],
                       [1.5, -0.866]])

    edges = np.array([[0, 1],
                      [1, 2],
                      [2, 3],
                      [3, 4],
                      [4, 5],
                      [5, 0],
                      [5, 6],
                      [6, 7],
                      [7, 8],
                      [8, 9],
                      [9, 0],
                      [9, 10],
                      [10, 11],
                      [11, 12],
                      [12, 1]])

    Nv, Ne = len(points), len(edges)
    return points, edges, (Nc, Nv, Ne)


def three_cells_sheet(zaxis=False):
    '''
    Creates the apical junctions mesh of three packed hexagonal cells.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates, with `z = 0`.

    Cells have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    cell_df: the cells `DataFrame` indexed from 0 to 2
    jv_df: the junction vertices `DataFrame`
    je_df: the junction edges `DataFrame`

    '''
    points, edges, (Nc, Nv, Ne) = three_cells_sheet_array(zaxis)

    if zaxis:
        coords = ['x', 'y', 'z']
    else:
        coords = ['x', 'y']

    cell_idx = pd.Index(range(Nc), name='cell')
    jv_idx = pd.Index(range(Nv), name='jv')

    _je_idx = [(0, 1, 0),
               (1, 2, 0),
               (2, 3, 0),
               (3, 4, 0),
               (4, 5, 0),
               (5, 0, 0),
               (0, 5, 1),
               (5, 6, 1),
               (6, 7, 1),
               (7, 8, 1),
               (8, 9, 1),
               (9, 0, 1),
               (0, 9, 2),
               (9, 10, 2),
               (10, 11, 2),
               (11, 12, 2),
               (12, 1, 2),
               (1, 0, 2)]

    je_idx = pd.MultiIndex.from_tuples(_je_idx, names=['srce', 'trgt', 'cell'])

    ### Cell - cell graph
    cc_idx = [(0, 1), (1, 2), (0, 2)]
    cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['cella', 'cellb'])
    ### Cells DataFrame
    cell_df = make_df(index=cell_idx, data_dict=cell_data)

    ### Junction vertices and edges DataFrames
    jv_df = make_df(index=jv_idx, data_dict=jv_data)
    je_df = make_df(index=je_idx, data_dict=je_data)

    jv_df[coords] = points
    return cell_df, jv_df, je_df



def make_df(index, data_dict):
    '''
    Creates a pd.DataFrame indexed by `index` with
    the `data_dict` keys as column names, filled with the
    value given by the `data_dict` values.

    '''
    #### See this pandas issue on how to specify mixed
    #### dtypes at instentiation time:

    dtypes = np.dtype([(name, val[1]) for name, val in data_dict.items()])
    N = len(index)
    arr = np.empty(N, dtype=dtypes)
    df = pd.DataFrame.from_records(arr, index=index)
    for name, val in data_dict.items():
        df[name] = val[0]
    return df
