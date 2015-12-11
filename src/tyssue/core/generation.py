import pandas as pd
import numpy as np



###
data_dicts = {
    'cell': {
        ## Cell Geometry
        'perimeter': (0., np.float),
        'area': (0., np.float),
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        'z': (0., np.float),
        ## Topology
        'num_sides': (6, np.int),
        ## Masks
        'is_alive': (1, np.bool)},
    'jv': {
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        'z': (0., np.float),
        ## Masks
        'is_active': (1, np.bool)},
    'je': {
        ## associated elements indexes
        'srce': (0, np.int),
        'trgt': (0, np.int),
        'cell': (0, np.int),
        ## Coordinates
        'dx': (0., np.float),
        'dy': (0., np.float),
        'dz': (0., np.float),
        'length': (0., np.float),
        ### Normals
        'nx': (0., np.float),
        'ny': (0., np.float),
        'nz': (0., np.float)}
    }

###
data_dicts2d = {
    'cell': {
        ## Cell Geometry
        'perimeter': (0., np.float),
        'area': (0., np.float),
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        ## Topology
        'num_sides': (6, np.int),
        ## Masks
        'is_alive': (1, np.bool)},
    'jv': {
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        ## Masks
        'is_active': (1, np.bool)},
    'je': {
        ## associated elements indexes
        'srce': (0, np.int),
        'trgt': (0, np.int),
        'cell': (0, np.int),
        ## Coordinates
        'dx': (0., np.float),
        'dy': (0., np.float),
        'length': (0., np.float),
        ### Normals
        'nz': (0., np.float)}
    }


def three_cells_sheet_array():
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
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Cells have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    cell_df: the cells `DataFrame` indexed from 0 to 2
    jv_df: the junction vertices `DataFrame`
    je_df: the junction edges `DataFrame`

    '''
    points, _, (Nc, Nv, Ne) = three_cells_sheet_array()

    if zaxis:
        coords = ['x', 'y', 'z']
    else:
        coords = ['x', 'y']

    cell_idx = pd.Index(range(Nc), name='cell')
    jv_idx = pd.Index(range(Nv), name='jv')

    _je_e_idx = np.array([[0, 1, 0],
                          [1, 2, 0],
                          [2, 3, 0],
                          [3, 4, 0],
                          [4, 5, 0],
                          [5, 0, 0],
                          [0, 5, 1],
                          [5, 6, 1],
                          [6, 7, 1],
                          [7, 8, 1],
                          [8, 9, 1],
                          [9, 0, 1],
                          [0, 9, 2],
                          [9, 10, 2],
                          [10, 11, 2],
                          [11, 12, 2],
                          [12, 1, 2],
                          [1, 0, 2]])

    je_idx = pd.Index(range(_je_e_idx.shape[0]), name='je')

    ### Cell - cell graph
    cc_idx = [(0, 1), (1, 2), (0, 2)]
    cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['cella', 'cellb'])
    ### Cells DataFrame
    cell_df = make_df(index=cell_idx, data_dict=data_dicts['cell'])

    ### Junction vertices and edges DataFrames
    jv_df = make_df(index=jv_idx, data_dict=data_dicts['jv'])
    je_df = make_df(index=je_idx, data_dict=data_dicts['je'])
    je_df['srce'] = _je_e_idx[:, 0]
    je_df['trgt'] = _je_e_idx[:, 1]
    je_df['cell'] = _je_e_idx[:, 2]

    jv_df.loc[:, coords[:2]] = points
    if zaxis:
        jv_df.loc[:, coords[2:]] = 0.

    datasets = {'cell': cell_df, 'jv': jv_df, 'je': je_df}
    return datasets


def make_df(index, data_dict):
    '''
    Creates a pd.DataFrame indexed by `index` with
    the `data_dict` keys as column names, filled with the
    value given by the `data_dict` values.

    '''
    dtypes = np.dtype([(name, val[1]) for name, val in data_dict.items()])
    N = len(index)
    arr = np.empty(N, dtype=dtypes)
    df = pd.DataFrame.from_records(arr, index=index)
    for name, val in data_dict.items():
        df[name] = val[0]
    return df
