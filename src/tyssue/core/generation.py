import pandas as pd
import numpy as np



###
data_dicts = {
    'face': {
        ## Face Geometry
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
        'face': (0, np.int),
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
    'face': {
        ## Face Geometry
        'perimeter': (0., np.float),
        'area': (0., np.float),
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        ## Topology
        'num_sides': (6, np.int),
        ## Masks
        'is_alive': (1, np.bool)
        },
    'jv': {
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        ## Masks
        'is_active': (1, np.bool)
        },
    'je': {
        ## associated elements indexes
        'srce': (0, np.int),
        'trgt': (0, np.int),
        'face': (0, np.int),
        ## Coordinates
        'dx': (0., np.float),
        'dy': (0., np.float),
        'length': (0., np.float),
        ### Normals
        'nz': (0., np.float)
        }
    }

data_dicts3d = {
    'cell': {
        ## Face Geometry
        'vol': (0., np.float),
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        'z': (0., np.float),
        ## Topology
        'num_faces': (6, np.int),
        ## Masks
        'is_alive': (1, np.bool)
        },
    'face': {
        ## Face Geometry
        'perimeter': (0., np.float),
        'area': (0., np.float),
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        'z': (0., np.float),
        ## Topology
        'num_sides': (6, np.int),
        ## Masks
        'is_alive': (1, np.bool)
        },
    'jv': {
        ## Coordinates
        'x': (0., np.float),
        'y': (0., np.float),
        'z': (0., np.float),
        ## Masks
        'is_active': (1, np.bool)
        },
    'je': {
        ## associated elements indexes
        'srce': (0, np.int),
        'trgt': (0, np.int),
        'face': (0, np.int),
        'cell': (0, np.int),
        ## Coordinates
        'dx': (0., np.float),
        'dy': (0., np.float),
        'dz': (0., np.float),
        'length': (0., np.float),
        ### Normals
        'nx': (0., np.float),
        'ny': (0., np.float),
        'nz': (0., np.float)
        }
    }



def three_faces_sheet_array():
    '''
    Creates the apical junctions mesh of three packed hexagonal faces.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Faces have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    points: (13, ndim) np.array of floats
      the positions, where ndim is 2 or 3 depending on `zaxis`
    edges: (15, 2)  np.array of ints
      indices of the edges
    (Nc, Nv, Ne): triple of ints
      number of faces, vertices and edges (3, 13, 15)

    '''


    Nc = 3 # Number of faces

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


def three_faces_sheet(zaxis=False):
    '''
    Creates the apical junctions mesh of three packed hexagonal faces.
    If `zaxis` is `True` (defaults to False), adds a `z` coordinates,
    with `z = 0`.

    Faces have a side length of 1.0 +/- 1e-3.

    Returns
    -------

    face_df: the faces `DataFrame` indexed from 0 to 2
    jv_df: the junction vertices `DataFrame`
    je_df: the junction edges `DataFrame`

    '''
    points, _, (Nc, Nv, Ne) = three_faces_sheet_array()

    if zaxis:
        coords = ['x', 'y', 'z']
    else:
        coords = ['x', 'y']

    face_idx = pd.Index(range(Nc), name='face')
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

    ### Face - face graph
    cc_idx = [(0, 1), (1, 2), (0, 2)]
    cc_idx = pd.MultiIndex.from_tuples(cc_idx, names=['facea', 'faceb'])
    ### Faces DataFrame
    face_df = make_df(index=face_idx, data_dict=data_dicts['face'])

    ### Junction vertices and edges DataFrames
    jv_df = make_df(index=jv_idx, data_dict=data_dicts['jv'])
    je_df = make_df(index=je_idx, data_dict=data_dicts['je'])
    je_df['srce'] = _je_e_idx[:, 0]
    je_df['trgt'] = _je_e_idx[:, 1]
    je_df['face'] = _je_e_idx[:, 2]

    jv_df.loc[:, coords[:2]] = points
    if zaxis:
        jv_df.loc[:, coords[2:]] = 0.

    datasets = {'face': face_df, 'jv': jv_df, 'je': je_df}
    return datasets, data_dicts


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


def hexa_grid3d(nx, ny, nz, distx=1., disty=1., distz=1., noise=None):
    """Creates an hexagonal grid of points
    """
    cz, cy, cx = np.mgrid[0:nz, 0:ny, 0:nx]
    cx = cx.astype(np.float)
    cy = cy.astype(np.float)
    cz = cz.astype(np.float)
    cx[::2, :] += 0.5
    cy[::2, :] += 0.5

    centers = np.vstack([cx.flatten(),
                         cy.flatten(),
                         cz.flatten()]).T
    centers[:, 0] *= distx
    centers[:, 1] *= disty
    centers[:, 2] *= distz
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers


def from_3d_voronoi(voro):
    el_idx = []

    for f_idx, (rv, rp) in enumerate(
        zip(voro.ridge_vertices,
            voro.ridge_points)):

        if -1 in rv:
            continue
        face_verts = voro.vertices[rv]
        f_center = face_verts.mean(axis=0)
        c0 = voro.points[rp[0]]
        ctof = f_center - c0

        for rv0, rv1 in zip(rv, np.roll(rv, 1, axis=0)):
            fv0 = voro.vertices[rv0]
            fv1 = voro.vertices[rv1]
            edge_v = fv1 - fv0
            fto0 = fv0 - f_center
            normal = np.cross(fto0, edge_v)
            dotp = np.dot(ctof, normal)
            if np.sign(dotp) > 0:
                el_idx.append([rv0, rv1, f_idx, rp[0]])
                el_idx.append([rv1, rv0, f_idx, rp[1]])
            else:
                el_idx.append([rv1, rv0, f_idx, rp[0]])
                el_idx.append([rv0, rv1, f_idx, rp[1]])

    el_idx = np.array(el_idx)

    coords = ['x', 'y', 'z']
    je_idx = pd.Index(range(el_idx.shape[0]), name='je')
    je_df = make_df(je_idx, data_dicts3d['je'])

    for i, elem in enumerate(['srce', 'trgt', 'face', 'cell']):
        je_df[elem] = el_idx[:, i]

    jv_idx = pd.Index(range(voro.vertices.shape[0]), name='jv')
    jv_df = make_df(jv_idx, data_dicts3d['jv'])
    jv_df[coords] = voro.vertices

    cell_idx = pd.Index(range(voro.points.shape[0]), name='cell')
    cell_df = make_df(cell_idx, data_dicts3d['cell'])
    cell_df[coords] = voro.points

    nfaces = len(voro.ridge_vertices)
    face_idx = pd.Index(np.arange(nfaces), name='face')
    face_df = make_df(face_idx, data_dicts3d['cell'])
    je_df.sort_values(by='cell', inplace=True)

    datasets= {
        'jv': jv_df,
        'je': je_df,
        'face': face_df,
        'cell': cell_df,
        }
    return datasets


def hexa_grid2d(nh, nv, disth, distv, noise=None):
    """Creates an hexagonal grid of points
    """
    cy, cx = np.mgrid[0:nv, 0:nh]
    cx = cx.astype(np.float)
    cy = cy.astype(np.float)
    cx[::2, :] += 0.5

    centers = np.vstack([cx.flatten(),
                         cy.flatten()]).astype(np.float).T
    centers[:, 0] *= disth
    centers[:, 1] *= distv
    if noise is not None:
        pos_noise = np.random.normal(scale=noise, size=centers.shape)
        centers += pos_noise
    return centers
