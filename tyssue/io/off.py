'''
Export data to the OFF format
'''


def off_string(points, edges):
    '''
    Parameters
    ----------

    points: (Np, 3) sequence
       points positions
    edges: (Ne, 2)  sequence of pairs
       indices of the edges

    Returns
    -------

    off_str: string
      content to be written to the OFF file.

    Note
    ----

    Bellow the spec. of the OFF file format:

    Line 1
        OFF
    Line 2
        vertex_count face_count edge_count
    One line for each vertex:
        x y z
        for vertex 0, 1, ..., vertex_count-1
    One line for each polygonal face:
        n v1 v2 ... vn,
        the number of vertices, and the vertex indices for each face.
    '''

    raise NotImplemented

    Nv = points.shape[0]
    Ne = edges.shape[0]
    Nf = len(set(edges[:, 2]))
    ndim = points.shape[1]

    off_str = 'OFF\n'
    off_str += '{} {} {}\n'.format(Nv, Nf, Ne)
    for p in points:
        if ndim == 2:
            off_str += '{} {} 0\n'.format(p[0], p[1])
        elif ndim == 3:
            off_str += '{} {} {}\n'.format(p[0], p[1], p[2])

    # TODO: fix that

    # off_str += '6 0 1 2 3 4 5\n'
    # off_str += '6 0 5 6 7 8 9\n'
    # off_str += '6 0 9 10 11 12 1\n'

    return off_str
