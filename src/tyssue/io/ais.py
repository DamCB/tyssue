'''
Export data to the ais format


see:
http://doc.cgal.org/latest/Linear_cell_complex/group__PkgLinearCellComplexConstructions.html#gaa356d78601f8844476fc2e039f0df83e
'''


def ais_string(points, edges):
    '''

    Parameters
    ----------

    points: (Np, 3) sequence
       points positions
    edges: (Ne, 2)  sequence of pairs
       indices of the edges

    Returns
    -------

    ais: string
      the ais string as accepted by CGAL
      http://doc.cgal.org/latest/Linear_cell_complex/group__PkgLinearCellComplexConstructions.html#gaa356d78601f8844476fc2e039f0df83e

    '''

    ais = '\n'.join(
        ['{} {}'.format(Nv, Ne),
         ' '.join(['{} {}'.format(x, y) for (x, y) in points]),
         ' '.join(['{} {}'.format(srce, trgt) for (srce, trgt) in edges])
        ])

    return ais
