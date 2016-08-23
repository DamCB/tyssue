'''
Generic forces and energies
'''


def elastic_force(element_df, var, elasticity, prefered):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    force = element_df.eval('{K} * ({x} - {x0})'.format(**params))
    return force


def elastic_energy(element_df, var, elasticity, prefered):
    params = {'x': var,
              'K': elasticity,
              'x0': prefered}
    energy = element_df.eval(
        '0.5 * {K} * ({x} - {x0}) ** 2'.format(**params))
    return energy
