import numpy as np
import pandas as pd


def apoptosis_time_table(sheet,
                         apoptotic_cell,
                         events,
                         start_t=0):

    settings = sheet.settings['apoptosis']
    shrink_steps = settings['shrink_steps']
    rad_tension = settings['rad_tension']
    contractile_increase = settings['contractile_increase']
    contract_span = settings['contract_span']

    apoptotic_edges = sheet.edge_df[sheet.edge_df['face'] == apoptotic_cell]

    n_sides = apoptotic_edges.shape[0]
    # Number of type 1 transitions to solve the rosette
    n_type1 = n_sides - 3
    end_shrink = start_t + shrink_steps
    end_t = start_t + shrink_steps + n_type1

    times = range(start_t, end_t)
    shrink_times = range(start_t, end_shrink)

    cell_time_idx = pd.MultiIndex.from_tuples(
        [(t, apoptotic_cell) for t in times],
        names=['t', 'face'])

    time_table = pd.DataFrame(index=cell_time_idx,
                              columns=events.keys())

    pref_vols = np.linspace(1., 0., shrink_steps)
    time_table.loc[start_t: end_shrink-1, 'shrink'] = pref_vols

    rad_tensions = np.linspace(0, rad_tension, shrink_steps)
    time_table.loc[start_t: end_shrink-1, 'ab_pull'] = rad_tensions

    time_table.loc[end_shrink: end_t-1, 'type1_at_shorter'] = 1
    time_table.loc[end_t, 'type3'] = 1

    neighbors = sheet.get_neighborhood(apoptotic_cell, contract_span)
    nb_t_idx = pd.MultiIndex.from_product([shrink_times, neighbors['face']],
                                          names=['t', 'face'])

    contracts = np.linspace(1, contractile_increase,
                            shrink_steps).repeat(neighbors.shape[0])
    contracts = contracts.reshape((shrink_steps,
                                   neighbors.shape[0]))
    contracts = contracts / np.atleast_2d(neighbors.order.values)
    time_table = pd.concat([time_table,
                            pd.DataFrame(contracts.ravel(),
                                         index=nb_t_idx,
                                         columns=['contract', ])])

    return times, time_table.sort_index()
