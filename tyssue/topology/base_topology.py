
def add_vert(eptm, edge):
    """
    Adds a vertex in the middle of the edge,
    which is split as is its opposite
    """

    srce, trgt = eptm.edge_df.loc[edge, ['srce', 'trgt']]
    opposites = eptm.edge_df[(eptm.edge_df['srce'] == trgt) &
                             (eptm.edge_df['trgt'] == srce)]
    parallels = eptm.edge_df[(eptm.edge_df['srce'] == srce) &
                             (eptm.edge_df['trgt'] == trgt)]

    new_vert = eptm.vert_df.loc[[srce, trgt]].mean()
    eptm.vert_df = eptm.vert_df.append(new_vert, ignore_index=True)
    new_vert = eptm.vert_df.index[-1]
    new_edges = []

    for p, p_data in parallels.iterrows():
        eptm.edge_df.loc[p, 'trgt'] = new_vert
        eptm.edge_df = eptm.edge_df.append(p_data, ignore_index=True)
        new_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_edge, 'srce'] = new_vert
        eptm.edge_df.loc[new_edge, 'trgt'] = trgt
        new_edges.append(new_edge)

    new_opp_edges = []
    for o, o_data in opposites.iterrows():
        eptm.edge_df.loc[o, 'srce'] = new_vert
        eptm.edge_df = eptm.edge_df.append(o_data, ignore_index=True)
        new_opp_edge = eptm.edge_df.index[-1]
        eptm.edge_df.loc[new_opp_edge, 'trgt'] = new_vert
        eptm.edge_df.loc[new_opp_edge, 'srce'] = trgt
        new_opp_edges.append(new_opp_edge)

    # ## Sheet special case
    if len(new_edges) == 1:
        new_edges = new_edges[0]
    if len(new_opp_edges) == 1:
        new_opp_edges = new_opp_edges[0]
    elif len(new_opp_edges) == 0:
        new_opp_edges = None
    return new_vert, new_edges, new_opp_edges
