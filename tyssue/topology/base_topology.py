import logging
import pandas as pd
from itertools import combinations

logger = logging.getLogger(name=__name__)

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


def close_face(eptm, face):

    face_edges = eptm.edge_df[eptm.edge_df['face'] == face]
    srces = set(face_edges['srce'])
    trgts = set(face_edges['trgt'])

    if srces == trgts:
        logger.info('Face {} already closed'.format(face))
        return
    try:
        single_srce, = srces.difference(trgts)
        single_trgt, = trgts.difference(srces)
    except ValueError as err:
        print('Closing only possible with exactly two dangling vertices')
        raise err

    eptm.edge_df = eptm.edge_df.append(
        face_edges.iloc[0],
        ignore_index=True)
    eptm.edge_df.index.name = 'edge'
    new_edge = eptm.edge_df.index[-1]
    eptm.edge_df.loc[new_edge, ['srce', 'trgt']] = single_trgt, single_srce


def condition_4i(eptm):
    """
    Return an index over the faces violating condition 4 i in Okuda et al 2013,
    that is edges (from the same face) sharing two vertices simultaneously.
    """
    num_srces = eptm.edge_df.groupby('face')['srce'].apply(lambda s: len(set(s)))
    num_sides = eptm.face_df['num_sides']
    return eptm.face_df[num_srces != num_sides].index


def get_neighbour_face_pairs(eptm):
    """
    Returns a pandas Series of neighboring face pairs (as forzen sets of 2 indexes)
    """
    pairs = []
    eptm.edge_df['v_pair'] =  eptm.edge_df[['srce', 'trgt']].apply(frozenset, axis=1)

    _ = eptm.edge_df.groupby('v_pair')['face'].apply(
        lambda s: pairs.extend([frozenset((a, b)) for a, b in combinations(s.values, 2)]))
    return pd.Series(pairs).drop_duplicates()

def get_num_common_edges(eptm):
    """
    Returns the number of common edges between two neighboring faces
    this number is set to -1 if those faces are opposite and share the
    same edges.
    """
    pairs = get_neighbour_face_pairs(eptm)
    face_v_pair_orbit = eptm.edge_df.groupby('face').apply(
        lambda df: frozenset(df['v_pair']))
    n_common = [
        len(face_v_pair_orbit.loc[fa].intersection(face_v_pair_orbit.loc[fb]))
        if face_v_pair_orbit.loc[fb] != face_v_pair_orbit.loc[fa]
        else -1 for fa, fb in pairs]
    n_common = pd.Series(n_common, index=pd.Index(pairs, name='face_pairs'))
    return n_common

def condition_4ii(eptm):
    """
    Return a list of face pairs sharing more than one edge, as defined
    in Okuda et al. 2013 condition 4 ii
    """
    n_common = get_num_common_edges(eptm)
    return list(n_common[n_common >= 2].index)
