import pandas as pd
import logging
logger = logging.getLogger(name=__name__)


def load_datasets(h5store, data_names=['face', 'vert', 'edge']):
    transl = {'face': 'face',
              'vert': 'jv',
              'edge': 'je'}
    rewrite = False
    with pd.get_store(h5store) as store:
        try:
            data = {name: store[name] for name in data_names}
        except KeyError:
            logger.warning('Old names file, will be rewritten')
            data = {name: store[transl[name]] for name in data_names}
            rewrite = True
    if rewrite:
        with pd.get_store(h5store) as store:
            for key, val in data.items():
                store.put(key, val)

    return data

def save_datasets(h5store, eptm):
    with pd.get_store(h5store) as store:
        for key in eptm.data_names:
            store.put(key, getattr(eptm, '{}_df'.format(key)))
