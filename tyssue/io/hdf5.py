import os
import pandas as pd
import logging
logger = logging.getLogger(name=__name__)


def load_datasets(h5store, data_names=['face', 'vert', 'edge']):
    if not os.path.isfile(h5store):
        raise FileNotFoundError('file %s not found' % h5store)
    with pd.HDFStore(h5store) as store:
        data = {name: store[name] for name in data_names}
    return data


def save_datasets(h5store, eptm):
    with pd.HDFStore(h5store) as store:
        for key in eptm.data_names:
            store.put(key, getattr(eptm, '{}_df'.format(key)))
