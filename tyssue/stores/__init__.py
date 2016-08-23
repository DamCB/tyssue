import os
from ..io import hdf5

def load_datasets(store, **kwargs):
    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # root = os.path.dirname(cur_dir)

    filename = os.path.join(cur_dir, store)
    return hdf5.load_datasets(filename, **kwargs)
