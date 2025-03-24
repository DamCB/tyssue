import os
import warnings
from ..io import hdf5

stores_dir = os.path.abspath(os.path.dirname(__file__))
stores_list = os.listdir(stores_dir)

__doc__ = """Available predefined datasets:""" + "\n".join(stores_list)


def load_datasets(store, **kwargs):
    """
    This will soon be deprecated, use the leaner
    `tyssue.stores.stores_dir and tyssue.stores.stores_list
    """

    cur_dir = os.path.abspath(os.path.dirname(__file__))
    # root = os.path.dirname(cur_dir)
    warnings.warn(
        """
    This will soon be deprecated, use the leaner
    `tyssue.stores.stores_dir and tyssue.stores.stores_list
    """
    )
    filename = os.path.join(cur_dir, store)
    return hdf5.load_datasets(filename, **kwargs)
