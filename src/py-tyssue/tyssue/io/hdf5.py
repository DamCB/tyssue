
import pandas as pd
import json

def load_epithelium(h5store, data_names=['cell', 'jv', 'je']):
    with pd.get_store(h5store) as store:
        data = {name: store[name] for name in data_names}
    return data

def save_dfs(h5store, eptm):
    with pd.get_store(h5store) as store:
        for key in eptm.data_names:
            store[key] = key
