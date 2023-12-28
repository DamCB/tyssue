import json

import numpy as np


def filter_settings(settings):

    filtered = settings.copy()
    for k, v in settings.items():
        if isinstance(v, np.ndarray):
            filtered[k] = v.tolist()
        try:
            json.dumps(filtered[k])
        except TypeError:
            filtered.pop(k)
    return filtered
