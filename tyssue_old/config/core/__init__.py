from ..json_parser import load_spec
import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))


def bulk_spec():
    """
    Default settings to perform edge subdivisions,
    see tyssue.io.point_cloud.py
    """
    specfile = os.path.join(CURRENT_DIR, "bulk.json")
    return load_spec(specfile)
