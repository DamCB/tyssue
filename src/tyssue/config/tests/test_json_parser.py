import os

from tyssue.config.json_parser import load_spec, save_spec


CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TESTCONFIG = os.path.join(CURRENT_DIR, 'test_config.json')



def test_load_spec():
    fname = TESTCONFIG
    config = load_spec(TESTCONFIG)
    assert 'face' in config
    assert config['face']['num_sides'] == 6
