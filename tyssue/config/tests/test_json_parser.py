import os
import tempfile
from tyssue.config.json_parser import load_spec, save_spec
from tyssue import config
from nose.tools import assert_raises




CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
TESTCONFIG = os.path.join(CURRENT_DIR, 'test_config.json')


def test_load_spec():
    config = load_spec(TESTCONFIG)
    assert 'face' in config
    assert config['face']['num_sides'] == 6

def test_save_spec():
    config = load_spec(TESTCONFIG)
    tmp = tempfile.NamedTemporaryFile()
    save_spec(config, tmp.name, overwrite=True)
    saved_config = load_spec(tmp.name)
    assert saved_config['face']['num_sides'] == 6
    assert_raises(IOError, save_spec, config, tmp.name, False)

def test_default():
    spec = config.geometry.sheet_spec()
    assert spec['face']['x'] == 0.0
