import os
import tempfile
from tyssue.config.json_parser import load_spec, save_spec
from tyssue import config
import pytest


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
    with pytest.raises(IOError):
        save_spec(config, tmp.name, False)


def test_default():
    spec = config.geometry.cylindrical_sheet()
    assert spec['face']['x'] == 0.0
