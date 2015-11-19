from tyssue.utils import utils

def test_update_default():
    a = {'a': 0, 'b': 2}
    b = utils.update_default(a, params=None)
    b["a"] = 1
    assert (a['a'] == 0)

    a = {'a': 0, 'b': 2}
    b = utils.update_default(a, params=a)
    b["a"] = 1
    assert (a['a'] == 0)

    a = {'a': 0, 'b': 2}
    c = {'c': 2, 'b': 1}
    b = utils.update_default(a, params=c)
    assert (a == {'a': 0, 'b': 2})
    assert (b == {'a': 0, 'b': 1, 'c':2})
    assert (c == {'c': 2, 'b': 1})
