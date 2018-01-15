from tyssue.generation import extrude, three_faces_sheet
from tyssue import Monolayer, config


def test_copy():

    datasets, specs = three_faces_sheet()
    extruded = extrude(datasets, method='translation')
    mono = Monolayer('test', extruded,
                     config.geometry.bulk_spec())
    assert mono.Nc == 3
    assert mono.Nf == 24
    assert mono.Ne == 108
    assert mono.Nv == 26

    mono2 = mono.copy()

    assert mono2.Nc == 3
