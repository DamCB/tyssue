from numpy.testing import assert_array_equal
import numpy as np

from tyssue.generation import extrude, three_faces_sheet
from tyssue import Monolayer, config, Sheet
from tyssue.core.monolayer import MonolayerWithLamina

def test_monolayer():

    sheet = Sheet('test', *three_faces_sheet())

    mono = Monolayer.from_flat_sheet(
        'test', sheet, config.geometry.bulk_spec())

    assert_array_equal(mono.apical_verts.values,
                       np.arange(13))
    assert_array_equal(mono.basal_verts.values,
                       np.arange(13)+13)

    assert_array_equal(mono.apical_edges.values,
                       np.arange(18))
    assert_array_equal(mono.basal_edges.values,
                       np.arange(18)+18)
    assert_array_equal(mono.sagittal_edges.values,
                       np.arange(72)+36)

    assert_array_equal(mono.apical_faces.values,
                       np.arange(3))
    assert_array_equal(mono.basal_faces.values,
                       np.arange(3)+3)
    assert_array_equal(mono.sagittal_faces.values,
                       np.arange(18)+6)


def test_monolayer_with_lamina():

    sheet_dsets, _ = three_faces_sheet()
    dsets = extrude(sheet_dsets, method='translation')
    mono = MonolayerWithLamina('test', dsets,
                               config.geometry.bulk_spec())
    assert mono.lamina_edges.shape == (3,)


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
    assert isinstance(mono2, Monolayer)
