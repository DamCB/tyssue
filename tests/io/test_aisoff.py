import tempfile
from tyssue.generation import three_faces_sheet
from tyssue import Sheet
from tyssue.io import ais


def test_aisstring():
    sheet = Sheet('test', *three_faces_sheet())
    lines = ais.ais_string(sheet.vert_df[sheet.coords].values,
                           sheet.edge_df[['srce', 'trgt']].values).split('\n')

    assert len(lines) == 3
    assert lines[0] == '13 13'
