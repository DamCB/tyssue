import tempfile

from tyssue import Sheet
from tyssue.generation import three_faces_sheet
from tyssue.io import csv


def test_write_storm():
    sheet = Sheet("test", *three_faces_sheet())
    fh = tempfile.mktemp(suffix=".csv")
    csv.write_storm_csv(fh, sheet.vert_df[sheet.coords])
    with open(fh) as fb:
        lines = fb.readlines()
    assert len(lines) == 14
    assert "frame" in lines[0]
