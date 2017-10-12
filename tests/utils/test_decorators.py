from tyssue import Epithelium
from tyssue.utils.decorators import do_undo, validate
from tyssue.core.generation import three_faces_sheet

@do_undo
def bad_action(eptm):
    eptm.edge_df['srce'] = 0
    print(eptm.edge_df.srce.max())
    raise Exception

def test_do_undo():
    eptm = Epithelium('t', *three_faces_sheet())
    max_srce = eptm.edge_df.srce.max()
    try:
        bad_action(eptm)
    except:
        assert eptm.edge_df.srce.max() == max_srce
