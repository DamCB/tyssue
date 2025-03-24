import time

from functools import wraps


def do_undo(func):
    """Decorator that creates a copy of the first argument
    (usually an epithelium object) and restores it if the function fails.

    The first argument in `*args` should have `backup()` and `restore()` methods.
    """

    @wraps(func)
    def with_bckup(*args, **kwargs):
        eptm = args[0]
        eptm.backup()
        try:
            res = func(*args, **kwargs)
            return res
        except Exception as err:
            eptm.restore()
            raise err

    return with_bckup


def validate(func):
    """Decorator that validate the epithelium after the
    decorated function was applied. the first argument
    of `func` should be an epithelium instance, and
    is at least assumed to have a `validate` method.
    """

    @wraps(func)
    def with_validate(*args, **kwargs):
        eptm = args[0]
        result = func(*args, **kwargs)
        if not eptm.validate():
            raise ValueError(
                """
An invalid epithelium was produced

To see which edges are invalid, you can inspect
the 'is_valid' column of the `edge_df` dataframe,
or for example the bad cells involved:

>>> bad_edges = eptm.edge_df[~eptm.edge_df['is_valid']].index
>>> bad_cells = eptm.edge_df.loc[bad_edges, 'cell'].unique()

If case the epithelium was restored after being invalidated, you can find the
invalid epithelium as the `_bad` attribute of the restored one"""
            )

        return result

    return with_validate


def time_exe(func):
    def with_time_exe(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        print("function : {} \ttime: {2:2f}sec".format(func.__name__, end - start))

        return result

    return with_time_exe


def face_lookup(func):
    @wraps(func)
    def with_face_lookup(*args, **kwargs):
        sheet = args[0]
        face_id = kwargs["face_id"]
        face = sheet.idx_lookup(face_id, "face")
        if face is None:
            return
        kwargs["face"] = face
        return func(*args, **kwargs)

    return with_face_lookup


def cell_lookup(func):
    @wraps(func)
    def with_cell_lookup(*args, **kwargs):
        sheet = args[0]
        cell_id = kwargs["cell_id"]
        cell = sheet.idx_lookup(cell_id, "cell")
        if cell is None:
            return
        kwargs["cell"] = cell
        return func(*args, **kwargs)

    return with_cell_lookup
