def do_undo(func):
    """Decorator that creates a copy of the first argument
    (usually an epithelium object) and restores it if the function fails.

    The first argument in `*args` should have a `backup()` method.
    """
    def with_bckup(*args, **kwargs):
        eptm = args[0]
        eptm.backup()
        try:
            return func(*args, **kwargs)
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
    def with_validate(*args, **kwargs):
        eptm = args[0]
        result = func(*args, **kwargs)
        if not eptm.validate():
            raise ValueError('An invalid epithelium was produced')
        return result
    return with_validate
