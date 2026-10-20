"""Stage-1 errors."""


class Stage1Error(RuntimeError):
    """A step can't run on its inputs; the preprocessing scripts print it and exit 1."""
