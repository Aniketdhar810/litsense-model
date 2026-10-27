"""Evaluation errors. Kept out of ml/common/errors.py on purpose: every data stage imports that
module, so editing it would change their code fingerprints and rerun the whole data pipeline
(D41)."""


class LeakageError(ValueError):
    """An evaluation input is newer than its cutoff, or comes from another cutoff's artifact set."""


class DegenerateScoresError(ValueError):
    """A scorer returned a row whose finite scores all tie (the zero-user-vector bug, s2.2 #7)."""
