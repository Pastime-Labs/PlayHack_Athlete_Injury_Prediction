import warnings

import numpy as np

from audit_external_domain import _summaries


def test_summaries_preserve_missing_periods_without_warnings():
    values = np.ones((2, 1, 10), dtype=float)
    observed = np.ones_like(values, dtype=bool)
    observed[0, 0, :-7] = False
    values[1, 0, -7:] = 3.0

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        summaries = _summaries(values, observed)

    assert np.isnan(summaries[0, 3])
    assert summaries[1, 3] == 2.0
