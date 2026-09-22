import numpy as np

from run_e2_stacker import _logit_features


def test_stacker_logit_features_are_finite_and_monotonic():
    values = _logit_features(
        np.array([0.0, 0.2, 0.8, 1.0]),
        np.array([0.1, 0.3, 0.7, 0.9]),
    )

    assert np.isfinite(values).all()
    assert np.all(np.diff(values[:, 0]) > 0)
    assert np.all(np.diff(values[:, 1]) > 0)
