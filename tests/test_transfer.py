from dataclasses import replace

import numpy as np
import pytest
import torch

from run_tcn import HazardTCN
from src.transfer import (
    PretrainConfig,
    initialize_hazard_encoder,
    pretrain_masked_tcn,
    subject_holdout,
)


def _external_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(17)
    subjects = np.repeat(np.arange(8), 2)
    values = rng.normal(size=(len(subjects), 3, 12)).astype(np.float32)
    observed = rng.random(values.shape) > 0.10
    values[~observed] = np.nan
    return values, observed, subjects


def test_subject_holdout_is_deterministic_and_subject_disjoint():
    subjects = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    left = subject_holdout(subjects, validation_fraction=0.25, seed=7)
    right = subject_holdout(subjects, validation_fraction=0.25, seed=7)
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    training, validation = left
    assert set(subjects[training]).isdisjoint(subjects[validation])


def test_checkpoint_initializes_only_hazard_encoder(tmp_path):
    values, observed, subjects = _external_sequences()
    checkpoint = tmp_path / 'external_ssl.pt'
    config = PretrainConfig(epochs=2, batch_size=4, patience=0, seed=31)
    result = pretrain_masked_tcn(values, observed, subjects, checkpoint, config)

    target = HazardTCN(values.shape[1])
    hazard_before = {name: value.clone() for name, value in target.hazard.state_dict().items()}
    metadata = initialize_hazard_encoder(target, checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)

    assert result.epochs_completed == 2
    assert np.isfinite(result.best_validation_loss)
    assert metadata['data_sha256'] == saved['data_sha256']
    for name, value in target.encoder.state_dict().items():
        torch.testing.assert_close(value, saved['encoder_state'][name])
    for name, value in target.hazard.state_dict().items():
        torch.testing.assert_close(value, hazard_before[name])


def test_resume_matches_uninterrupted_training(tmp_path):
    values, observed, subjects = _external_sequences()
    complete = tmp_path / 'complete.pt'
    resumed = tmp_path / 'resumed.pt'
    config = PretrainConfig(epochs=2, batch_size=4, patience=0, seed=43)

    pretrain_masked_tcn(values, observed, subjects, complete, config)
    pretrain_masked_tcn(values, observed, subjects, resumed, replace(config, epochs=1))
    pretrain_masked_tcn(values, observed, subjects, resumed, config, resume=True)
    expected = torch.load(complete, map_location='cpu', weights_only=False)
    actual = torch.load(resumed, map_location='cpu', weights_only=False)

    assert actual['history'] == pytest.approx(expected['history'])
    for name, value in expected['training_encoder_state'].items():
        torch.testing.assert_close(
            actual['training_encoder_state'][name], value, rtol=0, atol=0
        )


def test_observed_values_must_be_finite(tmp_path):
    values, observed, subjects = _external_sequences()
    values[0, 0, 0] = np.nan
    observed[0, 0, 0] = True
    with pytest.raises(ValueError, match='finite'):
        pretrain_masked_tcn(values, observed, subjects, tmp_path / 'bad.pt')


def test_unavailable_channel_uses_neutral_stats_and_safe_initialization(tmp_path):
    values, observed, subjects = _external_sequences()
    observed[:, 2, :] = False
    values[:, 2, :] = np.nan
    checkpoint = tmp_path / 'missing_channel.pt'
    config = PretrainConfig(epochs=1, batch_size=4, patience=0, seed=59)

    pretrain_masked_tcn(values, observed, subjects, checkpoint, config)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    target = HazardTCN(values.shape[1])
    missing_weight_before = target.encoder[0].weight[:, 2, :].detach().clone()
    metadata = initialize_hazard_encoder(target, checkpoint)

    assert saved['normalisation']['median'][2] == 0
    assert saved['normalisation']['mean'][2] == 0
    assert saved['normalisation']['std'][2] == 1
    assert metadata['observed_coverage'][2] == 0
    assert metadata['training_observed_coverage'][2] == 0
    torch.testing.assert_close(
        target.encoder[0].weight[:, 2, :], missing_weight_before
    )
    torch.testing.assert_close(
        target.encoder[0].weight[:, :2, :],
        saved['encoder_state']['0.weight'][:, :2, :],
    )
