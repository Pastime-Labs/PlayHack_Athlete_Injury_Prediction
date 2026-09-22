from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from run_tcn import HazardTCN


FORMAT_VERSION = 2


@dataclass(frozen=True)
class PretrainConfig:
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 8e-4
    weight_decay: float = 2e-3
    mask_probability: float = 0.20
    validation_fraction: float = 0.20
    patience: int = 8
    seed: int = 2026
    device: str = 'cpu'


@dataclass(frozen=True)
class PretrainResult:
    checkpoint: Path
    epochs_completed: int
    best_validation_loss: float
    train_subjects: int
    validation_subjects: int


class MaskedReconstructionTCN(nn.Module):
    # HazardTCN-compatible encoder with a temporary reconstruction head.
    def __init__(self, channels: int):
        super().__init__()
        self.encoder = HazardTCN(channels).encoder
        self.decoder = nn.Conv1d(32, channels, kernel_size=1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(values))


def subject_holdout(
    subject_ids: np.ndarray,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    subjects = np.asarray(subject_ids).astype(str)
    if subjects.ndim != 1:
        raise ValueError('subject_ids must be one-dimensional')
    unique = np.unique(subjects)
    if len(unique) < 2:
        raise ValueError('At least two distinct subjects are required')
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    rng = np.random.default_rng(seed)
    held_count = min(len(unique) - 1, max(1, round(len(unique) * validation_fraction)))
    held_subjects = set(rng.permutation(unique)[:held_count])
    validation = np.flatnonzero(np.isin(subjects, list(held_subjects)))
    training = np.flatnonzero(~np.isin(subjects, list(held_subjects)))
    return training, validation


def _validate_inputs(
    values: np.ndarray,
    observed: np.ndarray,
    subject_ids: np.ndarray,
    config: PretrainConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values)
    observed = np.asarray(observed)
    subjects = np.asarray(subject_ids)
    if values.ndim != 3 or min(values.shape) == 0:
        raise ValueError('values must have non-empty shape [N, C, T]')
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError('values must use a floating dtype')
    if observed.shape != values.shape or observed.dtype != np.bool_:
        raise ValueError('observed must be a boolean mask with the same shape as values')
    if subjects.ndim != 1 or len(subjects) != len(values):
        raise ValueError('subject_ids must contain one ID per sequence')
    if not np.isfinite(values[observed]).all():
        raise ValueError('Observed values must be finite')
    if not observed.any():
        raise ValueError('At least one observed channel is required')
    if not observed.reshape(len(observed), -1).any(axis=1).all():
        raise ValueError('Every sequence needs at least one observed value')
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError('epochs and batch_size must be positive')
    if not 0 < config.mask_probability < 1:
        raise ValueError('mask_probability must be between zero and one')
    if config.learning_rate <= 0 or config.weight_decay < 0 or config.patience < 0:
        raise ValueError('Invalid optimizer or patience configuration')
    return values.astype(np.float32, copy=False), observed, subjects


def _fingerprint(values: np.ndarray, observed: np.ndarray, subjects: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(values.shape).encode())
    digest.update(json.dumps(subjects.astype(str).tolist()).encode())
    digest.update(np.ascontiguousarray(observed).tobytes())
    visible = np.where(observed, values, 0).astype('<f4', copy=False)
    digest.update(np.ascontiguousarray(visible).tobytes())
    return digest.hexdigest()


def _normalise(
    values: np.ndarray,
    observed: np.ndarray,
    training: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    channel_count = values.shape[1]
    median = np.zeros(channel_count, dtype=np.float32)
    mean = np.zeros(channel_count, dtype=np.float32)
    std = np.ones(channel_count, dtype=np.float32)
    for channel in range(channel_count):
        channel_observed = observed[training, channel]
        if channel_observed.any():
            visible = values[training, channel][channel_observed]
            median[channel] = np.median(visible)
            mean[channel] = np.mean(visible)
            channel_std = np.std(visible)
            std[channel] = channel_std if channel_std > 1e-6 else 1.0
    filled = np.where(observed, values, median[None, :, None])
    normalised = (filled - mean[None, :, None]) / std[None, :, None]
    normalised[~observed] = 0.0
    return normalised.astype(np.float32), {
        'median': median.astype(np.float32),
        'mean': mean.astype(np.float32),
        'std': std.astype(np.float32),
    }


def _corruption_mask(
    observed: np.ndarray,
    probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = observed & (rng.random(observed.shape) < probability)
    if not selected.any():
        candidates = np.flatnonzero(observed)
        if not len(candidates):
            raise ValueError('A batch contains no observed values to reconstruct')
        selected.flat[int(rng.choice(candidates))] = True
    return selected


def _state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _config_signature(config: PretrainConfig) -> dict[str, object]:
    signature = asdict(config)
    signature.pop('epochs')
    return signature


def _save_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def _validation_loss(
    model: MaskedReconstructionTCN,
    values: np.ndarray,
    corruption: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            target_np = values[start : start + batch_size]
            selected_np = corruption[start : start + batch_size]
            masked_np = target_np.copy()
            masked_np[selected_np] = 0.0
            target = torch.from_numpy(target_np).to(device)
            selected = torch.from_numpy(selected_np).to(device)
            prediction = model(torch.from_numpy(masked_np).to(device))
            total += float(
                F.smooth_l1_loss(prediction[selected], target[selected], reduction='sum')
            )
            count += int(selected.sum())
    return total / count


def pretrain_masked_tcn(
    values: np.ndarray,
    observed: np.ndarray,
    subject_ids: np.ndarray,
    checkpoint: str | Path,
    config: PretrainConfig = PretrainConfig(),
    *,
    resume: bool = False,
) -> PretrainResult:
    # This API accepts no labels or PlayHack bundle, keeping pretraining external-only.
    values, observed, subjects = _validate_inputs(values, observed, subject_ids, config)
    training, validation = subject_holdout(subjects, config.validation_fraction, config.seed)
    if not observed[training].any() or not observed[validation].any():
        raise ValueError('Training and validation partitions both need observed values')
    normalised, statistics = _normalise(values, observed, training)
    observed_coverage = observed.mean(axis=(0, 2)).astype(np.float32)
    training_observed_coverage = observed[training].mean(axis=(0, 2)).astype(np.float32)
    fingerprint = _fingerprint(values, observed, subjects)
    split_hash = hashlib.sha256(training.tobytes() + validation.tobytes()).hexdigest()

    device = torch.device(config.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(config.seed)

    model = MaskedReconstructionTCN(values.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    rng = np.random.default_rng(config.seed)
    validation_rng = np.random.default_rng(config.seed + 1)
    validation_corruption = _corruption_mask(
        observed[validation], config.mask_probability, validation_rng
    )
    path = Path(checkpoint)
    start_epoch, stale_epochs = 0, 0
    history: list[dict[str, float | int]] = []
    best_loss = float('inf')
    best_encoder = _state(model.encoder)

    if resume and path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get('format_version') != FORMAT_VERSION:
            raise ValueError('Unsupported checkpoint format')
        if saved.get('data_sha256') != fingerprint or saved.get('split_sha256') != split_hash:
            raise ValueError('Checkpoint data or subject split does not match')
        if saved.get('config_signature') != _config_signature(config):
            raise ValueError('Checkpoint training configuration does not match')
        model.encoder.load_state_dict(saved['training_encoder_state'])
        model.decoder.load_state_dict(saved['decoder_state'])
        optimizer.load_state_dict(saved['optimizer_state'])
        torch.set_rng_state(saved['torch_rng_state'].cpu())
        if device.type == 'cuda' and saved.get('cuda_rng_state') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng_state'])
        rng.bit_generator.state = saved['numpy_rng_state']
        start_epoch = int(saved['epochs_completed'])
        stale_epochs = int(saved['stale_epochs'])
        history = list(saved['history'])
        best_loss = float(saved['best_validation_loss'])
        best_encoder = saved['encoder_state']

    for epoch in range(start_epoch, config.epochs):
        model.train()
        loss_sum, masked_count = 0.0, 0
        batches = np.array_split(
            rng.permutation(training),
            max(1, int(np.ceil(len(training) / config.batch_size))),
        )
        for batch_indices in batches:
            target_np = normalised[batch_indices]
            selected_np = _corruption_mask(
                observed[batch_indices], config.mask_probability, rng
            )
            masked_np = target_np.copy()
            masked_np[selected_np] = 0.0
            target = torch.from_numpy(target_np).to(device)
            selected = torch.from_numpy(selected_np).to(device)
            prediction = model(torch.from_numpy(masked_np).to(device))
            loss = F.smooth_l1_loss(prediction[selected], target[selected])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            count = int(selected.sum())
            loss_sum += float(loss.detach()) * count
            masked_count += count

        validation_loss = _validation_loss(
            model,
            normalised[validation],
            validation_corruption,
            config.batch_size,
            device,
        )
        history.append({
            'epoch': epoch + 1,
            'train_loss': loss_sum / masked_count,
            'validation_loss': validation_loss,
        })
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_encoder = _state(model.encoder)
            stale_epochs = 0
        else:
            stale_epochs += 1

        payload = {
            'format_version': FORMAT_VERSION,
            'channels': values.shape[1],
            'config': asdict(config),
            'config_signature': _config_signature(config),
            'data_sha256': fingerprint,
            'split_sha256': split_hash,
            'epochs_completed': epoch + 1,
            'stale_epochs': stale_epochs,
            'best_validation_loss': best_loss,
            'history': history,
            'normalisation': {
                name: torch.from_numpy(value) for name, value in statistics.items()
            },
            'observed_coverage': torch.from_numpy(observed_coverage),
            'training_observed_coverage': torch.from_numpy(training_observed_coverage),
            'encoder_state': best_encoder,
            'training_encoder_state': _state(model.encoder),
            'decoder_state': _state(model.decoder),
            'optimizer_state': optimizer.state_dict(),
            'numpy_rng_state': rng.bit_generator.state,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': (
                torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
            ),
        }
        _save_checkpoint(path, payload)
        if config.patience and stale_epochs >= config.patience:
            break

    completed = history[-1]['epoch'] if history else start_epoch
    return PretrainResult(
        checkpoint=path,
        epochs_completed=int(completed),
        best_validation_loss=best_loss,
        train_subjects=len(np.unique(subjects[training].astype(str))),
        validation_subjects=len(np.unique(subjects[validation].astype(str))),
    )


def initialize_hazard_encoder(
    model: HazardTCN,
    checkpoint: str | Path,
) -> dict[str, object]:
    # Load only the best external encoder; the supervised hazard head stays fresh.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('format_version') != FORMAT_VERSION:
        raise ValueError('Unsupported checkpoint format')
    first_weight = model.encoder.state_dict()['0.weight']
    if int(saved['channels']) != first_weight.shape[1]:
        raise ValueError(
            'Channel mismatch: checkpoint has {}, model expects {}'.format(
                saved['channels'], first_weight.shape[1]
            )
        )
    encoder_state = {
        name: value.clone() for name, value in saved['encoder_state'].items()
    }
    unavailable = saved['training_observed_coverage'].eq(0)
    if unavailable.any():
        encoder_state['0.weight'][:, unavailable, :] = (
            first_weight.detach().cpu()[:, unavailable, :]
        )
    model.encoder.load_state_dict(encoder_state, strict=True)
    return {
        'data_sha256': saved['data_sha256'],
        'split_sha256': saved['split_sha256'],
        'best_validation_loss': saved['best_validation_loss'],
        'epochs_completed': saved['epochs_completed'],
        'normalisation': saved['normalisation'],
        'observed_coverage': saved['observed_coverage'],
        'training_observed_coverage': saved['training_observed_coverage'],
    }
