from pathlib import Path

import run_nested_transfer
from run_nested_transfer import _initialize_encoder, _parse_args


def test_initialization_mode_parsing_and_dispatch(monkeypatch):
    random_args = _parse_args(["--random-init"])
    external_args = _parse_args(["--checkpoint", "external.pt"])
    assert random_args.random_init and random_args.checkpoint is None
    assert not external_args.random_init and external_args.checkpoint == "external.pt"

    calls = []
    monkeypatch.setattr(
        run_nested_transfer,
        "initialize_hazard_encoder",
        lambda model, checkpoint: calls.append((model, checkpoint)),
    )
    model = object()
    _initialize_encoder(model, None)
    assert calls == []
    _initialize_encoder(model, Path("external.pt"))
    assert calls == [(model, Path("external.pt"))]
