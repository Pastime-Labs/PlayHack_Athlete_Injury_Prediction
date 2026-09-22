from pathlib import Path

import pytest

import acquire_external
from acquire_external import (
    RUNFITCHECK_SHA256,
    RUNFITCHECK_URL,
    _select_pmdata_items,
    acquire_pmdata,
    acquire_runfitcheck,
)


def _item(name: str, kind: str) -> dict:
    return {
        "attributes": {"name": name, "kind": kind},
        "links": {"download": f"https://example.invalid/{name}"},
    }


def test_pmdata_selection_records_optional_files_and_folders():
    steps = _item("steps.json", "file")
    selected, missing = _select_pmdata_items(
        "p12",
        "fitbit",
        {"steps.json", "lightly_active_minutes.json"},
        {"steps.json": steps},
    )
    _, missing_folder = _select_pmdata_items("p12", "pmsys", {"srpe.csv"}, None)

    assert selected == [steps]
    assert missing == [{
        "participant": "p12",
        "folder": "fitbit",
        "name": "lightly_active_minutes.json",
    }]
    assert missing_folder == [{"participant": "p12", "folder": "pmsys", "name": None}]


def _mock_pmdata(monkeypatch, fitbit_files: list[dict]) -> list[Path]:
    storage = _item("osfstorage", "folder")
    pmdata = _item("pmdata", "folder")
    license_file = _item("license.txt", "file")
    participant = _item("p12", "folder")
    fitbit = _item("fitbit", "folder")
    children = {
        "osfstorage": [pmdata, license_file],
        "pmdata": [participant],
        "p12": [fitbit],
        "fitbit": fitbit_files,
    }
    monkeypatch.setattr(acquire_external, "_paged", lambda _: [storage])
    monkeypatch.setattr(
        acquire_external,
        "_children",
        lambda item: children[item["attributes"]["name"]],
    )
    downloaded = []

    def fake_download(url, target, force):
        downloaded.append(target)
        return {
            "url": url,
            "path": str(target),
            "bytes": 1,
            "sha256": "digest",
            "status": "existing",
        }

    monkeypatch.setattr(acquire_external, "_download", fake_download)
    return downloaded


def test_acquire_pmdata_keeps_available_files_and_records_missing(tmp_path, monkeypatch):
    downloaded = _mock_pmdata(monkeypatch, [_item("steps.json", "file")])
    manifest = acquire_pmdata(tmp_path, force=False)

    assert [path.name for path in downloaded] == ["license.txt", "steps.json"]
    assert manifest["participants"] == 1
    assert {
        "participant": "p12",
        "folder": "fitbit",
        "name": "lightly_active_minutes.json",
    } in manifest["missing_files"]
    assert {"participant": "p12", "folder": "pmsys", "name": None} in manifest["missing_files"]


def test_acquire_pmdata_requires_a_compatible_fitbit_file(tmp_path, monkeypatch):
    _mock_pmdata(monkeypatch, [_item("unrelated.json", "file")])
    with pytest.raises(RuntimeError, match="no compatible selected Fitbit"):
        acquire_pmdata(tmp_path, force=False)


def test_acquire_runfitcheck_pins_source_and_checksum(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, target, force):
        calls.append((url, target, force))
        return {"url": url, "path": str(target), "bytes": 1, "sha256": RUNFITCHECK_SHA256}

    monkeypatch.setattr(acquire_external, "_download", fake_download)
    manifest = acquire_runfitcheck(tmp_path, force=False)

    assert calls == [(
        RUNFITCHECK_URL,
        tmp_path / "raw" / "runfitcheck" / "RFC Analysebestand.csv",
        False,
    )]
    assert manifest["persistent_id"] == "10.4121/12895664.v1"
    assert manifest["license"] == "CC0"
