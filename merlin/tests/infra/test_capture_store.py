from __future__ import annotations

import json
from pathlib import Path

from merlin.perf import capture_store as STORE


def _sha(char: str) -> str:
    return char * 64


def _pins() -> dict[str, dict[str, str]]:
    return {
        "gsim_binary": {"sha256": _sha("1")},
        "gsim_firrtl": {"sha256": _sha("2")},
        "gsim_model": {"sha256": _sha("3")},
        "verilator_binary": {"sha256": _sha("4")},
        "verilator_firrtl": {"sha256": _sha("5")},
    }


def _capture(workload: str, reference: dict[str, object]) -> dict[str, object]:
    pins = _pins()
    return {
        "elf_sha256": _sha("a"),
        "workload_sha256": workload,
        "semantic_reference": reference,
        "reference": {
            "binary_sha256": pins["verilator_binary"]["sha256"],
            "firrtl_sha256": pins["verilator_firrtl"]["sha256"],
        },
        "candidate": {
            "binary_sha256": pins["gsim_binary"]["sha256"],
            "firrtl_sha256": pins["gsim_firrtl"]["sha256"],
            "model_sha256": pins["gsim_model"]["sha256"],
        },
    }


def _temporary_store(monkeypatch, tmp_path: Path) -> Path:
    root = tmp_path / "captures"
    root.mkdir()
    monkeypatch.setattr(STORE, "store_root", lambda _target: root)
    return root


def test_same_elf_distinct_workload_bindings_do_not_overwrite(monkeypatch, tmp_path):
    root = _temporary_store(monkeypatch, tmp_path)
    first = _capture(_sha("b"), {"output": [1]})
    second = _capture(_sha("c"), {"output": [2]})

    first_path = STORE.store("gemmini", elf_sha256=_sha("a"), pins=_pins(), document=first)
    second_path = STORE.store("gemmini", elf_sha256=_sha("a"), pins=_pins(), document=second)

    assert first_path is not None and second_path is not None and first_path != second_path
    assert len(list(root.glob("*.json"))) == 2
    assert STORE.lookup(
        "gemmini", elf_sha256=_sha("a"), pins=_pins(),
        workload_sha256=_sha("b"), semantic_reference={"output": [1]}) == first
    assert STORE.lookup(
        "gemmini", elf_sha256=_sha("a"), pins=_pins(),
        workload_sha256=_sha("c"), semantic_reference={"output": [2]}) == second


def test_matching_legacy_entry_is_promoted_to_scoped_key(monkeypatch, tmp_path):
    root = _temporary_store(monkeypatch, tmp_path)
    document = _capture(_sha("b"), {"output": [1]})
    legacy = STORE.capture_key(_sha("a"), _pins())
    assert legacy is not None
    (root / f"{legacy}.json").write_text(json.dumps(document), encoding="utf-8")

    assert STORE.lookup(
        "gemmini", elf_sha256=_sha("a"), pins=_pins(),
        workload_sha256=_sha("b"), semantic_reference={"output": [1]}) == document

    scoped = STORE.capture_key(
        _sha("a"), _pins(), workload_sha256=_sha("b"),
        semantic_reference={"output": [1]})
    assert scoped is not None and scoped != legacy
    assert (root / f"{scoped}.json").is_file()


def test_model_pin_is_part_of_hit_validation(monkeypatch, tmp_path):
    _temporary_store(monkeypatch, tmp_path)
    document = _capture(_sha("b"), {"output": [1]})
    document["candidate"]["model_sha256"] = _sha("f")

    assert STORE.store(
        "gemmini", elf_sha256=_sha("a"), pins=_pins(), document=document) is None
