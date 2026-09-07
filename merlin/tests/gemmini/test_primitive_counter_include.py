"""Counter wrapper identity follows header bytes, not checkout location or shadow headers."""
from __future__ import annotations

import hashlib
import importlib
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.perf import hw_counters as hc
from merlin.runtime.backends.base import get_backend


@pytest.fixture
def bracket_environment(tmp_path, monkeypatch):
    backend = get_backend("gemmini")
    adapter = importlib.import_module(f"{backend.__name__}.gemmini_primitive_probe")
    codegen = importlib.import_module(f"{backend.__name__}.gemmini_codegen_mlir")
    header = (repo_root() / "merlin/experiments/capsule_bench/targets/gemmini/contracts/"
              "hwbringup_gemmini_v0/isa_include/gemmini_counter.h").read_bytes()
    digest = hashlib.sha256(header).hexdigest()
    roots = [tmp_path / name for name in ("original", "relocated")]
    for root in roots:
        (root / "include").mkdir(parents=True)
        (root / "include/gemmini_counter.h").write_bytes(header)
    discovery = {"status": "derived", "header": str(roots[0] / "include/gemmini_counter.h"),
                 "header_sha256": digest, "event_codes": hc.event_codes(header.decode())}
    monkeypatch.setattr(hc, "counters_for_target", lambda target: discovery)
    slots = len(hc.derive_occupancy_counters(header.decode()).by_combination)
    monkeypatch.setattr(codegen, "_counter_slots", lambda: {"status": "derived", "slots": slots})
    # Hardware proof is orthogonal to this include-resolution regression. Keep actual header
    # digest validation, event extraction, bracket generation and filesystem search unmocked.
    monkeypatch.setattr(adapter, "get_backend", lambda target: SimpleNamespace(
        counter_partition_inputs=lambda: {"status": "available", "hw_text": "unused",
                                          "module": "unused", "counter_module": "unused", "source": "test"}))
    monkeypatch.setattr(hc, "prove_occupancy_partition_from_circt",
                        lambda *args, **kwargs: {"status": "proved", "scope": "test prerequisite only"})
    work = tmp_path / "work"
    work.mkdir()
    return adapter, roots, work, discovery


def test_identical_relocated_header_produces_identical_bracket(bracket_environment):
    adapter, roots, work, discovery = bracket_environment
    first = adapter._occupancy_bracket([roots[0]], work)
    discovery["header"] = str(roots[1] / "include/gemmini_counter.h")
    second = adapter._occupancy_bracket([roots[1]], work)
    assert first[1:] == second[1:]
    assert first[1] == '#include "include/gemmini_counter.h"'
    assert first[0]["compiled_header"] != second[0]["compiled_header"]
    assert first[0]["header_sha256"] == second[0]["header_sha256"]


@pytest.mark.parametrize("shadow_location", ["workdir", "earlier_include_root"])
def test_first_shadow_header_mismatch_refused(bracket_environment, shadow_location, tmp_path):
    adapter, roots, work, _ = bracket_environment
    shadow = work if shadow_location == "workdir" else tmp_path / "shadow"
    (shadow / "include").mkdir(parents=True)
    (shadow / "include/gemmini_counter.h").write_text("/* incompatible earlier header */\n")
    include_roots = roots if shadow_location == "workdir" else [shadow, *roots]
    with pytest.raises(ValueError, match="build recipe counter header differs from discovered event schema"):
        adapter._occupancy_bracket(include_roots, work)
