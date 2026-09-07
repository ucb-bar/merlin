from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from merlin.perf import model_placement as P


SOURCE = '''builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %0 : tensor<2xi32>
  }
}'''


class _Attr:
    def __init__(self, value):
        self.data = value


class _Op:
    def __init__(self, region):
        self.attributes = {"prov.region_id": _Attr(region)}


@dataclass(frozen=True)
class _Shape:
    op: str
    parallel: tuple[int, ...]
    reduction: tuple[int, ...]


def test_placement_is_weighted_by_macs_not_region_count(monkeypatch) -> None:
    monkeypatch.setattr(P, "observe_contractions", lambda _path: [
        (_Op("large"), _Shape("linalg.matmul", (64, 64), (64,))),
        (_Op("small"), _Shape("linalg.matmul", (1, 16), (16,))),
    ])

    got = P.contraction_placement("model.mlir", [
        {"region": "large", "lane": "engine_a"},
        {"region": "small", "lane": "engine_b"},
    ])

    assert got["status"] == "complete"
    assert got["macs_by_lane"] == {"engine_a": 64 * 64 * 64, "engine_b": 1 * 16 * 16}
    assert got["mac_fraction_by_lane"]["engine_a"] > 0.99
    assert got["licence"].startswith("exact structural")


def test_missing_region_placement_stays_explicit(monkeypatch) -> None:
    monkeypatch.setattr(P, "observe_contractions", lambda _path: [
        (_Op("known"), _Shape("linalg.matmul", (16, 16), (16,))),
        (_Op("missing"), _Shape("linalg.matmul", (8, 8), (8,))),
    ])

    got = P.contraction_placement(
        "model.mlir", [{"region": "known", "lane": "engine"}])

    assert got["status"] == "partial"
    assert got["unresolved_placement_macs"] == 8 * 8 * 8
    assert got["unresolved"][0]["region"] == "missing"


def test_prepared_and_legacy_graph_evidence_are_identical(tmp_path: Path) -> None:
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = P.prepare_captured_source(source)

    assert P.captured_global_graph(
        source, prepared_source=prepared) == P.captured_global_graph(source)


def test_prepared_source_refuses_file_identity_drift(tmp_path: Path) -> None:
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = P.prepare_captured_source(source)
    source.write_text(SOURCE + "\n")

    with pytest.raises(ValueError, match="does not match immutable source bytes"):
        P.captured_global_graph(source, prepared_source=prepared)


def test_prepared_source_refuses_graph_digest_drift(tmp_path: Path) -> None:
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = P.prepare_captured_source(source)
    prepared.graph.entry = "changed_after_preparation"

    with pytest.raises(ValueError, match="graph changed after host construction"):
        P.captured_global_graph(source, prepared_source=prepared)


def test_placement_consumes_the_prepared_module(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "model.mlir"
    source.write_text(SOURCE)
    prepared = P.prepare_captured_source(source)
    observed_sources = []

    def observe(parsed):
        observed_sources.append(parsed)
        return []

    monkeypatch.setattr(P, "observe_contractions", observe)
    got = P.contraction_placement(source, [], prepared_source=prepared)

    assert got["status"] == "UNKNOWN"
    assert observed_sources == [prepared.parsed_module]
