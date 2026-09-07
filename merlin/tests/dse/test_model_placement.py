from __future__ import annotations

from dataclasses import dataclass

from merlin.perf import model_placement as P


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
