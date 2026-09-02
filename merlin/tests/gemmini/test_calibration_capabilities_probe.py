"""The Gemmini calibration capability probe only advertises demonstrated operations."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

from merlin.perf.calibration_plan import build_calibration_plan_from_rtl
from merlin.runtime.backends.base import get_backend


def _module():
    backend = get_backend("gemmini")
    return importlib.import_module(f"{backend.__name__}.calibration_capabilities")


def _facts() -> dict:
    return {"generator": {"tool": "synthetic CIRCT extractor"},
            "inputs": {"core_hw_sha256": "a" * 64}, "facts": {
        "source": "synthetic elaborated core", "arrays": [
            {"name": "mesh", "rows": 3, "cols": 5}],
        "datapaths": [{"name": "input", "dtype": "i8"}],
    }}


def _install_facts(monkeypatch, module, tmp_path: Path) -> Path:
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(_facts()), encoding="utf-8")
    monkeypatch.setattr(module, "ensure_facts", lambda _target, explicit=None: path)
    monkeypatch.setattr(module, "_core_circt_receipt",
                        lambda: {"path": "synthetic-core.hw.mlir", "sha256": "a" * 64})
    return path


def test_complete_native_emission_probes_establish_compute_dma_and_protocol_capabilities(
        monkeypatch, tmp_path: Path) -> None:
    module = _module()
    _install_facts(monkeypatch, module, tmp_path)
    backend = get_backend("gemmini")
    dma = importlib.import_module(f"{backend.__name__}.gemmini_dma_calibration")
    auxiliary = importlib.import_module(f"{backend.__name__}.gemmini_roofline_auxiliary")

    def emit(cb, *, stage):
        return {"status": "accepted", "stage": stage, "command_buffer": cb,
                "command_buffer_sha256": module._canonical_sha256(cb),
                "emitted_mlir_sha256": "b" * 64, "argument_order": []}

    protocols = [
        {"requested_cache_condition": "cold", "measurement_protocol": "fresh_elf_process",
         "cache_state": "unknown", "cache_state_observed": False},
        {"requested_cache_condition": "warm", "measurement_protocol": "one_unmeasured_predecessor",
         "cache_state": "unknown", "cache_state_observed": False},
    ]
    monkeypatch.setattr(module, "_emit_or_compile", emit)
    monkeypatch.setattr(module, "_measurement_protocols", lambda: (protocols, None))
    monkeypatch.setattr(dma, "derived_transfer_ladder", lambda _facts, *, points: (5, 10, 15, 20))
    monkeypatch.setattr(dma, "probe_dma_capability", lambda direction, size, _facts, *, stage: {
        "status": "accepted", "direction": direction, "requested_payload_bytes": size,
        "physical_traffic_bytes": {"status": "unmeasured", "value": None}, "stage": stage,
    })
    monkeypatch.setattr(auxiliary, "probe_empty_workload", lambda *_args, **_kwargs: {
        "status": "accepted", "artifact_sha256": "c" * 64})
    monkeypatch.setattr(auxiliary, "probe_joint_occupancy", lambda *_args, **_kwargs: {
        "status": "accepted", "artifact_sha256": "d" * 64})

    artifact = module.probe_calibration_capabilities(stage="emission")

    assert artifact["status"] == "complete"
    assert artifact["compute"]["workload_emitter"]["value"] is True
    assert artifact["compute"]["tile_multiples"]["value"] == [1, 2, 3, 4]
    assert artifact["dma"]["measurement_protocols"]["value"] == [
        "fresh_elf_process", "one_unmeasured_predecessor"]
    assert artifact["cache_states"]["status"] == "unknown"
    assert artifact["dma"]["directions"]["value"] == ["read", "write", "copy"]
    assert artifact["dma"]["read"]["sizes_bytes"]["value"] == [5, 10, 15, 20]
    assert all(row["physical_traffic_bytes"]["status"] == "unmeasured"
               for row in artifact["dma"]["direction_probes"].values())
    assert artifact["measurement_auxiliary"]["empty_workload_emitter"]["value"] is True
    assert artifact["measurement_auxiliary"]["joint_occupancy_probe"]["value"] is True
    assert artifact["artifact_sha256"] == module._canonical_sha256({
        key: value for key, value in artifact.items() if key != "artifact_sha256"})
    assert all(row["command_buffer"]["tensors"]["probe_input"]["shape"][1] == 5
               for row in artifact["probe_receipts"])
    # The artifact is passed through unchanged: all six DMA protocol/direction sweeps and compute are
    # now ready, while physical traffic remains a required runtime counter receipt.
    plan = build_calibration_plan_from_rtl(_facts(), artifact)
    assert plan.sweeps[-1].ready and len(plan.sweeps[-1].points) == 4
    assert all(sweep.ready for sweep in plan.sweeps)


def test_one_failed_probe_leaves_compute_unknown_instead_of_listing_a_partial_ladder(
        monkeypatch, tmp_path: Path) -> None:
    module = _module()
    _install_facts(monkeypatch, module, tmp_path)

    def emit(cb, *, stage):
        multiple = cb["tensors"]["probe_input"]["shape"][0] // 3
        return ({"status": "unknown", "why": "synthetic compiler rejection"}
                if multiple == 3 else {"status": "accepted"})

    monkeypatch.setattr(module, "_emit_or_compile", emit)
    monkeypatch.setattr(module, "_measurement_protocols", lambda: ([], "synthetic harness unavailable"))
    monkeypatch.setattr(module, "_dma_probe_requests", lambda _facts, _sizes, *, stage: (
        {direction: {"status": "accepted"} for direction in ("read", "write", "copy")},
        (5, 10, 15, 20), True, ""))

    artifact = module.probe_calibration_capabilities()

    assert artifact["status"] == "partial"
    assert artifact["compute"]["status"] == "unknown"
    assert "tile_multiples" not in artifact["compute"]
    assert artifact["dma"]["measurement_protocols"]["status"] == "unknown"


def test_stale_or_legacy_facts_cannot_establish_compute_capability(
        monkeypatch, tmp_path: Path) -> None:
    module = _module()
    facts = _facts()
    facts["inputs"]["core_hw_sha256"] = "a" * 16
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts), encoding="utf-8")
    monkeypatch.setattr(module, "ensure_facts", lambda _target, explicit=None: path)
    monkeypatch.setattr(module, "_core_circt_receipt",
                        lambda: {"path": "synthetic-core.hw.mlir", "sha256": "a" * 64})
    monkeypatch.setattr(module, "_emit_or_compile",
                        lambda cb, *, stage: (_ for _ in ()).throw(
                            AssertionError("stale geometry reached emitter")))

    artifact = module.probe_calibration_capabilities()

    assert artifact["status"] == "unknown"
    assert artifact["compute"]["status"] == "unknown"
    assert "matching the exact CIRCT artifact" in artifact["compute"]["why"]


def test_probe_rejects_unknown_stage_before_claiming_any_capability(tmp_path: Path) -> None:
    module = _module()

    try:
        module.probe_calibration_capabilities(stage="pretend", facts_path=tmp_path / "unused.json")
    except ValueError as exc:
        assert "stage" in str(exc)
    else:  # pragma: no cover - guards a test that otherwise looks green on a false acceptance
        raise AssertionError("invalid stage unexpectedly accepted")


def test_requested_dma_payload_sizes_are_advertised_only_after_every_pure_probe_accepts(
        monkeypatch, tmp_path: Path) -> None:
    module = _module()
    _install_facts(monkeypatch, module, tmp_path)
    monkeypatch.setattr(module, "_emit_or_compile", lambda cb, *, stage: {"status": "accepted"})
    monkeypatch.setattr(module, "_measurement_protocols", lambda: ([], "unavailable"))
    backend = get_backend("gemmini")
    dma = importlib.import_module(f"{backend.__name__}.gemmini_dma_calibration")
    monkeypatch.setattr(dma, "probe_dma_capability", lambda direction, size, _facts, *, stage: {
        "status": "accepted", "direction": direction, "requested_payload_bytes": size,
        "physical_traffic_bytes": {"status": "unmeasured", "value": None}, "stage": stage,
    })

    artifact = module.probe_calibration_capabilities(dma_transfer_sizes=(7, 19))

    probes = artifact["dma"]["direction_probes"]
    assert set(probes) == {"read", "write", "copy"}
    for direction, row in probes.items():
        assert row["status"] == "accepted"
        assert row["requested_payload_sizes_bytes"] == [7, 19]
        assert row["physical_traffic_bytes"]["status"] == "unmeasured"
        receipts = row["probe_receipts"]
        assert [receipt["requested_payload_bytes"] for receipt in receipts] == [7, 19]
        assert all(receipt["status"] == "accepted" for receipt in receipts)
    assert artifact["dma"]["directions"]["value"] == ["read", "write", "copy"]
    assert artifact["dma"]["copy"]["sizes_bytes"]["value"] == [7, 19]


def test_dma_probe_refuses_nonintegral_rtl_element_size_without_rounding(tmp_path: Path) -> None:
    module = _module()
    facts = _facts()
    facts["facts"]["datapaths"] = [{"name": "input", "dtype": "i16"}]
    path = tmp_path / "facts.json"
    path.write_text(json.dumps(facts), encoding="utf-8")

    artifact, sizes, accepted, why = module._dma_probe_requests(facts, (3,), stage="emission")

    assert sizes == (3,)
    assert not accepted
    assert "exact multiple of the RTL-derived input element width" in why
    for row in artifact.values():
        assert row["status"] == "unknown"
        assert row["probe_receipts"][0]["status"] == "unknown"
        assert "exact multiple" in row["probe_receipts"][0]["why"]
