"""Core ATen offload must prove execution, retaining separate host/device verdicts."""

import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from merlin.common.paths import repo_root
from merlin.runtime.backends import spike_model
from merlin.targetgen import core_aten_device as D
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
from merlin.targetgen.core_aten_capture import case_capture_name


@pytest.fixture
def facts():
    return repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json"


def _record():
    return {
        "overload": "aten.fixture.default",
        "case_id": "fixture-a",
        "status": "bundled",
        "output_indices": [0],
        "output_abi": [{"dtype": "i32", "shape": [1]}],
        "case": {"expected": {"kind": "tensor", "dtype": "int32", "shape": [1], "values": [7]}},
    }


def _device(evidence, region="fixture-contraction"):
    """Device routing of one case with its execution ``evidence``."""
    return dict(
        lane="device",
        routed=True,
        target=evidence["target"],
        execution_evidence=evidence,
        routed_operations=[{"source_region": region}],
    )


def _elf(word):
    # Synthetic ELF64 with one executable section and one word; no toolchain required.
    hdr = bytearray(64)
    hdr[:7] = b"\x7fELF\x02\x01\x01"
    struct.pack_into("<HHI", hdr, 16, 2, 243, 1)
    names = b"\0.text\0.shstrtab\0"
    shoff = 68 + len(names)
    struct.pack_into("<QQQ", hdr, 24, 0, 0, shoff)
    struct.pack_into("<IHHHHHH", hdr, 48, 0, 64, 0, 0, 64, 3, 2)

    def section(name, kind, flags, address, offset, size):
        return struct.pack("<IIQQQQIIQQ", name, kind, flags, address, offset, size, 0, 0, 4, 0)

    sections = section(1, 1, 6, 4096, 64, 4) + section(7, 3, 0, 0, 68, len(names))
    return bytes(hdr) + struct.pack("<I", word) + names + bytes(64) + sections


def _evidence(tmp_path, facts, *, count=1):
    document = json.loads(facts.read_text())
    opcode = document["facts"]["interfaces"][0]["custom_opcode"]
    elf = tmp_path / "case.elf"
    elf.write_bytes(_elf(opcode))
    trace = tmp_path / "trace.log"
    trace.write_text(
        (f"core   0: 0x00001000 (0x{opcode:08x}) custom\n" * count) + f"core 0: 0x2000 (0x{opcode:08x}) outside_ELF\n"
    )
    with D.selected_facts(document["target"], facts):
        evidence = D.executed_device_instructions(trace, elf, document["target"])
        evidence["case_id"] = "fixture-a"
        return evidence


def test_executed_opcode_is_derived_and_requires_matching_pc_word(tmp_path, facts):
    evidence = _evidence(tmp_path, facts)
    assert evidence["executed_instructions"] == 1
    assert D.verify_execution_evidence(evidence)
    Path(evidence["trace"]).write_text("core 0: 0x1000 (0x1234562b) different_word\n")
    assert not D.verify_execution_evidence(evidence)
    with D.selected_facts(evidence["target"], facts), pytest.raises(ValueError, match="differs"):
        D.executed_device_instructions(Path(evidence["trace"]), Path(evidence["elf"]), evidence["target"])


@pytest.mark.parametrize("missing", ["trace", "elf", "facts_path"])
def test_device_evidence_missing_files_fail_closed(tmp_path, facts, missing):
    evidence = _evidence(tmp_path, facts)
    if missing == "facts_path":
        evidence[missing] = str(tmp_path / "absent-facts.json")
    else:
        Path(evidence[missing]).unlink()
    record = dict(_record(), routing=_device(evidence))
    assert grade_core_aten_batch({"cases": [record]}, [np.array([7], np.int32).tobytes()])["passed_count"] == 0


def test_device_pass_requires_nonzero_execution_and_numeric_match(tmp_path, facts):
    record = dict(_record(), routing=_device(_evidence(tmp_path, facts)))
    raw = np.array([7], np.int32).tobytes()
    verdict = grade_core_aten_batch({"cases": [record]}, [raw])
    assert verdict["passed_by_lane"] == {"host": 0, "device": 1}
    assert grade_core_aten_batch({"cases": [record]}, [np.array([8], np.int32).tobytes()])["passed_count"] == 0
    record["routing"]["execution_evidence"] = _evidence(tmp_path, facts, count=0)
    assert grade_core_aten_batch({"cases": [record]}, [raw])["cases"]["fixture-a"]["status"] == "execution_failed"
    record["routing"] = {"lane": "host", "routed": False, "reason": "dtype declined"}
    assert grade_core_aten_batch({"cases": [record]}, [raw])["passed_by_lane"] == {"host": 1, "device": 0}


def test_scalar_batch_uses_current_dump_api_and_all_result_bytes(tmp_path, monkeypatch):
    record = _record()
    report = {"cases": [record], "output_count": 1}
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["dump_all_outputs"] is True
        assert "output_bytes" not in kwargs
        assert kwargs["device"] is None
        return {"elf": tmp_path / "unused.elf", "mem_bytes": 1024}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(
        spike_model, "run", lambda *a, **kw: {"output_bytes": [np.array([7], np.int32).tobytes()], "console": "DONE\n"}
    )
    execution, verdict = D.run_spike_bundle(tmp_path, report, arena_mb=1, timeout=1)
    assert execution["status"] == "ran" and verdict["passed_by_lane"] == {"host": 1, "device": 0}
    assert len(calls) == 1


def test_overlay_variants_do_not_collide():
    first = {"overload": "aten.mm.default", "case_id": "variant-a"}
    second = {**first, "case_id": "variant-b"}
    assert case_capture_name(first) != case_capture_name(second)
    assert case_capture_name({"overload": first["overload"]}) == "aten__mm__default"


def test_facts_override_is_scoped_and_not_regenerated(facts, monkeypatch):
    monkeypatch.setenv("MERLIN_RTL_FACTS", "previous")
    with D.selected_facts("synthetic_core_aten", facts):
        import os

        from merlin.system.offload import device_dtype_triples

        assert os.environ["MERLIN_RTL_FACTS"] == str(facts.resolve())
        assert device_dtype_triples("synthetic_core_aten") == (("i8", "i8", "i32"),)
    assert os.environ["MERLIN_RTL_FACTS"] == "previous"
    with pytest.raises(FileNotFoundError):
        with D.selected_facts("synthetic_core_aten", facts.with_name("absent.json")):
            pass


_SOURCE = """builtin.module {
 func.func @forward(%a: tensor<2x3xi8>, %b: tensor<3x4xi8>) -> tensor<2x4xi32> {
  %z = arith.constant 0 : i32
  %init = tensor.splat %z : tensor<2x4xi32>
  %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : tensor<2x3xi8>, tensor<3x4xi8>) outs(%init : tensor<2x4xi32>) {
    ^bb0(%x: i8, %y: i8, %acc: i32):
      %xx = arith.extsi %x : i8 to i32
      %yy = arith.extsi %y : i8 to i32
      %mul = arith.muli %xx, %yy : i32
      %add = arith.addi %acc, %mul : i32
      linalg.yield %add : i32
  } -> tensor<2x4xi32>
  func.return %r : tensor<2x4xi32>
 }
}"""


def test_routing_admits_only_fact_eligible_source_and_source_bound_provider(tmp_path, facts):
    from merlin.llvmlower.device_build import DeviceRouting

    package = tmp_path / "backend"
    module = tmp_path / "model.mlir"
    module.write_text(_SOURCE)
    calls = []

    def routing(directory, **kwargs):
        calls.append(kwargs)
        return DeviceRouting(
            device=kwargs["target"],
            package_dir=kwargs["package"],
            operand_dtype="i8",
            accum_dtype="i32",
            catalog_builder=lambda *a: None,
        )

    with D.selected_facts("synthetic_core_aten", facts) as document:
        result, reason = D.routing_for_bundle(
            tmp_path, "synthetic_core_aten", package, SimpleNamespace(routing=routing), document
        )
        assert result.catalog_builder is not None and "source-bound" in reason
        assert len(calls[0]["eligible"]) == 1
        # Same syntax with wider operands is a different, declined datapath.
        module.write_text(_SOURCE.replace("xi8>", "xi16>").replace(": i8", ": i16"))
        result, reason = D.routing_for_bundle(
            tmp_path, "synthetic_core_aten", package, SimpleNamespace(routing=routing), document
        )
        assert result is None and "dtypes" in reason and len(calls) == 1


def test_device_bundle_records_attribution_and_rejects_missing_trace(tmp_path, facts, monkeypatch):
    from merlin.llvmlower import device_offload

    record = _record()
    target = "synthetic_core_aten"
    elf = tmp_path / "case.elf"
    opcode = json.loads(facts.read_text())["facts"]["interfaces"][0]["custom_opcode"]
    elf.write_bytes(_elf(opcode))
    report = {"cases": [record], "output_count": 1}
    selected = SimpleNamespace()
    monkeypatch.setattr(
        D,
        "load_execution_provider",
        lambda *a: SimpleNamespace(__file__=str(facts), runner_options=lambda: {"extension": "synthetic"}),
    )
    monkeypatch.setattr(D, "routing_for_bundle", lambda *a: (selected, "source-bound provider"))
    monkeypatch.setattr(device_offload, "load_sidecar", lambda *a: {"routed": [{"source_region": "region-a"}]})

    def build(*a, **kw):
        assert kw["device"] is selected
        return {"elf": elf, "mem_bytes": 1024}

    def run(*a, **kw):
        assert kw["extension"] == "synthetic"
        Path(kw["trace_path"]).write_text(f"core 0: 0x1000 (0x{opcode:08x}) custom\n")
        return {"output_bytes": [np.array([7], np.int32).tobytes()], "console": "DONE\n"}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(spike_model, "run", run)
    execution, verdict = D.run_spike_bundle(
        tmp_path, report, arena_mb=1, timeout=1, target=target, device_package=tmp_path, rtl_facts=facts
    )
    assert execution["status"] == "ran" and verdict["passed_by_lane"]["device"] == 1
    assert record["routing"]["routed_operations"][0]["source_region"] == "region-a"
    (tmp_path / "spike-trace.log").unlink()
    monkeypatch.setattr(spike_model, "run", lambda *a, **kw: {"output_bytes": [b""], "console": "DONE\n"})
    execution, verdict = D.run_spike_bundle(
        tmp_path, report, arena_mb=1, timeout=1, target=target, device_package=tmp_path, rtl_facts=facts
    )
    assert execution["status"] == "failed" and verdict["passed_count"] == 0


def test_device_mode_rejects_multiple_case_attribution(tmp_path, facts, monkeypatch):
    report = {"cases": [_record(), {**_record(), "case_id": "fixture-b"}], "output_count": 2}
    monkeypatch.setattr(spike_model, "build", lambda *a, **kw: pytest.fail("ambiguous shard reached compiler"))
    execution, verdict = D.run_spike_bundle(
        tmp_path, report, arena_mb=1, timeout=1, target="synthetic_core_aten", device_package=tmp_path, rtl_facts=facts
    )
    assert execution["status"] == "failed" and "exactly one" in execution["error"]
    assert verdict["passed_count"] == 0


def test_duplicate_case_ids_are_refused_before_capture_or_bundle(tmp_path):
    from merlin.targetgen.core_aten_batch import build_core_aten_batch
    from merlin.targetgen.core_aten_capture import capture_case_corpus

    corpus = {"cases": [_record(), _record()]}
    with pytest.raises(ValueError, match="duplicate"):
        build_core_aten_batch(corpus, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        capture_case_corpus(corpus, tmp_path)


def test_device_evidence_cannot_be_reused_for_another_case_or_a_batch(tmp_path, facts):
    record = dict(_record(), routing=_device(_evidence(tmp_path, facts), "fixture"))
    raw = np.array([7], np.int32).tobytes()
    other = {**record, "case_id": "different-case"}
    assert grade_core_aten_batch({"cases": [other]}, [raw])["passed_count"] == 0
    assert grade_core_aten_batch({"cases": [record, other]}, [raw])["passed_count"] == 0


def test_routing_reason_does_not_hide_execution_failure():
    record = _record()
    record["routing"] = {"lane": "host", "routed": False, "reason": "unsupported operand dtype"}
    verdict = grade_core_aten_batch({"cases": [record]}, execution_error="compiler unavailable")
    case = verdict["cases"]["fixture-a"]
    assert case["reason"] == "compiler unavailable"
    assert case["routing_reason"] == "unsupported operand dtype"


def test_cli_keeps_overlay_variants_separate_in_single_case_device_shards(tmp_path, monkeypatch):
    from merlin.targetgen.core_aten_cover import json_bytes
    from merlin.targetgen.plugins import load_module

    cli = load_module(repo_root() / "build_tools/scripts", "build_core_aten_batch.py", package_name="batch_cli_test")
    first = _record()
    second = {**first, "case_id": "fixture-b"}
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_bytes(json_bytes({"cases": [first, second]}))
    output = tmp_path / "batch"
    calls = []

    def bundle(corpus, captures, directory, *, selected_overloads=None):
        directory.mkdir(parents=True, exist_ok=True)
        selected = [selected_overloads is None or case["case_id"] in selected_overloads for case in corpus["cases"]]
        statuses = ["bundled" if chosen else "not_in_shard" for chosen in selected]
        cases = [{**case, "status": status} for case, status in zip(corpus["cases"], statuses)]
        count = sum(case["status"] == "bundled" for case in cases)
        return {"cases": cases, "bundled_count": count, "case_count": 2, "input_count": 0, "output_count": count}

    def run(directory, report, **kwargs):
        calls.append(kwargs)
        return {"status": "ran"}, grade_core_aten_batch(report, [np.array([7], np.int32).tobytes()])

    provider = tmp_path / "provider.py"
    provider.write_text('TARGET = "synthetic_core_aten"\n')
    monkeypatch.setenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", str(provider))
    monkeypatch.setattr(cli, "build_core_aten_batch", bundle)
    monkeypatch.setattr(cli, "_run_spike_bundle", run)
    args = ["--corpus", str(corpus_path), "--captures", str(tmp_path), "--output-dir", str(output)]
    assert cli.main(args + ["--target", "synthetic_core_aten", "--device-package", str(tmp_path), "--run-spike"]) == 0
    ledger = json.loads((output / "shard_executions.json").read_text())
    assert [row["case_ids"] for row in ledger] == [["fixture-a"], ["fixture-b"]]
    assert all(row["overloads"] == [first["overload"]] for row in ledger)
    assert len(calls) == 2 and all(call["target"] == "synthetic_core_aten" for call in calls)
    verdict = json.loads((output / "core_aten_batch_verdict.json").read_text())
    assert verdict["passed_count"] == 2 and set(verdict["cases"]) == {"fixture-a", "fixture-b"}


def test_execution_provider_selection_refuses_legacy_default(tmp_path, monkeypatch):
    monkeypatch.delenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", raising=False)
    # Even a legacy-layout file is never rediscovered.
    legacy = tmp_path / "examples/synthetic/phase0/core_aten/execution_provider.py"
    legacy.parent.mkdir(parents=True)
    legacy.write_text('TARGET = "synthetic"\n')
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="explicit target-owned execution provider"):
        D.load_execution_provider("synthetic")
    with pytest.raises(ValueError, match="does not exist"):
        D.load_execution_provider("synthetic", tmp_path / "missing.py")
    monkeypatch.setenv("MERLIN_CORE_ATEN_EXECUTION_PROVIDER", str(legacy))
    assert D.load_execution_provider("synthetic").TARGET == "synthetic"
    other = tmp_path / "explicit.py"
    other.write_text('TARGET = "other"\n')
    assert D.load_execution_provider("other", other).TARGET == "other"
    with pytest.raises(ValueError, match="selected target"):
        D.load_execution_provider("synthetic", other)
