"""A real shared-pass chain gates downstream consumption; incomplete evidence stops it."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.llvmlower.passes_xdsl import preprocess_text
from merlin.verify.lean import check, handoff, stages

SOURCE = """func.func @forward(%a: i32, %b: i32) -> i32 {
  %dead = arith.muli %a, %b : i32
  %r = arith.addi %a, %b : i32
  return %r : i32
}"""


def _options(tools, bits=64):
    return handoff.Options("forward", bits, tools.m2m_python)


def test_scoped_capture_isolated_from_environment_and_other_threads(tmp_path, monkeypatch):
    ambient = tmp_path / "ambient"
    monkeypatch.setenv(stages.ENV, str(ambient))

    def record(k):
        with stages.capture(tmp_path / f"worker-{k}") as capture:
            for n in range(3):
                stages.record_boundary(str(n), "test", f"{k}:{n}", f"{k}:{n + 1}")
        return capture

    with ThreadPoolExecutor(max_workers=4) as pool:
        recordings = list(pool.map(record, range(4)))
    assert not ambient.exists()
    assert len({r.run for r in recordings}) == 4
    for recording in recordings:
        assert recording.complete
        published = stages.completed(recording.root)
        assert len(published) == 3
        assert {json.loads((s / "meta.json").read_text())["run"] for s in published} == {recording.run}
    stages.record_boundary("ambient", "test", "a", "b")
    assert len(stages.completed(ambient)) == 1


def test_scoped_capture_refuses_reuse_nesting_and_interrupted_success(tmp_path, monkeypatch):
    monkeypatch.delenv(stages.ENV, raising=False)
    with pytest.raises(RuntimeError, match="interrupted"):
        with stages.capture(tmp_path / "capture") as recording:
            with pytest.raises(ValueError, match="nested"):
                with stages.capture(tmp_path / "nested"):
                    pytest.fail("nested capture entered")
            raise RuntimeError("interrupted")
    assert not recording.complete and not stages.enabled()
    with pytest.raises(FileExistsError):
        with stages.capture(recording.root):
            pytest.fail("capture reused")


@pytest.mark.parametrize("bits", [32, 64])
def test_actual_preprocessing_requires_a_replayed_defined_chain(tmp_path, lean_tools, bits):
    plain, plain_stats = preprocess_text(SOURCE)
    result = handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(lean_tools, bits), preprocess_text)
    assert result.text == plain and result.stats == plain_stats
    report = json.loads(result.report.read_text())
    chain = json.loads(Path(report["chain"]).read_text())
    assert report["verdict"] == chain["verdict"] == "verified"
    assert chain["source_definedness"]["verdict"] == "verified"
    assert chain["replayed"] and chain["chained"]
    assert len(report["boundaries"]) == 4
    assert report["handoff_sha256"] == chain["handoff_sha256"]


@pytest.mark.parametrize("fault", ["no_records", "source", "output", "foreign", "unfinished", "corrupt"])
def test_incomplete_capture_refuses_before_checking(tmp_path, monkeypatch, fault):
    def never_check(*args, **kwargs):
        pytest.fail("incomplete capture reached the checker")

    monkeypatch.setattr(check, "check_typed_stage", never_check)

    def broken(source):
        if fault == "no_records":
            return source, {}
        stage = stages.record_boundary("test", "test", source, source)
        if fault in {"foreign", "unfinished"}:
            meta = json.loads((stage / "meta.json").read_text())
            if fault == "foreign":
                meta["run"] = "another run"
                (stage / "meta.json").write_text(json.dumps(meta))
            else:
                (stage.parent / ".reserved" / "000001").mkdir()
        if fault in {"source", "corrupt"}:
            (stage / "before.mlir").write_text("another source")
            if fault == "source":
                # Keep metadata consistent: the endpoint identity itself must still be checked.
                meta = json.loads((stage / "meta.json").read_text())
                meta["before_sha256"] = hashlib.sha256(b"another source").hexdigest()
                (stage / "meta.json").write_text(json.dumps(meta))
        return source + ("\n// unrecorded" if fault == "output" else ""), {}

    with pytest.raises(handoff.VerificationError):
        handoff.run(SOURCE, tmp_path / "upstream.mlir", handoff.Options("forward", 64, Path("absent")), broken)
    report = next(tmp_path.glob("lean-shared-passes/*/verification.json"))
    assert json.loads(report.read_text())["verdict"] != "verified"


def test_actual_invalid_rewrite_and_undefined_source_stop_the_gate(tmp_path, lean_tools):
    def wrong(source):
        target = source.replace("arith.addi", "arith.subi")
        stages.record_boundary("wrong-rewrite", "test", source, target)
        return target, {}

    with pytest.raises(handoff.VerificationError, match="function body did not verify"):
        handoff.run(SOURCE, tmp_path / "wrong.mlir", _options(lean_tools), wrong)
    poison = """func.func @forward() -> tensor<2xi32> {
      %e = tensor.empty() : tensor<2xi32>
      return %e : tensor<2xi32>
    }"""
    with pytest.raises(handoff.VerificationError, match="source definedness"):
        handoff.run(poison, tmp_path / "poison.mlir", _options(lean_tools), preprocess_text)


@pytest.mark.parametrize("verdict", ["abstracted", "unmeasured", "refuted"])
def test_nonverified_receipt_never_reaches_downstream(tmp_path, monkeypatch, verdict):
    from merlin.llvmlower import lower

    monkeypatch.setattr(check, "check_typed_stage", lambda *a, **kw: {"verdict": verdict})
    monkeypatch.setattr(lower, "lower_to_llvm_ir", lambda *a, **kw: pytest.fail("downstream compiler invoked"))
    with pytest.raises(handoff.VerificationError):
        lower.lower_model(
            SOURCE,
            tmp_path,
            targets=(),
            index_bits=64,
            shared_pass_verification=handoff.Options("forward", 64, Path("absent")),
        )
    assert not (tmp_path / "model.ll").exists()


def test_verified_lowering_consumes_exact_text_and_propagates_options(tmp_path, monkeypatch, lean_tools):
    from merlin.llvmlower import lower

    consumed = []

    def backend(text, **kwargs):
        consumed.append(text)
        kwargs["lowering_selection"].update(index_bits=64, effective_pipeline="test-only backend")
        return "define void @forward() { ret void }"

    monkeypatch.setattr(lower, "lower_to_llvm_ir", backend)
    # The post-lowering helper is imported locally, so patch its defining module.
    from merlin.llvmlower import ciface_inline

    monkeypatch.setattr(ciface_inline, "inline_wide_ciface_implementations", lambda text: (text, {}))
    result = lower.lower_model_file(
        _source_file(tmp_path),
        tmp_path / "build",
        targets=(),
        index_bits=64,
        shared_pass_verification=_options(lean_tools),
    )
    assert consumed == [(result.workdir / "model.upstream.mlir").read_text()]
    assert json.loads(Path(result.stats["shared_pass_verification"]).read_text())["verdict"] == "verified"


def _source_file(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(SOURCE)
    return source


def test_gate_rechecks_inventory_after_chain_replay(tmp_path, monkeypatch, lean_tools):
    compose = check.compose_chain

    def race(*args, **kwargs):
        result = compose(*args, **kwargs)
        assert result["verdict"] == "verified", result.get("reason")
        meta = args[0][0] / "meta.json"
        meta.write_text(meta.read_text() + "\n")
        return result

    monkeypatch.setattr(check, "compose_chain", race)
    with pytest.raises(handoff.VerificationError, match="changed during validation"):
        handoff.run(SOURCE, tmp_path / "upstream.mlir", _options(lean_tools), preprocess_text)


@pytest.mark.parametrize("bits", [None, 32])
def test_lowering_requires_the_same_explicit_index_width(tmp_path, bits):
    from merlin.llvmlower import lower

    with pytest.raises(ValueError, match="index widths must match"):
        lower.lower_model(
            SOURCE,
            tmp_path,
            targets=(),
            index_bits=bits,
            shared_pass_verification=handoff.Options("forward", 64, Path("absent")),
        )
