"""Checked capture-to-boundary IR joins; these tests do not qualify native capture."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_lean_quantize_contract import source

from merlin.llvmlower.passes_xdsl import lower_quant_ext, parse_mlir_text, prune_dead_pure_tensor_ops
from merlin.verify import compiler_qdq as qdq
from merlin.verify.gate_checks import tensor_refinement
from merlin.verify.gate_errors import GateRejected
from merlin.xdsl_dialects._common import text


def parameters(tools):
    return {"sdk_bin": str(tools.sdk_bin), "compiler_python": str(tools.compiler_python), "index_bits": 64}


def test_actual_dce_prefix_joins_captured_qdq_ir(tmp_path, pass_gate_tools):
    captured = source().replace("  %q =", "  %dead = arith.constant 9 : i32\n  %q =")
    module = parse_mlir_text(captured)
    assert prune_dead_pure_tensor_ops(module) == 1
    actual = text(module, generic=True)
    proof, refs = qdq._check_prefix(captured, actual, tmp_path / "prefix", parameters(pass_gate_tools))
    assert proof["status"] == "verified" and proof["relation"] == "strict_typed_refinement"
    assert proof["source_spec_approved"] is False
    assert Path(proof["before"]["path"]).read_text() == captured
    assert Path(proof["after"]["path"]).read_text() == actual
    for ref in refs:
        qdq.bindings._bound(ref)
    lowered = parse_mlir_text(actual)
    assert lower_quant_ext(lowered) == 1
    directory = tmp_path / "lowering"
    directory.mkdir()
    before, after = directory / "before.mlir", directory / "after.mlir"
    before.write_text(actual)
    after.write_text(text(lowered, generic=True))
    detail = tensor_refinement(
        {
            "directory": str(directory),
            "before": str(before),
            "after": str(after),
            "label": qdq.SELECTOR,
            "params": parameters(pass_gate_tools),
        }
    )
    composed, refs = qdq._compose_prefix(proof, detail, directory, parameters(pass_gate_tools))
    assert composed["status"] == "verified" and set(composed["chains"]) == {"forward"}
    chain = json.loads(Path(composed["chains"]["forward"]["path"]).read_text())
    assert chain["refinement_verdict"] == "verified"
    for ref in refs:
        qdq.bindings._bound(ref)


def test_changed_source_qdq_contract_cannot_form_a_prefix(tmp_path, pass_gate_tools):
    captured = source()
    with pytest.raises(GateRejected):
        qdq._check_prefix(
            captured,
            captured.replace("quant_max = 127", "quant_max = 126"),
            tmp_path / "prefix",
            parameters(pass_gate_tools),
        )


@pytest.mark.parametrize("value", [True, False])
def test_prefix_selection_is_explicit_and_serialized(tmp_path, value):
    selected = qdq.SourceCapture(
        tmp_path, tmp_path / "python", tmp_path / "producer", tmp_path / "wheel", prove_prefix=value
    )
    assert selected.describe()["prove_prefix"] is value


def test_truthy_prefix_selection_is_not_an_explicit_boolean(tmp_path):
    with pytest.raises(ValueError, match="explicit boolean"):
        qdq.SourceCapture(tmp_path, tmp_path / "python", tmp_path / "producer", tmp_path / "wheel", prove_prefix=1)


@pytest.mark.parametrize("raw,expected", [("0", False), ("1", True), ("yes", None)])
def test_environment_prefix_requires_zero_or_one(tmp_path, monkeypatch, raw, expected):
    monkeypatch.setenv(qdq.ENV_CAPTURE, str(tmp_path))
    for name in ("MERLIN_QDQ_CAPTURE_PYTHON", "MERLIN_QDQ_PRODUCER", "MERLIN_QDQ_TORCH_WHEEL"):
        monkeypatch.setenv(name, str(tmp_path))
    monkeypatch.setenv("MERLIN_QDQ_PROVE_PREFIX", raw)
    if expected is None:
        with pytest.raises(ValueError, match="0 or 1"):
            qdq.SourceCapture.from_environment()
    else:
        assert qdq.SourceCapture.from_environment().prove_prefix is expected


def test_opted_in_host_transport_marks_prefix_as_still_needing_proof(tmp_path, monkeypatch):
    """A substituted child tests transport only, never native source qualification."""
    from merlin.common import owned_process

    capture = tmp_path / "capture"
    capture.mkdir()
    for name in ("original.pt2", "qdq-frontend.json", "qdq-producer.json", "qdq-source-check.json"):
        (capture / name).write_text("host transport fixture; not a real capture")
    (capture / "qdq-source.mlir").write_text(source())
    selected = qdq.SourceCapture(
        capture, Path(sys.executable), tmp_path / "producer", tmp_path / "wheel", prove_prefix=True
    )
    result = {
        "status": "matched",
        "original_program_bound": True,
        "sites": [{"host_transport_control": True}],
        "native_execution_replayed": False,
    }
    monkeypatch.setattr(
        owned_process,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(result),
            stderr="",
            owned_process_cleanup={"complete": True},
        ),
    )
    received, _ = qdq._capture(selected.describe(), source().replace("quant_max = 127", "quant_max = 126"), tmp_path)
    assert received["actual_input_matches_capture"] is False
    assert "capture_to_boundary" not in received
    assert received["native_execution_replayed"] is False
