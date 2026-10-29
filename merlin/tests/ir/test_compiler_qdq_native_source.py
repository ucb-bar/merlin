"""Real pinned source capture through actual compiler boundaries; no policy approval."""

import json
import os
import shutil
from pathlib import Path

import pytest
from xdsl.dialects import arith, func

from merlin.common import owned_process
from merlin.common.paths import repo_root
from merlin.llvmlower import lower, passes_xdsl
from merlin.verify import compiler_qdq as qdq
from merlin.verify import pass_validation
from merlin.verify.gate_errors import GateRejected
from merlin.verify.speculative_gate import GateFailure, SpeculativeGate


@pytest.fixture(scope="module")
def native_qdq_capture(tmp_path_factory):
    names = (
        "MERLIN_QDQ_CAPTURE_PYTHON",
        "MERLIN_QDQ_PRODUCER",
        "MERLIN_QDQ_TORCH_WHEEL",
        "MERLIN_QDQ_CAPTURE_DEPS",
    )
    selections = {name: os.environ.get(name) for name in names}
    missing = [name for name, value in selections.items() if not value or not Path(value).exists()]
    if missing:
        reason = "explicit pinned Q/DQ source tools unavailable: " + ", ".join(missing)
        if os.environ.get("MERLIN_LEAN_REQUIRED"):
            pytest.fail(reason)
        pytest.skip(reason)
    directory = tmp_path_factory.mktemp("native-qdq-source")
    root = repo_root()
    imports = [str(root / "src"), selections[names[1]], selections[names[3]]]
    script = (
        "import json,runpy,sys;sys.path[:0]=json.loads(sys.argv.pop(1));"
        "sys.argv.pop(0);runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    for mode in ("capture", "replay-source"):
        command = [
            selections[names[0]],
            "-I",
            "-B",
            "-c",
            script,
            json.dumps(imports),
            str(root / "build_tools/scripts/qualify_qdq_capture.py"),
            mode,
            "--output",
            str(directory / "captures"),
        ]
        if mode == "capture":
            command += ["--producer", selections[names[1]], "--torch-wheel", selections[names[2]]]
        process = owned_process.run(command, capture_output=True, text=True, timeout=300)
        (directory / f"{mode}.stdout").write_text(process.stdout)
        (directory / f"{mode}.stderr").write_text(process.stderr)
        cleanup = getattr(process, "owned_process_cleanup", None)
        (directory / f"{mode}.cleanup.json").write_text(json.dumps(cleanup, sort_keys=True) + "\n")
        assert cleanup and cleanup.get("complete") is True, cleanup
        assert process.returncode == 0, process.stdout + process.stderr
    return qdq.SourceCapture(
        directory / "captures",
        Path(selections[names[0]]),
        Path(selections[names[1]]),
        Path(selections[names[2]]),
        Path(selections[names[3]]),
    )


def select(captures, name, *, prove_prefix=False):
    return qdq.SourceCapture(
        captures.directory / name,
        captures.python,
        captures.producer,
        captures.torch_wheel,
        captures.dependencies,
        prove_prefix=prove_prefix,
    )


@pytest.mark.parametrize("name", ["quantize", "dequantize_tensor", "dequantize_channel", "composition"])
@pytest.mark.parametrize("textual", [False, True])
def test_actual_native_source_gate_matches_ungated_llvm(tmp_path, pass_gate_tools, native_qdq_capture, name, textual):
    capture = select(native_qdq_capture, name)
    source = (capture.directory / "qdq-source.mlir").read_text()
    baseline = lower.lower_model(source, tmp_path / "baseline", targets=(), textual=textual)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({qdq.SELECTOR}), qdq_source=capture)
    checked = lower.lower_model(source, tmp_path / "checked", targets=(), textual=textual, pass_validation=validation)
    assert checked.ll_path.read_bytes() == baseline.ll_path.read_bytes()
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    correspondence = detail["source_correspondence"]
    assert correspondence["status"] == "matched" and correspondence["original_program_bound"] is True
    assert correspondence["actual_input_matches_capture"] is True
    assert correspondence["capture_to_boundary"] == {"status": "exact_generic_ir"}
    assert detail["source_spec_approved"] is False and detail["source_model_status"] == "proposed"
    assert detail["framework_equivalence"] is False and detail["native_fidelity"] is False
    assert correspondence["native_execution_replayed"] is False


def test_actual_native_source_prefix_is_freshly_proved_and_composed(tmp_path, pass_gate_tools, native_qdq_capture):
    capture = select(native_qdq_capture, "quantize", prove_prefix=True)
    module = passes_xdsl.parse_mlir_text((capture.directory / "qdq-source.mlir").read_text())
    function = next(operation for operation in module.walk() if isinstance(operation, func.FuncOp))
    block = function.body.block
    block.insert_op_before(arith.ConstantOp.from_int_and_width(9, 32), block.first_op)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({qdq.SELECTOR}), qdq_source=capture)
    with pass_validation.validating(validation):
        assert passes_xdsl.lower_quant_ext(module) == 1
    assert len(gate.verdicts) == 1 and gate.verdicts[0]["verdict"] == "accepted"
    detail = gate.verdicts[0]["detail"]
    correspondence = detail["source_correspondence"]
    assert correspondence["actual_input_matches_capture"] is False
    assert correspondence["capture_to_boundary"]["status"] == "verified"
    composed = correspondence["capture_to_output"]
    assert composed["status"] == "verified" and set(composed["chains"]) == {"forward"}
    chain = json.loads(Path(composed["chains"]["forward"]["path"]).read_text())
    assert chain["refinement_verdict"] == "verified" and len(chain["links"]) == 2
    assert detail["source_spec_approved"] is False and detail["source_closure_verified"] is False


def test_original_native_program_swap_refuses_before_lean(tmp_path, native_qdq_capture):
    capture = select(native_qdq_capture, "quantize")
    directory = tmp_path / "mutated"
    shutil.copytree(capture.directory, directory)
    shutil.copyfile(native_qdq_capture.directory / "dequantize_tensor/original.pt2", directory / "original.pt2")
    selection = {**capture.describe(), "directory": str(directory)}
    work = tmp_path / "reconstruction"
    work.mkdir()
    with pytest.raises(GateRejected, match="original ExportedProgram differs"):
        qdq._capture(selection, (directory / "qdq-source.mlir").read_text(), work)


def test_changed_native_source_contract_cannot_be_joined(tmp_path, pass_gate_tools, native_qdq_capture):
    capture = select(native_qdq_capture, "quantize", prove_prefix=True)
    source = (capture.directory / "qdq-source.mlir").read_text().replace("quant_max = 127", "quant_max = 126")
    module = passes_xdsl.parse_mlir_text(source)
    gate = SpeculativeGate(tmp_path / "gate", job_timeout=900)
    validation = pass_validation.PassValidation(gate, pass_gate_tools, frozenset({qdq.SELECTOR}), qdq_source=capture)
    with pytest.raises(GateFailure):
        with pass_validation.validating(validation):
            assert passes_xdsl.lower_quant_ext(module) == 1
    assert gate.verdicts[0]["verdict"] == "not_established"
