"""Source opportunities stay separate from actual native emitted images."""
import hashlib
import json

import pytest
from xdsl.dialects import func

from merlin.compile import compiler_observations as C
from merlin.compile.scheduling.multioutput_fusion import compile_source_multioutput
from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest
from merlin.xdsl_dialects.lowering.outline import outline_dispatches
from test_multioutput_fusion import model, source_program


def test_actual_source_fanout_and_packing_are_analysis_only():
    module, operations = source_program()
    requests = tuple(TensorPreparationRequest(operations[0].results[0], op, "a" * 64) for op in operations[1:])
    packing = C.packing(module, requests)
    assert packing["planned"] == {"pack_calls": 1}
    assert packing["repeated_source_reads"] == 2
    assert packing["opportunities"][0]["cross_invocation_reuse"] == "unproved"
    fusion = C.connected_fusion(module)
    assert fusion["planned"] == {"source_regions": 1}
    assert fusion["candidates"][0]["live_outputs"] == 2
    assert packing["emitted"] == fusion["emitted"] == {}
    assert "packed_bytes" not in packing["planned"]


def test_observed_producer_is_declined_not_silently_fused():
    module, operations = source_program()
    ret = next(op for op in module.walk() if isinstance(op, func.ReturnOp))
    # An extra observer invalidates private intermediate ownership.
    ret.operands = [*ret.operands, operations[0].results[0]]
    result = C.connected_fusion(module)
    assert result["planned"]["source_regions"] == 0
    assert result["candidates"][0]["status"] == "declined"
    assert "observ" in result["candidates"][0]["reason"]


def test_real_outlined_driver_retains_arena_and_stream_scope():
    module, _ = source_program()
    outlined = outline_dispatches(module.clone())
    result = C.lifetimes(outlined, weight_args=(1,))
    assert result["planned"]["allocations"] == 4
    assert result["planned"]["peak_live_bytes"] == 64
    assert result["arena_plan"]["arena_bytes"] == 64
    assert result["stored_argument_indices"] == [1]
    assert result["stream_plan"]["prepack"]["weights_known"]
    assert result["emitted"] == {}
    assert result["stream_plan"]["fences"] is None
    assert result["stream_plan"]["placement_status"] == "unknown"
    with pytest.raises(ValueError, match="ABI"):
        C.lifetimes(outlined, weight_args=(999,))


def test_real_emission_uses_issued_guard_and_actual_image_bytes(tmp_path):
    module, operations, options = model(tmp_path)
    compiled = compile_source_multioutput(module, operations, **options)
    result = C.emitted_multioutput(compiled)
    assert result["emitted"] == {"emitted_regions": 1}
    assert result["instruction_images"][0]["sha256"] == hashlib.sha256(compiled.images()[0]).hexdigest()
    assert result["maturity"] == "validated_instruction_image_not_linked_execution"
    with pytest.raises(ValueError, match="issued"):
        C.emitted_multioutput({"emitted_regions": 1})


def test_receipt_is_exact_pinned_and_cannot_overwrite(tmp_path):
    module, _ = source_program()
    path = tmp_path / "receipt.json"
    kwargs = dict(compiler_sha256="a" * 64, capsule_sha256="b" * 64, scope_sha256="c" * 64,
                  sections={"connected_fusion": C.connected_fusion(module)})
    records = C.write_receipt(path, **kwargs)
    assert records["connected_fusion"]["artifact"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text())["promotion_authority"] is False
    assert records["connected_fusion"]["emitted"] == {}
    with pytest.raises(ValueError, match="new direct"):
        C.write_receipt(path, **kwargs)
    kwargs["scope_sha256"] = None
    with pytest.raises(ValueError, match="pins"):
        C.write_receipt(tmp_path / "bad.json", **kwargs)


def test_actual_llvm_binding_is_not_runtime_peak():
    import runpy
    from merlin.common.paths import repo_root

    source = runpy.run_path(str(repo_root() / "merlin/tests/runtime/test_arena_bind.py"))["SEQUENTIAL"]
    lowered, report = C.bind_arena_observed(source)
    assert "@merlin_arena = internal global [128 x i8]" in lowered
    assert report["emitted"] == {"arena_bytes": 128, "bound_heap_allocations": 2}
    assert "peak_live_bytes" not in report["emitted"]
    assert report["emitted_llvm_sha256"] == hashlib.sha256(lowered.encode()).hexdigest()


def test_native_prepare_installation_retains_static_emission_scope():
    import runpy
    from merlin.common.paths import repo_root

    fixture = runpy.run_path(str(repo_root() / "merlin/tests/ir/test_prepared_operand_schedule.py"))["fixture"]
    module, plan, _, writers = fixture()
    native, observed = C.install_prepared_observed(
        module, plan, writers, prepare_symbol="prepare", borrowed_symbol="prepared_borrow",
        validate_physical=lambda *args: None,
    )
    assert native["epochs"] == observed["emitted"]["pack_calls"] == 1
    assert observed["maturity"] == "installed_source_schedule_not_linked_execution"
    assert "packed_bytes" not in observed["emitted"]
    assert observed["source_ir_sha256"] != observed["emitted_ir_sha256"]
