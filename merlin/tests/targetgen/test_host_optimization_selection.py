"""Ordinary original-executor host choices preserve source/object custody."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.llvmlower import codegen, pipeline, toolchain
from merlin.llvmlower.host_optimization_request import (
    HostOptimizationContext,
    HostOptimizationRequest,
    HostOptimizationSelection,
    prepare_host_optimization,
    request_from_record,
)
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.package_runtime import scoped_package_executor

_LLVM = "module { llvm.func @kernel() { llvm.return } }"
_LOGICAL = """module {
func.func @kernel(%x: tensor<2x3xi32>, %b: tensor<3xi32>) -> tensor<2x3xi32> {
 %e = tensor.empty() : tensor<3x2xi32>
 %t = linalg.transpose ins(%x:tensor<2x3xi32>) outs(%e:tensor<3x2xi32>) permutation = [1,0]
 %e1 = tensor.empty() : tensor<3x2xi32>
 %a = linalg.generic {indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
 affine_map<(d0,d1)->(d0)>,affine_map<(d0,d1)->(d0,d1)>],iterator_types=["parallel","parallel"]}
 ins(%t,%b:tensor<3x2xi32>,tensor<3xi32>) outs(%e1:tensor<3x2xi32>) {
 ^bb0(%v:i32,%bias:i32,%init:i32):
 %r = arith.addi %v,%bias : i32
 linalg.yield %r : i32
 } -> tensor<3x2xi32>
 %e2 = tensor.empty() : tensor<2x3xi32>
 %r = linalg.transpose ins(%a:tensor<3x2xi32>) outs(%e2:tensor<2x3xi32>) permutation = [1,0]
 func.return %r : tensor<2x3xi32>
} }"""


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _selection(source, *, choices=None, context=None):
    original = HostOptimizationContext(
        "kernel",
        _sha("original reviewed semantics"),
        permitted_stages=("logical_host", "structured_host", "llvm"),
        permitted_optimizations=("propagate_layout", "fusion", "hoist_static_allocs", "sink_deallocs", "static_arena"),
        index_bits=64,
        data_layout="e-p:64:64-i64:64-n32:64-S128",
        independent_element_order_unobserved=True,
        sequential_execution=True,
        successful_allocations=True,
        non_reentrant_entry=True,
    )
    return HostOptimizationSelection(
        replace(original, **(context or {})),
        HostOptimizationRequest(_sha(source), **(choices or {})),
    )


class _OriginalExecutor:
    def __init__(self, selection):
        self.host_optimization_selection = selection

    def run_entrypoint(self, *_args, **_kwargs):
        raise AssertionError("unexpected author execution")

    def build_package(self, *_args, **_kwargs):
        raise AssertionError("unexpected package build")


def _fake_object(source, output, *_args, **_kwargs):
    Path(output).write_bytes(Path(source).read_bytes())
    return Path(output)


def test_public_record_has_only_exact_source_bound_safe_choices():
    request = _selection(_LOGICAL, choices={"fusion_groups": ((0, 1),)}).request
    assert request_from_record(request.record()) == request
    for key, value in (("pipeline", "convert-everything"), ("semantic_sha256", _sha("invented")), ("index_bits", 32)):
        with pytest.raises(ValueError, match="unknown"):
            request_from_record({**request.record(), key: value})
    with pytest.raises(ValueError, match="exact booleans"):
        HostOptimizationRequest(_sha(_LLVM), static_arena=1)
    with pytest.raises(ValueError, match="unique node"):
        HostOptimizationRequest(_sha(_LLVM), fusion_groups=((0, 0),))


@pytest.mark.parametrize(
    "choice", [{"propagate_layout": True}, {"hoist_static_allocs": False}, {"sink_deallocs": True}]
)
def test_finished_llvm_refuses_tensor_and_structured_requests(choice):
    with pytest.raises(ValueError, match="finished LLVM"):
        prepare_host_optimization(_LLVM, _selection(_LLVM, choices=choice))


@pytest.mark.parametrize("premise", ["sequential_execution", "successful_allocations", "non_reentrant_entry"])
def test_arena_cannot_infer_original_runtime_premises_from_semantic_identity(premise):
    with pytest.raises(ValueError, match="premises|non-reentrant"):
        prepare_host_optimization(_LLVM, _selection(_LLVM, choices={"static_arena": True}, context={premise: False}))


def test_source_binding_original_admission_and_device_refusal():
    selection = _selection(_LOGICAL, choices={"propagate_layout": True})
    with pytest.raises(ValueError, match="actual compiler source"):
        prepare_host_optimization(_LOGICAL + "\n", selection)
    with pytest.raises(ValueError, match="has not admitted optimization"):
        prepare_host_optimization(
            _LOGICAL, _selection(_LOGICAL, choices={"propagate_layout": True}, context={"permitted_optimizations": ()})
        )
    with pytest.raises(ValueError, match="original index width"):
        prepare_host_optimization(
            _LOGICAL, _selection(_LOGICAL, choices={"propagate_layout": True}, context={"index_bits": None})
        )
    mixed = _LOGICAL.replace("func.return %r", '"foreign.device_event"() : () -> ()\n func.return %r')
    with pytest.raises(ValueError, match="unmodeled or device"):
        prepare_host_optimization(mixed, _selection(mixed, choices={"fusion_groups": ((0, 1),)}))
    floating = _LOGICAL.replace("i32", "f32").replace("arith.addi", "arith.addf")
    with pytest.raises(ValueError, match="floating environment"):
        prepare_host_optimization(floating, _selection(floating, choices={"fusion_groups": ((0, 1),)}))


def test_original_executor_applies_graph_before_actual_lowering_and_retains_source(monkeypatch, tmp_path):
    selection = _selection(
        _LOGICAL, choices={"propagate_layout": True, "hoist_static_allocs": False, "sink_deallocs": True}
    )
    observed = []

    def lower(source, *, lowering_selection, **options):
        assert source != _LOGICAL
        assert source.count('"linalg.transpose"') < _LOGICAL.count("linalg.transpose")
        assert "hoist-static-allocs" not in options["pipeline"]
        assert "buffer-hoisting" not in options["pipeline"]
        assert pipeline.SINK_DEALLOC_PASS in options["pipeline"]
        actual = pipeline._bind_index_width(options["pipeline"], options["index_bits"])
        lowering_selection.update(
            index_bits=options["index_bits"], data_layout=options["data_layout"], effective_pipeline=actual
        )
        observed.append((source, options))
        return "define void @kernel() {\n ret void\n}\n"

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lower)
    monkeypatch.setattr(codegen, "compile_ll", _fake_object)
    with scoped_package_executor(_OriginalExecutor(selection)):
        obj = compiler.llvm_mlir_to_object(_LOGICAL, tmp_path)
    report = json.loads((tmp_path / "kernel.host_optimization.json").read_text())
    assert (tmp_path / "kernel.llvm.mlir").read_text() == _LOGICAL
    assert report["status"] == "applied" and report["selection_sha256"] == selection.identity()
    assert report["products"]["object"]["sha256"] == hashlib.sha256(obj.read_bytes()).hexdigest()
    assert "index-bitwidth=64" in report["observed_lowering"]["effective_pipeline"]
    assert report["formal_proof"] == "not_issued" and report["timing"] == "UNKNOWN"
    translations = [invocation_record.verify(p) for p in tmp_path.rglob("invocation.json")]
    call = next(row for row in translations if row["stage"] == "llvm_translation")
    assert len(call["inputs"]) == 2 and len(observed) == 1


def test_unselected_route_remains_unmodified_and_no_report_claims_selection(monkeypatch, tmp_path):
    def lower(source, **options):
        assert source == _LLVM and set(options) == {"workdir"}
        return "define void @kernel() {\n ret void\n}\n"

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lower)
    monkeypatch.setattr(codegen, "compile_ll", _fake_object)
    compiler.llvm_mlir_to_object(_LLVM, tmp_path)
    assert not (tmp_path / "kernel.host_optimization.json").exists()


def test_unchanged_arena_distinguished_and_changed_source_cannot_complete(monkeypatch, tmp_path):
    selection = _selection(_LLVM, choices={"static_arena": True})
    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lambda *_args, **_kw: "define void @kernel() {\n ret void\n}\n")
    monkeypatch.setattr(codegen, "compile_ll", _fake_object)
    executor = _OriginalExecutor(selection)
    with scoped_package_executor(executor):
        compiler.llvm_mlir_to_object(_LLVM, tmp_path)
    report = json.loads((tmp_path / "kernel.host_optimization.json").read_text())
    assert report["arena"]["bound"] == 0 and report["status"] == "unchanged"

    def mutated(source, output, *_args, **_kwargs):
        result = _fake_object(source, output)
        (tmp_path / "kernel.host.mlir").write_text(_LLVM + "\n")
        return result

    monkeypatch.setattr(codegen, "compile_ll", mutated)
    with scoped_package_executor(executor), pytest.raises(ValueError, match="selected IR changed"):
        compiler.llvm_mlir_to_object(_LLVM, tmp_path)
    assert not (tmp_path / "kernel.host_optimization.json").exists()


def _unused_renderer(*_args, **_kwargs):
    raise AssertionError("object compilation does not render a harness")


def _arena_mlir():
    allocs = []
    for name in ("a", "b"):
        allocs.append(f"""%{name} = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
 %{name}i = llvm.ptrtoint %{name} : !llvm.ptr to i64
 %{name}j = llvm.add %{name}i,%pad : i64
 %{name}k = llvm.urem %{name}j,%alignment : i64
 %{name}l = llvm.sub %{name}j,%{name}k : i64
 %{name}p = llvm.inttoptr %{name}l : i64 to !llvm.ptr
 llvm.store %value,%{name}p : i32,!llvm.ptr
 %{name}v = llvm.load %{name}p : !llvm.ptr -> i32
 llvm.call @free(%{name}) : (!llvm.ptr) -> ()""")
    return (
        """module {
 llvm.func @malloc(i64) -> !llvm.ptr
 llvm.func @free(!llvm.ptr)
 llvm.func @kernel(%out: !llvm.ptr) {
 %size = llvm.mlir.constant(128 : i64) : i64
 %pad = llvm.mlir.constant(63 : i64) : i64
 %alignment = llvm.mlir.constant(64 : i64) : i64
 %value = llvm.mlir.constant(7 : i32) : i32
 """
        + "\n".join(allocs)
        + """
 %sum = llvm.add %av,%bv : i32
 llvm.store %sum,%out : i32,!llvm.ptr
 llvm.return
 } }"""
    )


def test_actual_ordinary_executor_arena_object_preserves_translation_product(tmp_path):
    from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
    from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest

    selected_clang = Path(toolchain.clang()).resolve()
    if not selected_clang.is_file() or not toolchain.mlir_translate().is_file():
        pytest.skip("requires explicitly selected ordinary LLVM translator and compiler")
    script = tmp_path / "unused.ld"
    script.write_text("SECTIONS {}\n")
    owner = Path(__file__).resolve()
    recipe = HarnessBuildRecipe(
        selected_clang,
        (),
        (),
        script,
        0,
        cflags=("-march=rv64gc", "-mabi=lp64d"),
        kernel_stack_frame=KernelStackFramePolicy("kernel", 1024),
    )
    service = BuildOnlyService(
        "host_policy_fixture",
        recipe,
        _unused_renderer,
        tuple((str(path), file_digest(path)) for path in (owner, selected_clang, script)),
    )
    source = _arena_mlir()
    selection = _selection(source, choices={"static_arena": True})
    work = tmp_path / "actual"
    with scoped_package_executor(_OriginalExecutor(selection)):
        obj = compiler.llvm_mlir_to_object(
            source, work, target=service.target, _build_service=service, build_timeout_s=30
        )
    assert obj.read_bytes().startswith(b"\x7fELF")
    report = json.loads((work / "kernel.host_optimization.json").read_text())
    assert report["arena"]["bound"] == 2 and report["arena"]["reuse_factor"] > 1
    assert report["status"] == "applied" and report["final_object_transform"] == "unchanged"
    assert "call ptr @malloc" in (work / "kernel.ll").read_text()
    assert "call ptr @malloc" not in (work / "kernel.host.ll").read_text()
    rows = [invocation_record.verify(path) for path in work.rglob("invocation.json")]
    translation = next(row for row in rows if row["stage"] == "llvm_translation")
    object_call = next(row for row in rows if row["stage"] == "object")
    assert translation["outputs"][0]["path"] == str(work / "kernel.ll")
    assert object_call["inputs"][0]["path"] == str(work / "kernel.host.ll")
    assert report["products"]["object"]["sha256"] == file_digest(obj)
    stack = json.loads((work / "kernel.stack_frame.json").read_text())
    assert stack["stack_usage_report_sha256"] == file_digest(work / "kernel.su")
    assert stack["stack_usage_measurement_view"]["sha256"] == file_digest(work / "kernel.measured.su")
    object_record = next(
        path for path in work.rglob("invocation.json") if json.loads(path.read_text())["stage"] == "object"
    )
    (work / "kernel.su").write_text("changed raw compiler report\n")
    with pytest.raises(ValueError, match="product changed"):
        invocation_record.verify(object_record)
