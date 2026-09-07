"""Controller routing and fail-closed correspondence; no simulator or model run."""
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.perf import host_physical_transition_qualifier as Q


def source_chain():
    rows = []
    for i in range(2):
        incoming = "%arg" if i == 0 else "%p0"
        rows.append(f'''%e{i} = tensor.empty() : tensor<3x3xi8>
%p{i} = linalg.generic {{indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,
affine_map<(d0,d1)->(d0,d1)>], iterator_types = ["parallel", "parallel"]}}
ins({incoming}:tensor<3x3xi8>) outs(%e{i}:tensor<3x3xi8>) {{
^bb0(%x:i8,%unused:i8):
%one = arith.constant 1 : i8
%v = arith.addi %x, %one : i8
linalg.yield %v : i8
}} -> tensor<3x3xi8>''')
    return 'builtin.module {func.func @source(%arg:tensor<3x3xi8>)->tensor<3x3xi8>{\n' + '\n'.join(rows) + '\nfunc.return %p1:tensor<3x3xi8>}}'


def declaration(shape=(3, 3), reverse=True):
    return {"id": "copy", "kind": "static_strided_copy",
        "source_edge": {"producer_op_index": 1, "producer_result_index": 0,
                        "consumer_op_index": 3, "consumer_operand_index": 0},
        "source": {"dtype": "i8", "placement": "memory"},
        "destination": {"dtype": "i8", "placement": "memory"},
        "source_layout": {"shape": list(shape), "strides_elements": [shape[1], 1], "offset_elements": 0},
        "destination_layout": {"shape": list(shape), "strides_elements": [1, shape[0]] if reverse else [shape[1], 1],
                               "offset_elements": 0}}


def test_correspondence_compares_orientation_and_type_not_shape_or_identity():
    small, large = declaration(), declaration((16, 48))
    large["id"] = "other-copy-id"
    assert Q._mechanism(small) == Q._mechanism(large)
    assert Q._mechanism(small) != Q._mechanism(declaration(reverse=False))
    with pytest.raises(ValueError, match="dense axis"):
        small["source_layout"]["strides_elements"] = [4, 1]
        Q._mechanism(small)


@pytest.mark.parametrize("marked", ["none", "declaration", "llvm", "malformed"])
def test_dispatch_never_falls_back_after_marked_copy_refusal(marked):
    artifact = {"command_buffer": {"params": {"global_program_plan": {}}}, "lowered_text": ""}
    if marked == "declaration": artifact["command_buffer"]["params"]["global_program_plan"]["physical_transitions"] = [declaration()]
    elif marked == "llvm": artifact["lowered_text"] = '"merlin.transition_source" = "copy"'
    elif marked == "malformed": artifact["command_buffer"]["params"]["global_program_plan"]["physical_transitions"] = None
    calls = []
    class Provider:
        abi_provenance = {"host": "test"}
        def __init__(self, name): self.name = name
        def __call__(self, **kwargs):
            calls.append(self.name)
            return {"status": "UNKNOWN"}
    dispatch = Q.ChangedRegionQualifierDispatch(physical=Provider("physical"), legacy=Provider("legacy"))
    experiment = SimpleNamespace(previous_artifacts=lambda _: {}, current_artifacts=lambda _: artifact)
    assert dispatch(candidate=Path("candidate"), experiment=experiment, timeout_s=60)["status"] == "UNKNOWN"
    assert calls == ["legacy" if marked == "none" else "physical"]


def experiment_fixture(tmp_path, monkeypatch, *, mode="match"):
    source = tmp_path / "full.mlir"
    source.write_text(source_chain())
    llvm = 'builtin.module { llvm.func @kernel(%a: !llvm.ptr, %b: !llvm.ptr) { llvm.return } }'
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    def cb(rows):
        return {"params": {"global_program_plan": {"physical_transitions": rows}},
                "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}]}}
    before, after = ({"interface": source, "lowered_text": llvm,
        "candidate_lowered_sha256": sha(llvm), "command_buffer": cb(rows)} for rows in ([], [declaration()]))
    plan = {"status": "verified", "source_sha256": sha(source.read_text()), "candidate_lowered_sha256": sha(llvm)}
    iterations = [{"analysis": {"diagnostics": {"verified_global_plan_emission": deepcopy(plan)}}} for _ in range(2)]
    if mode == "stale": iterations[-1]["analysis"]["diagnostics"]["verified_global_plan_emission"]["candidate_lowered_sha256"] = "0"*64
    # Proof engine itself has independent real typed-LLVM regression tests. Here
    # isolate orchestration and source-edge correspondence from its parser.
    monkeypatch.setattr(Q, "verify_physical_transitions", lambda **kw:
        {"status": "verified" if kw["command_buffer"]["params"]["global_program_plan"]["physical_transitions"] else "not_declared"})
    calls = []
    def compile_arm(arm, candidate, interface, scratch, **kwargs):
        calls.append(arm)
        rows = [] if arm == "before" or mode == "missing" else [declaration(reverse=mode != "orientation")]
        if mode == "edge" and rows: rows[0]["source_edge"]["consumer_operand_index"] = 1
        return {"lowered": SimpleNamespace(returncode=0, stdout=llvm, stderr=""),
                "command_buffer_emission": SimpleNamespace(returncode=0), "command_buffer": cb(rows)}
    experiment = SimpleNamespace(previous_artifacts=lambda _: before, current_artifacts=lambda _: after,
        iterations=iterations,
        compile_previous_probe_candidate=lambda *a, **k: compile_arm("before", *a, **k),
        compile_probe_candidate=lambda *a, **k: compile_arm("after", *a, **k))
    return experiment, calls


@pytest.mark.parametrize("mode", ["match", "missing", "orientation", "edge", "stale"])
def test_both_compilers_must_reproduce_the_changed_source_mechanism(tmp_path, monkeypatch, mode):
    experiment, calls = experiment_fixture(tmp_path, monkeypatch, mode=mode)
    provider = Q.HostPhysicalTransitionQualifier(native_layout=lambda _: {}, expected_symbol="kernel",
        abi_provenance={"fixture": "native"}, output=tmp_path / "receipts")
    native = []
    def native_stub(candidate, exp, work, *args):
        native.append(work.name)
        return {"cases": [{"exact_output_bits": True}]}
    monkeypatch.setattr(provider, "_native", native_stub)
    receipt = provider(candidate=Path("candidate"), experiment=experiment, timeout_s=60)
    assert Path(receipt["detail_path"]).is_file()
    assert receipt["full_model_executed"] is False and receipt["cycles"] is None
    if mode == "match":
        assert receipt["status"] == "passed", receipt
        assert calls == native == ["before", "after"]
    else:
        assert receipt["status"] == "UNKNOWN", receipt
        assert native == []
        assert calls == ([] if mode == "stale" else ["before", "after"])


def test_native_bound_refuses_wrong_symbol_and_large_real_allocation():
    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
    from xdsl.dialects.llvm import LLVM
    context = make_context()
    context.load_dialect(LLVM)
    text = '''builtin.module {llvm.func @kernel(%a: !llvm.ptr) {
      %n = llvm.mlir.constant(1000000 : i64) : i64
      %p = llvm.alloca %n x i8 : (i64) -> !llvm.ptr
      llvm.return
    }}'''
    module = parse_mlir_text(text, context)
    with pytest.raises(ValueError, match="entry symbol"):
        Q._bounded_native_module(module, expected_symbol="wrong", argument_count=1)
    with pytest.raises(ValueError, match="allocation/work"):
        Q._bounded_native_module(module, expected_symbol="kernel", argument_count=1)


def test_unknown_intrinsic_and_module_initializer_refuse_before_native():
    from xdsl.dialects.builtin import ModuleOp
    from xdsl.ir import Operation
    from merlin.frontends.linalg_mlir import make_context, parse_mlir_text
    from xdsl.dialects.llvm import LLVM
    context = make_context()
    context.load_dialect(LLVM)
    for where in ("body", "module"):
        module = parse_mlir_text('builtin.module {llvm.func @kernel() {llvm.return}}', context)
        if where == "body":
            class Intrinsic(Operation):
                name = "llvm.call_intrinsic"
            block = module.body.block.first_op.body.blocks.first
            block.insert_op_before(Intrinsic.create(), block.last_op)
        else:
            class Initializer(Operation):
                name = "llvm.mlir.global"
            module.body.block.add_op(Initializer.create())
        with pytest.raises(ValueError, match="closed native"):
            Q._bounded_native_module(module, expected_symbol="kernel", argument_count=0)
