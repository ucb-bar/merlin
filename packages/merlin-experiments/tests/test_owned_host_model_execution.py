"""Source call and device work cannot borrow another call's successful proof."""

import pytest

from merlin.targetgen.owned_host_model_execution import _host_call_proof, _mandatory_source_commands


def test_actual_host_cfg_preserves_each_reused_source_call(tmp_path):
    from merlin.llvmlower import toolchain

    if not toolchain.mlir_translate().is_file():
        pytest.skip("requires selected LLVM importer; no simulation")
    path = tmp_path / "host.ll"

    def source(body):
        path.write_text(
            'target triple = "riscv64-unknown-elf"\n'
            "declare void @device_call()\n"
            f"define void @forward(i1 %c) {{\n{body}\n}}\n"
            "define void @_mlir_ciface_forward(i1 %c) {\n"
            "call void @forward(i1 %c)\nret void\n}\n"
        )

    assigned = [{"symbol": "device_call"}, {"symbol": "device_call"}]
    source("call void @device_call()\ncall void @device_call()\nret void")
    assert _host_call_proof(path, assigned=assigned, output=tmp_path)["mandatory_call_roster"] == {"device_call": 2}
    source("call void @device_call()\nret void")
    with pytest.raises(ValueError, match="roster"):
        _host_call_proof(path, assigned=assigned, output=tmp_path)
    source(
        "br i1 %c, label %work, label %exit\nwork:\ncall void @device_call()\n"
        "call void @device_call()\nbr label %exit\nexit:\nret void"
    )
    with pytest.raises(ValueError, match="bypassed"):
        _host_call_proof(path, assigned=assigned, output=tmp_path)


def test_fused_owned_source_operations_need_individual_work_attribution():
    from xdsl.dialects.builtin import StringAttr
    from xdsl.parser import Parser

    from merlin.targetgen.oot_starterkit.llvm_context import make_llvm_context

    module = Parser(
        make_llvm_context(),
        """builtin.module {
      llvm.func @kernel(%p: !llvm.ptr) {
        %a = llvm.load %p : !llvm.ptr -> i64
        %b = llvm.load %p : !llvm.ptr -> i64
        llvm.return
      }
    }""",
    ).parse_module()
    function = next(op for op in module.body.block.ops if op.name == "llvm.func")
    first, second = [op for op in function.walk() if op.name == "llvm.load"]
    obligations = [{"source_op_index": i, "region_id": f"r{i}", "semantic_family": "contraction"} for i in range(2)]
    first.attributes["prov.region_id"] = StringAttr("r0")
    with pytest.raises(ValueError, match="operation 1"):
        _mandatory_source_commands(function, {"contraction": {first}}, obligations)
    second.attributes["prov.region_id"] = StringAttr("r1")
    assert len(_mandatory_source_commands(function, {"contraction": {first, second}}, obligations)) == 2
    obligations[1]["semantic_family"] = "elementwise_map"
    with pytest.raises(ValueError, match="elementwise_map"):
        _mandatory_source_commands(function, {"contraction": {first, second}}, obligations)


def test_claimed_numeric_result_without_actual_owned_build_never_promotes():
    from merlin.targetgen.capsule_grade import enforce_model_execution_check

    capsule = {"semantic": {"must_accelerate": True}, "required_oracle_tiers": ["L0", "L1", "L2", "L3"]}
    result = {
        "status": "pass",
        "kind": "model",
        "owned_host_native_execution": {
            "status": "verified",
            "numeric": {"status": "pass"},
            "tiers": {tier: {"status": "pass"} for tier in capsule["required_oracle_tiers"]},
        },
    }
    checked = enforce_model_execution_check(result, capsule, target="synthetic")
    assert checked["status"] == "incomplete"
    assert checked["model_execution_check"]["status"] == "incomplete"
