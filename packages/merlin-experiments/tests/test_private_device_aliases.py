"""An ABI alias preserves source-proven input transport without exempting helpers."""

import subprocess
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback.private_device_audit import audit_built_device_host_compute


@pytest.mark.parametrize("alias", [False, True])
@pytest.mark.parametrize("cpu_helper", [False, True])
def test_alias_body_gets_closed_input_transport_but_helpers_remain_audited(tmp_path, monkeypatch, alias, cpu_helper):
    from merlin.llvmlower import device_shim, toolchain
    from merlin.targetgen.rocc import decode
    from merlin.verify import conv_input_preparation

    monkeypatch.setattr(device_shim, "kernel_abi_for",
                        lambda _target: SimpleNamespace(symbol="entry" if alias else "body"))
    monkeypatch.setattr(toolchain, "nm", lambda: "nm")
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=""))

    def instruction(funct, rs1, rs2, isa):
        return (("CONFIG_LD", {"stride": 2, "state_id": (rs1["raw"] >> 3) & 3})
                if funct == 0 else ("MVIN", {"dram": rs1, "rows": 1, "cols": 2}))

    semantics = SimpleNamespace(word_bits=64, isa_constants=lambda target: {"CUSTOM_OPCODE": 0x7B},
                                decode_instruction=instruction)
    monkeypatch.setattr(decode, "_semantics", lambda target: SimpleNamespace(operand_observer=semantics))
    monkeypatch.setattr(decode, "isa_constants", lambda target: {
        "CUSTOM_OPCODE": 0x7B, "FUNCT_CLASS": {0: "CONFIG_LD", 2: "MVIN"}
    })
    alias_ir = '''"llvm.mlir.alias"() <{sym_name = "entry",
      alias_type = !llvm.func<void (!llvm.ptr, !llvm.ptr, !llvm.ptr)>}> ({
      %address = llvm.mlir.addressof @body : !llvm.ptr
      llvm.return %address : !llvm.ptr
    }) : () -> ()''' if alias else ""
    helper_ir = """llvm.func @cpu_helper() {
      %v0 = llvm.mlir.constant(1.0 : f32) : f32
      """ + "\n".join(f"%v{i} = llvm.fmul %v{i-1}, %v{i-1} : f32" for i in range(1, 65)) + "\nllvm.return\n}" \
        if cpu_helper else ""
    module = """module { ALIAS
      llvm.func @body(%input: !llvm.ptr, %weight: !llvm.ptr, %output: !llvm.ptr) {
        %one = llvm.mlir.constant(1) : i64
        %two = llvm.mlir.constant(2) : i64
        %scratch = llvm.alloca %two x i8 : (i64) -> !llvm.ptr
        %value = llvm.load %input : !llvm.ptr -> i8
        llvm.store %value, %scratch : i8, !llvm.ptr
        %input1 = llvm.getelementptr %input[1] : (!llvm.ptr) -> !llvm.ptr, i8
        %scratch1 = llvm.getelementptr %scratch[1] : (!llvm.ptr) -> !llvm.ptr, i8
        %value1 = llvm.load %input1 : !llvm.ptr -> i8
        llvm.store %value1, %scratch1 : i8, !llvm.ptr
        %address = llvm.ptrtoint %scratch : !llvm.ptr to i64
        "llvm.inline_asm"(%one, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x0, x0, $0, $1",
          constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
        "llvm.inline_asm"(%address, %one) <{asm_string = ".insn r 0x7b, 0x3, 0x2, x0, $0, $1",
          constraints = "r,r,~{memory}", has_side_effects}> : (i64, i64) -> ()
        llvm.return
      }
      HELPER
    }""".replace("ALIAS", alias_ir).replace("HELPER", helper_ir)
    device = tmp_path / "device"
    device.mkdir()
    (device / "shared.device.mlir").write_text(module)
    (device / "shared.ll").write_text("actual linked input")
    (device / "shared.o").write_bytes(b"actual linked object")
    preparation = {"call_buffers": [
        {"role": role, "shape": shape, "dtype": dtype} for role, shape, dtype in (
            ("input_0", [2], "i8"), ("weight_0", [1], "i8"), ("out_0", [1], "i32")
        )
    ]}
    source = {
        "eligible_groups": [4],
        "eligible_group_provenance": [
            {"source_group": 4, "source_region_id": "region-a", "source_node_ids": ["node-a"]}
        ],
        "eligible_group_metrics": {4: {"elements": 1, "element_bytes": 4}},
        "conv_input_preparation": {"shared": preparation},
    }
    assigned = [{"symbol": "shared", "group": 1, "source_region_id": "region-a", "source_node_ids": ["node-a"]}]
    prove = conv_input_preparation.prove_input_preparation
    observed = []

    def observe(function, activity, context, **kwargs):
        result = prove(function, activity, context, **kwargs)
        assert result["status"] == "proven", result
        observed.append((function.sym_name.data, context, result))
        return result

    monkeypatch.setattr(conv_input_preparation, "prove_input_preparation", observe)
    if cpu_helper:
        with pytest.raises(Exception, match="cpu_helper.*compute on the host|compute on the host.*cpu_helper"):
            audit_built_device_host_compute(tmp_path, assigned, source, "fixture")
    else:
        records = audit_built_device_host_compute(tmp_path, assigned, source, "fixture")
        assert records[0]["defined_functions_audited"] == 1
        report, = records[0]["audit"]["groups"]
        assert report["symbol"] == "body" and report["input_preparation"]["status"] == "proven"
        assert report["host_raw_payload_bytes"] == 4 and report["host_payload_bytes"] == 0
    assert len(observed) == 1 and observed[0][:2] == ("body", preparation)
    assert observed[0][2]["status"] == "proven"
