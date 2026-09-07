"""Exact target-edge evidence for the generic deployment-admissibility gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from merlin.perf.deployment_admissibility import assess_deployment_admissibility
from merlin.runtime.backends import base as backends

gemmini_backend = backends.get_backend("gemmini")
deployment_evidence = gemmini_backend.gemmini_deployment_evidence
derive_physical_egress_evidence = deployment_evidence.derive_physical_egress_evidence
derive_wrapper_event_evidence = deployment_evidence.derive_wrapper_event_evidence
profile_with_sha256 = deployment_evidence.profile_with_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: str | bytes) -> tuple[Path, str]:
    path.write_bytes(payload.encode() if isinstance(payload, str) else payload)
    return path, _sha(path)


def _profile(tmp_path: Path):
    bindings = {}
    for role in ("contract", "config", "runtime_header", "bitstream"):
        bindings[role] = _write(tmp_path / f"{role}.bin", f"exact {role}\n")
    return profile_with_sha256(
        contract_path=bindings["contract"][0], contract_sha256=bindings["contract"][1],
        config_path=bindings["config"][0], config_sha256=bindings["config"][1],
        runtime_header_path=bindings["runtime_header"][0],
        runtime_header_sha256=bindings["runtime_header"][1],
        bitstream_path=bindings["bitstream"][0], bitstream_sha256=bindings["bitstream"][1],
        supported_physical_egress=[{"encoding": "signed_integer", "width_bits": 8}],
    )


def _command_buffer(output_dtype: str) -> dict:
    attributes = {"epilogue": [], "output_dtype": output_dtype}
    if output_dtype == "i8":
        attributes["acc_scale"] = 1.0
    return {
        "abi_version": "0.1",
        "target": "gemmini",
        "tensors": {
            "W": {"shape": [16, 16], "dtype": "i8", "role": "weight"},
            "A": {"shape": [16, 16], "dtype": "i8", "role": "input"},
            "Y": {"shape": [16, 16], "dtype": output_dtype, "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "R"},
             "attributes": {"layout": "packed_rhs"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A", "rhs": "R", "dst": "acc"}},
            {"opcode": "COMMIT", "operands": {"src": "acc", "dst": "Y"},
             "attributes": attributes},
        ],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": "W", "access": "read"},
                     {"tensor": "A", "access": "read"},
                     {"tensor": "Y", "access": "write"}],
            "outputs": ["Y"],
        },
        "outputs": ["Y"],
    }


def _wrapper(old_order: bool = False) -> str:
    measured = ("gemmini_kernel();\nuint64_t c1 = read_cycles();\ngemmini_fence();"
                if old_order else
                "gemmini_kernel();\ngemmini_fence();\nuint64_t c1 = read_cycles();")
    return ("extern void gemmini_kernel();\nint main() {\n"
            "gemmini_kernel();\ngemmini_fence();\n"
            "uint64_t c0 = read_cycles();\n" + measured + "\n"
            'printf("OUT Y 1 1");\nreturn 0;\n}\n')


def _loop_hardware_and_facts():
    hardware = '''hw.module @LoopMatmul(in %clk : !seq.clock, in %ready : i1) {
    %dc_selector = hw.constant 11 : i7
    %dc_match = comb.icmp bin eq %cmd_q.io_deq_bits_cmd_inst_funct, %dc_selector : i7
    %dc_gate = comb.and bin %ready, %dc_match : i1
    %c_addr = comb.extract %cmd_q.io_deq_bits_cmd_rs2 from 0 : (i64) -> i40
    %c_next = comb.mux bin %dc_gate, %c_addr, %loops_0_c_dram_addr : i40
    %loops_0_c_dram_addr = seq.firreg %c_next clock %clk : i40
    %run_selector = hw.constant 8 : i7
    %run_match = comb.icmp bin eq %cmd_q.io_deq_bits_cmd_inst_funct, %run_selector : i7
    %run_gate = comb.and bin %ready, %run_match : i1
    %full = comb.extract %cmd_q.io_deq_bits_cmd_rs1 from 1 : (i64) -> i1
    %full_next = comb.mux bin %run_gate, %full, %loops_0_full_c : i1
    %loops_0_full_c = seq.firreg %full_next clock %clk : i1
    hw.output
  }'''
    facts = {
        "inputs": {"core_hw_sha256": hashlib.sha256(hardware.encode()).hexdigest()},
        "facts": {
            "target": "gemmini",
            "arrays": [{"name": "mesh", "rows": 16, "cols": 16}],
            "datapaths": [
                {"name": "input", "dtype": "i8", "evidence": "synthetic scratchpad path"},
                {"name": "accumulator", "dtype": "i32", "evidence": "synthetic accumulator path"},
            ],
            "memories": [
                {"name": "scratchpad", "bytes": 4096, "depth": 16, "source": "synthetic RTL"},
                {"name": "accumulator", "bytes": 4096, "depth": 16, "source": "synthetic RTL"},
            ],
            "interfaces": [{
                "name": "funct_decode_table",
                "names": {"8": "LOOP_WS", "11": "LOOP_WS_CONFIG_ADDRS_DC"},
                "legal_funct": [8, 11], "custom_opcode": 123, "funct3": 3,
            }],
        },
    }
    return hardware, facts


def _loop_lowered() -> str:
    return '''module {
  llvm.func @gemmini_kernel(%a0: !llvm.ptr, %a1: !llvm.ptr,
                            %a2: !llvm.ptr, %a3: !llvm.ptr) {
    %zero = llvm.mlir.constant(0 : i64) : i64
    %full = llvm.mlir.constant(2 : i64) : i64
    %out0 = llvm.ptrtoint %a2 : !llvm.ptr to i64
    %out1 = llvm.ptrtoint %a3 : !llvm.ptr to i64
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 11, x0, $0, $1", "r,r"
      %zero, %out0 : (i64, i64) -> ()
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 8, x0, $0, $1", "r,r"
      %zero, %zero : (i64, i64) -> ()
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 11, x0, $0, $1", "r,r"
      %zero, %out1 : (i64, i64) -> ()
    llvm.inline_asm has_side_effects ".insn r 0x7b, 0x3, 8, x0, $0, $1", "r,r"
      %full, %zero : (i64, i64) -> ()
    llvm.return
  }
}
'''


def _whole_program_loop_command_buffer() -> dict:
    return {
        "abi_version": "0.1", "target": "gemmini",
        "tensors": {
            "I0": {"shape": [16, 16], "dtype": "i8", "role": "input"},
            "I1": {"shape": [16, 16], "dtype": "i8", "role": "input"},
            "Y8": {"shape": [16, 16], "dtype": "i8", "role": "intermediate"},
            "Y32": {"shape": [16, 16], "dtype": "i32", "role": "output"},
        },
        "commands": [],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": name, "access": "write" if name.startswith("Y") else "read"}
                     for name in ("I0", "I1", "Y8", "Y32")],
            "outputs": ["Y32"],
        },
        "params": {"global_program_plan": {"tasks": [
            {"kind": "contraction", "reads": ["I0"], "writes": ["Y8"]},
            {"kind": "host", "reads": ["Y8"], "writes": []},
            {"kind": "convolution", "reads": ["I1"], "writes": ["Y32"]},
        ]}},
    }


def _evidence(tmp_path: Path, profile_sha: str, output_dtype: str, *, old_wrapper=False):
    cb = _command_buffer(output_dtype)
    backend = backends.get_backend("gemmini")
    lowered = backend.gemmini_codegen_mlir.emit_kernel_mlir(cb)[0]
    cb_path, cb_sha = _write(tmp_path / f"{output_dtype}.json", json.dumps(cb, sort_keys=True))
    lowered_path, lowered_sha = _write(tmp_path / f"{output_dtype}.mlir", lowered)
    object_path, object_sha = _write(tmp_path / f"{output_dtype}.o", b"exact object")
    wrapper_path, wrapper_sha = _write(tmp_path / f"{output_dtype}.c", _wrapper(old_wrapper))
    candidate_sha = hashlib.sha256(b"candidate").hexdigest()
    identity = {
        "candidate_sha256": candidate_sha,
        "command_buffer_sha256": cb_sha,
        "lowered_sha256": lowered_sha,
        "object_sha256": object_sha,
    }
    egress = derive_physical_egress_evidence(
        profile_sha256=profile_sha, candidate_sha256=candidate_sha,
        command_buffer_path=cb_path, command_buffer_sha256=cb_sha,
        lowered_path=lowered_path, lowered_sha256=lowered_sha,
        object_path=object_path, object_sha256=object_sha)
    wrapper = derive_wrapper_event_evidence(
        wrapper_path=wrapper_path, wrapper_sha256=wrapper_sha,
        profile_sha256=profile_sha, emission_identity=identity)
    return identity, egress, wrapper


def _assess(tmp_path: Path, profile, profile_sha, identity, egress, wrapper):
    return assess_deployment_admissibility(
        profile, expected_profile_sha256=profile_sha,
        expected_emission_identity=identity,
        egress_evidence=egress, wrapper_evidence=wrapper,
        profile_root=tmp_path, wrapper_root=tmp_path)


def test_actual_narrow_readout_is_admitted(tmp_path):
    profile, profile_sha = _profile(tmp_path)
    identity, egress, wrapper = _evidence(tmp_path, profile_sha, "i8")

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert egress["producer_status"] == "verified"
    assert egress["egresses"] == [{
        "name": "Y", "status": "verified",
        "emitted_representation": {"encoding": "signed_integer", "width_bits": 8},
        "physical_readout": {"encoding": "signed_integer", "width_bits": 8},
        "actual_readout_instruction_count": 1,
    }]
    assert result["status"] == "admitted"


def test_actual_full_width_readout_is_refused_by_narrow_profile(tmp_path):
    profile, profile_sha = _profile(tmp_path)
    identity, egress, wrapper = _evidence(tmp_path, profile_sha, "i32")

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert egress["egresses"][0]["physical_readout"]["width_bits"] == 32
    assert result["status"] == "refused"
    assert "unsupported_physical_egress" in [
        row["code"] for row in result["ranked_actionable_diagnostics"]]


def test_old_end_before_fence_wrapper_is_refused(tmp_path):
    profile, profile_sha = _profile(tmp_path)
    identity, egress, wrapper = _evidence(tmp_path, profile_sha, "i8", old_wrapper=True)

    result = _assess(tmp_path, profile, profile_sha, identity, egress, wrapper)

    assert [row["event"] for row in wrapper["events"]][-3:] == ["end", "completion", "validation"]
    assert result["status"] == "refused"
    assert "measured_end_before_completion" in [
        row["code"] for row in result["ranked_actionable_diagnostics"]]


def test_unmatched_emitted_output_is_unknown_not_guessed(tmp_path):
    profile, profile_sha = _profile(tmp_path)
    cb = _command_buffer("i8")
    cb["kernel_abi"]["outputs"] = ["not_the_readout"]
    backend = backends.get_backend("gemmini")
    lowered = backend.gemmini_codegen_mlir.emit_kernel_mlir(cb)[0]
    cb_path, cb_sha = _write(tmp_path / "unmatched.json", json.dumps(cb, sort_keys=True))
    lowered_path, lowered_sha = _write(tmp_path / "unmatched.mlir", lowered)
    object_path, object_sha = _write(tmp_path / "unmatched.o", b"object")

    evidence = derive_physical_egress_evidence(
        profile_sha256=profile_sha, candidate_sha256=hashlib.sha256(b"candidate").hexdigest(),
        command_buffer_path=cb_path, command_buffer_sha256=cb_sha,
        lowered_path=lowered_path, lowered_sha256=lowered_sha,
        object_path=object_path, object_sha256=object_sha)

    assert evidence["producer_status"] == "UNKNOWN"
    assert evidence["missing_expected_egresses"] == ["not_the_readout"]
    assert evidence["metadata_gap"]["policy"] == "do not infer physical egress from command declarations alone"


def test_host_event_manifest_must_bind_exact_wrapper_token_stream(tmp_path):
    profile, profile_sha = _profile(tmp_path)
    identity, _egress, standard = _evidence(tmp_path, profile_sha, "i8")
    wrapper_path = Path(standard["wrapper_artifact"]["path"])
    manifest = {
        "schema": "host_wrapper_event_manifest_v1",
        "wrapper_sha256": standard["wrapper_artifact"]["sha256"],
        "token_stream_sha256": standard["token_stream_sha256"],
        "events": standard["events"],
    }
    manifest_path, manifest_sha = _write(
        tmp_path / "wrapper_events.json", json.dumps(manifest, sort_keys=True))

    evidence = derive_wrapper_event_evidence(
        wrapper_path=wrapper_path,
        wrapper_sha256=standard["wrapper_artifact"]["sha256"],
        profile_sha256=profile_sha, emission_identity=identity,
        event_manifest_path=manifest_path, event_manifest_sha256=manifest_sha)

    assert evidence["derivation_status"] == "verified"
    assert evidence["source_kind"] == "host_event_manifest"
    assert evidence["events"] == standard["events"]


def test_representative_whole_program_joins_fused_loop_task_writes(tmp_path):
    cb = _whole_program_loop_command_buffer()
    hardware, facts = _loop_hardware_and_facts()
    cb_path, cb_sha = _write(tmp_path / "whole.json", json.dumps(cb, sort_keys=True))
    lowered_path, lowered_sha = _write(tmp_path / "whole.mlir", _loop_lowered())
    object_path, object_sha = _write(tmp_path / "whole.o", b"exact object")
    facts_path, facts_sha = _write(tmp_path / "facts.json", json.dumps(facts, sort_keys=True))
    hardware_path, hardware_sha = _write(tmp_path / "core.hw.mlir", hardware)

    evidence = derive_physical_egress_evidence(
        profile_sha256=hashlib.sha256(b"profile").hexdigest(),
        candidate_sha256=hashlib.sha256(b"candidate").hexdigest(),
        command_buffer_path=cb_path, command_buffer_sha256=cb_sha,
        lowered_path=lowered_path, lowered_sha256=lowered_sha,
        object_path=object_path, object_sha256=object_sha,
        rtl_facts_path=facts_path, rtl_facts_sha256=facts_sha,
        elaborated_hardware_path=hardware_path,
        elaborated_hardware_sha256=hardware_sha)

    assert evidence["producer_status"] == "verified"
    assert evidence["unknown_instruction_indices"] == []
    assert evidence["missing_expected_egresses"] == []
    assert evidence["egresses"] == [
        {"name": "Y32", "status": "verified",
         "emitted_representation": {"encoding": "signed_integer", "width_bits": 32},
         "physical_readout": {"encoding": "signed_integer", "width_bits": 32},
         "actual_readout_instruction_count": 1},
        {"name": "Y8", "status": "verified",
         "emitted_representation": {"encoding": "signed_integer", "width_bits": 8},
         "physical_readout": {"encoding": "signed_integer", "width_bits": 8},
         "actual_readout_instruction_count": 1},
    ]
    loop = evidence["fused_loop_writeback_evidence"]
    assert loop["coverage_status"] == "complete"
    assert [row["destination"]["arg_index"] for row in loop["writebacks"]] == [2, 3]
