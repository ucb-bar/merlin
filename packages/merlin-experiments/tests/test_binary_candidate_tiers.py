"""Selected binary candidate readback survives the independent L2 re-audit.

These are synthetic pinned files and selected parser seams, not engine runs or
complete build qualifications. Numerical comparison and payload parsing are real.
"""

from __future__ import annotations

import base64
import hashlib
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from merlin.runtime.backends.base import parse_console
from merlin.runtime.out_bin import parse_binary_console
from merlin.targetgen import native_model_execution as native
from merlin.targetgen.capsule_golden import compare
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, FULL_VALUES_BIN, ReadbackPolicy
from merlin.targetgen.golden_store import load_golden, write_golden


def _frame(payload, transport):
    if transport == FULL_VALUES_BIN:
        checksum = 0xCBF29CE484222325
        for byte in payload:
            checksum = ((checksum ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
        return (
            f"OUT_BIN_BEGIN v1 out 1 {len(payload)} 1 u {len(payload)}\n".encode()
            + payload
            + f"OUT_BIN_END v1 {checksum:016x}\nDONE\n".encode()
        )
    return (
        f"OUT_B64_BEGIN v1 out 1 {len(payload)} 1 u\n"
        f"OUT_B64_CHUNK 00000000 {len(payload):04x} {base64.b64encode(payload).decode()}\n"
        "OUT_B64_END\nDONE\n"
    ).encode()


@pytest.fixture
def pinned_l2(tmp_path, monkeypatch):
    def case(payload=b"\xff\x00", transport=FULL_VALUES_BIN, *, matches=True):
        capsule = tmp_path / "capsule"
        capsule.mkdir(exist_ok=True)
        declaration = capsule / "capsule.yaml"
        declaration.write_text("numeric_policy:\n  compare: exact_int\n")
        expected = list(payload)
        if not matches:
            expected[-1] += 1
        write_golden(capsule, {"golden_source": "synthetic_test_only", "outputs": {"out": [expected]}})
        elf = tmp_path / "candidate.elf"
        elf.write_bytes(b"synthetic candidate ELF for binding test")
        console = tmp_path / "console_l2.data"
        console.write_bytes(_frame(payload, transport))
        cb = tmp_path / "command_buffer.json"
        cb.write_text(json.dumps({
            "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
            "tensors": {"out": {"shape": [1, len(payload)], "dtype": "u8", "role": "output"}},
            "commands": [],
        }))
        parser = parse_binary_console if transport == FULL_VALUES_BIN else parse_console
        raw_console = console.read_bytes() if transport == FULL_VALUES_BIN else console.read_text()
        observed, _ = parser(raw_console)
        numeric = compare(
            load_golden(capsule)["outputs"], observed, {"compare": "exact_int"},
            golden_source="synthetic_test_only",
        )
        tier_policy = {"required_tiers": ["L2"], "scope": "synthetic selected tier"}
        citation = {"binary": {"sha256": "synthetic_test_only"}}
        receipt = {
            "source": {
                "capsule_declaration": native._digest(declaration),
                "golden": native._digest(capsule / "golden.yaml"),
                "golden_arrays": native._digest(capsule / "golden.npz"),
            },
            "frozen_policy": tier_policy,
            "elf": native._digest(elf),
            "tiers": {"L2": {
                "status": numeric["status"], "engine": "spike", "engine_citation": citation,
                "elf": native._digest(elf), "console": native._digest(console), "numeric": numeric,
            }},
        }
        monkeypatch.setattr(native, "audit_candidate_static_tiers", lambda *_a, **_k: {})
        monkeypatch.setattr(native, "_frozen_model_policy", lambda *_a, **_k: tier_policy)
        monkeypatch.setattr(
            native, "_functional_engine",
            lambda *_a, **_k: (SimpleNamespace(parse_output=parser), citation, lambda: None),
        )

        def audit(*, selected=True):
            kwargs = {"readback_policy": ReadbackPolicy(transport)} if selected else {}
            return native.audit_candidate_tiers(
                {"command_buffer": native._digest(cb)}, {}, receipt,
                target="synthetic", entry_symbol="kernel", completed_dispatch=None, **kwargs,
            )

        return SimpleNamespace(audit=audit, console=console, receipt=receipt, cb=cb, numeric=numeric)

    return case


@pytest.mark.parametrize("payload", [b"\xff\x00", b"\x01\x02"])
def test_selected_binary_reaudit_keeps_arbitrary_payload_bytes(pinned_l2, payload):
    case = pinned_l2(payload)
    result = case.audit()
    assert result["tiers"]["L2"]["status"] == "pass"
    assert result["tiers"]["L2"]["numeric"] == case.numeric
    assert result["tiers"]["L2"]["console"] == native._digest(case.console)
    assert result["status"] == "pass" and "cycle_accurate" not in result["tiers"]["L2"]


@pytest.mark.parametrize("selected", [False, True])
def test_b64_default_and_explicit_policy_reaudit_are_unchanged(pinned_l2, selected):
    case = pinned_l2(b"\xff\x00", FULL_VALUES_B64)
    result = case.audit(selected=selected)
    assert result["status"] == "pass" and result["tiers"]["L2"]["numeric"] == case.numeric


def test_actual_binary_numerical_mismatch_remains_a_failure(pinned_l2):
    case = pinned_l2(matches=False)
    result = case.audit()
    assert result["status"] == "fail" and result["failed_tiers"] == ["L2"]
    assert result["tiers"]["L2"]["numeric"]["mismatch_count"] == 1


@pytest.mark.parametrize("damage", ["stale", "truncated", "checksum", "width", "unselected"])
def test_binary_reaudit_refuses_invalid_boundaries(pinned_l2, damage):
    case = pinned_l2()
    raw = case.console.read_bytes()
    if damage == "truncated":
        case.console.write_bytes(raw[:-5])
    elif damage in ("stale", "checksum"):
        case.console.write_bytes(raw.replace(b"\xff", b"\xfe", 1))
    elif damage == "width":
        case.console.write_bytes(_frame(b"\xff\x00\x00\x00", FULL_VALUES_BIN).replace(b"1 4 1 u 4", b"1 2 2 u 4"))
    if damage not in ("stale", "unselected"):
        case.receipt["tiers"]["L2"]["console"] = native._digest(case.console)
    result = case.audit(selected=damage != "unselected")
    assert result["tiers"]["L2"]["status"] == "unverified"
    assert result["status"] == "unverified"


def test_invalid_policy_capability_refuses_before_audit(pinned_l2):
    case = pinned_l2()
    with pytest.raises(ValueError, match="explicit trusted"):
        native.audit_candidate_tiers(
            {"command_buffer": native._digest(case.cb)}, {}, case.receipt,
            target="synthetic", entry_symbol="kernel", completed_dispatch=None,
            readback_policy={"transport": FULL_VALUES_BIN},
        )


def test_coherent_transport_cannot_reuse_serial_l2_values(pinned_l2):
    case = pinned_l2(transport=FULL_VALUES_B64)
    result = native.audit_candidate_tiers(
        {"command_buffer": native._digest(case.cb)}, {}, case.receipt,
        target="synthetic", entry_symbol="kernel", completed_dispatch=None,
        readback_policy=ReadbackPolicy("coherent_dump_v1"),
    )
    assert result["status"] == "unverified"
    assert "independent memory audit" in result["tiers"]["L2"]["detail"]


def test_normal_model_check_reopens_binary_only_under_trusted_selection(pinned_l2, monkeypatch):
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import capsule_grade as grade

    case = pinned_l2()
    monkeypatch.setattr(backends, "harness_build_recipe", lambda _t: SimpleNamespace(
        require_kernel_stack_frame=lambda: SimpleNamespace(entry_symbol="kernel"),
    ))
    for name in (
        "audit_emitted_host_compute", "audit_candidate_source_placement", "audit_candidate_completed_dispatch",
    ):
        monkeypatch.setattr(native, name, lambda *_a, **_k: {"status": "unverified"})

    def row():
        return {
            "candidate_emission": {"command_buffer": native._digest(case.cb)},
            "candidate_native_execution": case.receipt,
            # A candidate-authored record cannot grant transport selection.
            "readback_policy": ReadbackPolicy(FULL_VALUES_BIN).record(),
        }

    capsule = {"semantic": {"must_accelerate": True}}
    unselected = grade.enforce_model_execution_check(row(), capsule, target="synthetic")
    assert unselected["model_execution_check"]["candidate_required_tiers"]["tiers"]["L2"]["status"] == "unverified"
    selected = grade.enforce_model_execution_check(
        row(), capsule, target="synthetic", readback_policy=ReadbackPolicy(FULL_VALUES_BIN),
    )
    assert selected["model_execution_check"]["candidate_required_tiers"]["tiers"]["L2"]["status"] == "pass"
    # Missing build/source/dispatch authority remains missing despite decoding.
    assert selected["status"] == "incomplete"


@pytest.mark.parametrize("transport", [None, FULL_VALUES_B64, FULL_VALUES_BIN])
def test_model_child_forwards_original_context_policy_to_execution_and_reaudit(tmp_path, monkeypatch, transport):
    from merlin.targetgen import capsule_grade as grade
    from merlin.targetgen import capsule_runner as runner

    source = tmp_path / "capsule"
    source.mkdir()
    (source / "capsule.interface.mlir").write_text("module {}")
    (source / "capsule.yaml").write_text("name: model\nkind: model\n")
    generated = tmp_path / "generated"
    generated.mkdir()
    monkeypatch.setattr(runner, "make_run_paths", lambda *_a, **_k: SimpleNamespace(
        generated=generated, run_path=tmp_path / "run",
    ))

    def emit(*_args, **_kwargs):
        cb = {"kernel_abi": {"kind": "whole_program"}}
        (generated / "command_buffer.json").write_text(json.dumps(cb))
        (generated / "lowered.llvm.mlir").write_text("builtin.module {}")
        return object(), cb, "builtin.module {}"

    @contextmanager
    def bundle(*_args, **_kwargs):
        yield tmp_path, {"construction": "synthetic frozen"}, lambda: None

    expected = ReadbackPolicy(transport) if transport else None
    expected_kwargs = {"readback_policy": expected} if expected else {}
    calls = []

    def execute(**kwargs):
        calls.append(("execute", {key: value for key, value in kwargs.items() if key == "readback_policy"}))
        return {"status": "compiled_not_run"}

    def enforce(result, *_args, **kwargs):
        calls.append(("reaudit", {key: value for key, value in kwargs.items() if key == "readback_policy"}))
        return result

    monkeypatch.setattr(runner, "run_entrypoints", emit)
    monkeypatch.setattr(runner, "_model_runtime_bundle", bundle)
    monkeypatch.setattr(native, "independent_frozen_source_eligibility", lambda *_a, **_k: {"status": "synthetic"})
    monkeypatch.setattr(native, "execute_candidate_model", execute)
    monkeypatch.setattr(grade, "enforce_model_execution_check", enforce)
    runner._grade_candidate_model_capsule_inline(
        {"name": "model", "__dir__": str(source)}, target="synthetic", timeout=1,
        package_dir=tmp_path, pkg=object(), context={
            "runs_root": str(tmp_path), "run_id": "binding", "suite": "synthetic",
            "dtype": "u8", "contract": None, "fourth_output_name": "lowered.llvm.mlir",
            "package_prebuilt": True,
            **({"readback_policy": expected.record()} if expected else {}),
        },
    )
    assert calls == [("execute", expected_kwargs), ("reaudit", expected_kwargs)]


@pytest.mark.parametrize("transport", [None, FULL_VALUES_B64, FULL_VALUES_BIN])
def test_suite_reaudit_uses_selected_adapter_policy_not_result(tmp_path, monkeypatch, transport):
    from merlin.targetgen import capsule_grade as grade

    source = tmp_path / "capsule"
    source.mkdir()
    capsule = {"name": "model", "kind": "model", "label": "public", "__dir__": str(source)}
    monkeypatch.setattr(grade, "source_experiment_env", lambda _t: {})
    tool = tmp_path / "compiler.py"
    tool.write_text("# Synthetic package tool; compilation is separately stubbed below.\n")
    monkeypatch.setattr(grade, "load_package", lambda *_a, **_k: SimpleNamespace(integrity_exempt=False, tool=tool))
    monkeypatch.setattr(grade, "integrity_scan", lambda *_a, **_k: None)
    monkeypatch.setattr(grade, "build_package", lambda *_a, **_k: None)
    monkeypatch.setattr(grade.CR, "discover_capsules", lambda *_a, **_k: [capsule])
    monkeypatch.setattr(grade.CV, "aggregate", lambda *_a, **_k: {
        "by_tier_reached": {}, "instruction_class_coverage": {}, "mode_coverage": {},
        "unavailable": {}, "acceleratable_coverage": {},
    })
    calls = []

    def suite(*_args, **kwargs):
        results = [{
            "capsule": "model", "kind": "model", "label": "public", "status": "incomplete", "tiers": {},
            "readback_policy": ReadbackPolicy(FULL_VALUES_BIN).record(),
        }]
        kwargs["post_suite"](results)
        return results

    def enforce(result, *_args, **kwargs):
        calls.append({key: value for key, value in kwargs.items() if key == "readback_policy"})
        return result

    monkeypatch.setattr(grade.CR, "run_suite", suite)
    monkeypatch.setattr(grade, "enforce_model_execution_check", enforce)
    def adapter(*_args, **_kwargs):
        return None

    expected = ReadbackPolicy(transport) if transport else None
    if expected:
        adapter._merlin_readback_policy = expected
    grade.grade(
        tmp_path / "package", capsules_root=source, runs_root=tmp_path / "runs",
        target="synthetic", oracle_adapters={"L2": adapter},
    )
    assert calls == [({"readback_policy": expected} if expected else {})]


@pytest.fixture
def pinned_dispatch(pinned_l2, tmp_path, monkeypatch):
    """Real pinned readback/roster audit with explicitly synthetic ISA/build proofs."""
    from merlin.compile import model_execution_inputs
    from merlin.llvmlower import toolchain
    from merlin.perf import compiler_plan_evidence
    from merlin.runtime.backends import base as backends
    from merlin.targetgen import native_dispatch_accounting as dispatch
    from merlin.targetgen import oracle_policy
    from merlin.targetgen.rocc import decode

    def case(payload=b"\xff\x00", transport=FULL_VALUES_BIN, *, matches=True):
        result = pinned_l2(payload, transport, matches=matches)
        capsule = result.receipt["source"]["capsule_declaration"]["path"]
        from pathlib import Path

        capsule = Path(capsule).parent
        (capsule / "capsule.yaml").write_text(
            "numeric_policy:\n  compare: exact_int\ninterface_mlir: model.mlir\n"
            "operation:\n  attributes:\n    weights: weights.bin\n    weights_manifest: weights.json\n"
        )
        (capsule / "model.mlir").write_text("module {}")
        (capsule / "weights.bin").write_bytes(b"synthetic weights")
        (capsule / "weights.json").write_text("{}")
        for key, name in (
            ("capsule_declaration", "capsule.yaml"), ("interface", "model.mlir"),
            ("weights", "weights.bin"), ("weight_manifest", "weights.json"),
        ):
            result.receipt["source"][key] = native._digest(capsule / name)
        lowered = tmp_path / "lowered.mlir"
        lowered.write_text("builtin.module { llvm.func @kernel() { llvm.return } }")
        ll = tmp_path / "kernel.ll"
        ll.write_text("synthetic translated LLVM")
        obj = tmp_path / "kernel.o"
        obj.write_bytes(b"synthetic object")
        stack = tmp_path / "kernel.stack_frame.json"
        stack.write_text(json.dumps({
            "status": "passed", "repair": None, "entry_symbol": "kernel",
            "llvm_ir_sha256": native._digest(ll)["sha256"], "object_sha256": native._digest(obj)["sha256"],
        }))
        abi = tmp_path / "kernel.abi.json"
        abi.write_text(json.dumps({"object_sha256": native._digest(obj)["sha256"]}))
        result.receipt["build_artifacts"] = {
            key: native._digest(path) for key, path in (
                ("kernel.llvm.mlir", lowered), ("kernel.ll", ll), ("kernel.o", obj),
                ("kernel.stack_frame.json", stack), ("kernel.abi.json", abi),
            )
        }
        cb = json.loads(result.cb.read_text())
        result.receipt["candidate"] = {
            "lowered_mlir_sha256": native._digest(lowered)["sha256"],
            "command_buffer_sha256": hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest(),
        }
        facts = tmp_path / "facts.json"
        facts.write_text(json.dumps({"facts": {"interfaces": [{
            "name": "funct_decode_table", "custom_opcode": 19, "names": {},
        }]}}))
        selected = {"target": "synthetic", "config": "independent", "path": str(facts)}
        selection = {"available": True, "engine": "synthetic_rtl"}
        citation = {"binary": "synthetic_test_only"}
        result.receipt.update(
            status="numeric_match_diagnostic" if matches else "numeric_mismatch_diagnostic",
            simulator="synthetic_rtl", rtl_facts=selected, console=native._digest(result.console),
            simulator_provenance={"selection": selection, "citation": citation}, numeric=result.numeric,
            host_build_sources=[{"path": str(obj), "sha256": native._digest(obj)["sha256"]}],
        )
        monkeypatch.setattr(native, "audit_candidate_source_placement", lambda *_a, **_k: {
            "status": "clean", "source_sha256": native._digest(capsule / "model.mlir")["sha256"],
            "eligible_source_regions": [],
        })
        monkeypatch.setattr(compiler_plan_evidence, "verify_compiler_global_plan", lambda **_k: {
            "status": "verified", "control_flow": {"status": "verified"},
        })
        monkeypatch.setattr(model_execution_inputs, "selected_firrtl", lambda *_a, **_k: selected)
        monkeypatch.setattr(oracle_policy, "selected_l3_engine_report", lambda *_a, **_k: selection)
        parser = parse_binary_console if transport == FULL_VALUES_BIN else parse_console
        backend = SimpleNamespace(parse_output=parser)
        monkeypatch.setattr(model_execution_inputs, "native_engine", lambda *_a, **_k: (
            backend, citation, lambda: None, None,
        ))
        monkeypatch.setattr(backends, "get_backend", lambda _t: backend)
        monkeypatch.setattr(decode, "decode_module", lambda *_a, **_k: {
            "abi": {"custom_opcode": "0x13"}, "instructions": [],
        })
        monkeypatch.setattr(dispatch, "_verified_work_functs_by_family", lambda *_a: {})
        monkeypatch.setattr(dispatch, "_completed_eligible_tasks", lambda *_a: [{"scope": "synthetic fixture"}])
        monkeypatch.setattr(dispatch, "_kernel_command_inventory", lambda *_a, **_k: ([], []))
        monkeypatch.setattr(toolchain, "objdump", lambda: obj)

        def audit(*, selected_policy=True, readback_policy=None):
            return native.audit_candidate_completed_dispatch(
                {"command_buffer": native._digest(result.cb), "lowered_mlir": native._digest(lowered),
                 "source_interface": native._digest(capsule / "model.mlir")}, {}, result.receipt,
                target="synthetic", entry_symbol="kernel",
                **({"readback_policy": readback_policy or ReadbackPolicy(transport)} if selected_policy else {}),
            )

        result.dispatch_audit = audit
        return result

    return case


@pytest.mark.parametrize("payload", [b"\xff\x00", b"\x01\x02"])
def test_l3_completed_dispatch_reopens_actual_selected_binary(pinned_dispatch, payload):
    case = pinned_dispatch(payload)
    result = case.dispatch_audit()
    assert result["status"] == "verified" and result["numeric_status"] == "pass"
    assert result["console"] == native._digest(case.console)
    assert "cycle_accurate" not in result


@pytest.mark.parametrize("selected", [False, True])
def test_l3_b64_default_and_explicit_policy(pinned_dispatch, selected):
    assert pinned_dispatch(transport=FULL_VALUES_B64).dispatch_audit(selected_policy=selected)["status"] == "verified"


def test_coherent_transport_cannot_reuse_serial_dispatch_values(pinned_dispatch):
    case = pinned_dispatch(transport=FULL_VALUES_B64)
    result = case.dispatch_audit(readback_policy=ReadbackPolicy("coherent_dump_v1"))
    assert result["status"] == "unverified"
    assert "independent memory audit" in result["detail"]


def test_l3_binary_real_mismatch_is_verified_failure(pinned_dispatch):
    result = pinned_dispatch(matches=False).dispatch_audit()
    assert result["status"] == "verified" and result["numeric_status"] == "fail"


@pytest.mark.parametrize("damage", ["stale", "truncated", "checksum", "width", "unselected"])
def test_l3_binary_refusals_preserve_missing_authority(pinned_dispatch, damage):
    case = pinned_dispatch()
    raw = case.console.read_bytes()
    if damage == "truncated":
        case.console.write_bytes(raw[:-5])
    elif damage in ("stale", "checksum"):
        case.console.write_bytes(raw.replace(b"\xff", b"\xfe", 1))
    elif damage == "width":
        case.console.write_bytes(_frame(b"\xff\x00\x00\x00", FULL_VALUES_BIN).replace(b"1 4 1 u 4", b"1 2 2 u 4"))
    if damage not in ("stale", "unselected"):
        case.receipt["console"] = native._digest(case.console)
    assert case.dispatch_audit(selected_policy=damage != "unselected")["status"] == "unverified"
