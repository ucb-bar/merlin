"""Opt-in original-host policy binding on the actual post-outline public model.

Missing cost keeps the exact prepared source. This is a functional/custody test,
not an automatic candidate decision, timing transfer or Phase 2 release grant.
"""

import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import recomputation_execution_inputs as R
import recomputation_host_work as W
import yaml
from packing_execution_inputs import (
    PUBLIC_COMPILER_SHA,
    actual_observation,
    pin,
    read_selection,
    recheck_artifacts,
    recheck_selection,
)
from test_public_packing_execution import _trace_contract
from test_public_recomputation_execution import SELECTION_ENV, _all_pins, _products, _trace

from merlin.llvmlower import compilation_recipe as CR
from merlin.llvmlower import source_recomputation_policy as P
from merlin.runtime import source_recomputation_build as B
from merlin.transforms.certificates import ir_digest
from merlin.transforms.pipeline_state import PipelineContext
from merlin.transforms.recomputation import RecomputationBudgets, RecomputationPermissions

pytestmark = pytest.mark.target("gemmini")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _source_identity(source):
    module = R.parse_mlir_text(source)
    module.verify()
    return {
        "file_sha256": ir_digest(source),
        "canonical_ir_sha256": ir_digest(module),
        "has_call": any(op.name == "func.call" for op in module.walk()),
    }


def _build_owner(*, case, document, tools, check, live):
    """Original fixed-source test premises; no law about opaque target effects."""
    context = PipelineContext(
        hardware=_digest({name: document[name] for name in ("facts", "board_catalog", "dts")}),
        numerical_policy="fixed-public-exact-modular-i8-i32-K5-prefix-within-signed20",
        resources="engineering-fixture-only; formal-resource-certificate-OPEN",
        execution=_digest({"quiet": document["quiet_contract"], "tools": tools}),
        library=_digest(
            {
                "public_compiler": PUBLIC_COMPILER_SHA,
                "source_and_shared_merlin": [row["sha256"] for row in case.source_pins],
                "policy_owner": pin(Path(P.__file__).resolve())["sha256"],
                "ordinary_build_owner": pin(Path(B.__file__).resolve())["sha256"],
            }
        ),
    )
    state = {"source": None, "plan": None, "final": []}
    check_code = check.__code__

    def verify_current(**current):
        assert check.__code__ is check_code
        check()
        assert current["comparison"] is None, "this test has no qualified complete cost"
        assert current["settings"]["index_bits"] == 64 and current["settings"]["targets"] == ()
        assert not current["settings"]["shared_source_lowerings_selected"]
        assert current["plan"]["stamp"]["context"] == asdict(context)
        source = current["source"]
        identity = _source_identity(source)
        assert identity["has_call"], "bind after actual device outlining"
        assert identity["canonical_ir_sha256"] != _source_identity(case.source)["canonical_ir_sha256"]
        assert current["plan"]["stamp"]["source_sha256"] == identity["canonical_ir_sha256"]
        if state["source"] is None:
            state["source"], state["plan"] = source, json.loads(json.dumps(current["plan"]))
        else:
            assert source == state["source"] and _digest(current["plan"]) == _digest(state["plan"])
        live["prepared"].require_facts(live["facts"])

    def verify_final(**current):
        assert check.__code__ is check_code
        check()
        report = current["selection"]
        assert report["status"] == "MISSING_COST" and report["selected"] is False
        assert report["comparison"] is None
        assert report["plan"] == state["plan"]
        assert report["source_file_sha256"] == report["selected_file_sha256"] == ir_digest(state["source"])
        actual = current["context"]
        assert actual["lowering"]["index_bits"] == 64
        assert actual["build"]["backend"] == "scalar" and actual["build"]["device_requested"]
        assert actual["build"]["host_math_policy"] == "native"
        compiler = actual["build"]["compiler_plan"]["observation"]
        assert compiler["index_bits"] == 64 and compiler["compiler_sha256"] == tools["clang"]["sha256"]
        assert str(Path(actual["build"]["gcc"]).resolve()) == tools["gcc"]["path"]
        products = current["products"]
        elf_rows = [row for row in products if Path(row["path"]).name == "model.elf"]
        assert len(elf_rows) == 1
        elf = Path(elf_rows[0]["path"])
        recipe = next(row for row in products if Path(row["path"]).name == CR.FILENAME)
        CR.verify_completed_recipe(Path(recipe["path"]), executable=elf, expected_recipe_sha256=recipe["sha256"])
        policy_path, observation = actual_observation(elf.parent, "device_final_linked_elf_policy")
        admission = json.loads(Path(observation["stdout"]["path"]).read_bytes())
        assert live["prepared"].service.revalidate(elf=elf, result=admission, target="gemmini") == "accepted"
        live["prepared"].require_facts(live["facts"])
        if current["stage"] == "executed":
            execution = current["execution"]
            assert execution["run"] == "spike"
            assert execution["output"]["elf"] == str(elf)
            assert execution["output"]["elf_sha256"] == elf_rows[0]["sha256"]
            assert "outputs_npz" in execution["output"]
        else:
            assert current["stage"] == "linked" and current["execution"] is None
        state["final"].append({"stage": current["stage"], "elf": pin(elf), "policy": pin(policy_path)})
        check()

    policy = P.SourceRecomputationPolicy(
        context,
        RecomputationPermissions(
            "original-fixed-public-total-integer-host-source-v1",
            repeat_evaluation_no_faults=True,
            repeat_evaluation_unobserved=True,
            suppressed_materialization_no_faults=True,
            suppressed_materialization_unobserved=True,
            inputs_stable_during_region=True,
        ),
        RecomputationBudgets(8, 4096, 2, 32, 4096, 4096, 65536),
        verify_current=verify_current,
    )
    return B.SourceRecomputationBuild(policy, verify_final=verify_final), state


def test_original_public_compiler_binds_missing_cost_after_outlining(tmp_path, monkeypatch):
    selected = os.environ.get(SELECTION_ENV)
    if not selected:
        pytest.skip(f"set {SELECTION_ENV}; absent original inputs confer no target qualification")
    document, selected_pin = read_selection(selected)
    if document["trace_contract"] is None:
        pytest.skip("select the original trace runtime; accelerator execution is unestablished")
    from merlin_experiments.phase1.feedback import private_instruction_coordinator as C
    from merlin_experiments.phase1.feedback import private_instruction_declaration as D

    from merlin.compile.baremetal_model import compile_saved_model, read_full_outputs
    from merlin.compile.model_execution_inputs import selected_execution_firrtl, strict_tree_sha256
    from merlin.compile.route_before_build import plan_before_build
    from merlin.llvmlower import optional_passes, toolchain
    from merlin.runtime.backends import spike
    from merlin.runtime.input_observation import revalidate_observation
    from merlin.targetgen import target_registry

    case = R.original_case()
    quiet_path = Path(document["quiet_contract"]["path"])
    trace_path = Path(document["trace_contract"]["path"])
    quiet, trace = yaml.safe_load(quiet_path.read_bytes()), yaml.safe_load(trace_path.read_bytes())
    _trace_contract(quiet, trace)
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    monkeypatch.setenv("MERLIN_RTL_FACTS", document["facts"]["path"])
    monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
    block = quiet["runner"]["chipyard_rocc"]
    monkeypatch.setenv("MERLIN_SPIKE", block["engines"]["spike"]["binary"]["path"])
    monkeypatch.setenv("MERLIN_RISCV_GCC", block["toolchain"]["compiler"]["path"])
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    tools = {"clang": pin(toolchain.clang().resolve(strict=True)), "gcc": pin(spike.gcc_path().resolve(strict=True))}
    observer_pins = tuple(pin(path) for path in W.observer_paths())
    selected_files = tuple(row for row in document.values() if type(row) is dict and set(row) == {"path", "sha256"})
    original_pins = (
        *case.source_pins,
        pin(Path(__file__).resolve()),
        selected_pin,
        *selected_files,
        *observer_pins,
        *tools.values(),
    )
    controls = {
        key: os.environ.get(key)
        for key in (optional_passes.ENV, *(r.switch for r in optional_passes.REGISTRY if r.switch))
    }

    def functions():
        return (
            functions,
            _digest,
            _build_owner,
            _source_identity,
            R.parse_mlir_text,
            _all_pins,
            _products,
            _trace,
            _trace_contract,
            pin,
            actual_observation,
            recheck_artifacts,
            recheck_selection,
            R.materialize,
            R._types,
            R.PublicRecomputationCase.revalidate,
            R.PublicRecomputationCase.source_for,
            CR.verify_completed_recipe,
            *W.functions(),
        )

    reader, code = functions, functions.__code__
    work_reader, work_code = W.functions, W.functions.__code__
    members = tuple((function, function.__code__) for function in functions())

    def check():
        assert functions is reader and reader.__code__ is code
        assert W.functions is work_reader and work_reader.__code__ is work_code
        assert all(
            f is original and original.__code__ is original_code
            for f, (original, original_code) in zip(reader(), members, strict=True)
        )
        case.revalidate()
        recheck_artifacts(original_pins)
        recheck_selection(document, selected_pin)
        assert controls == {key: os.environ.get(key) for key in controls}
        assert os.environ["MERLIN_TARGET_CONTRACT"] == str(quiet_path)

    live = {}
    # Freeze original host policy, callback/code/source/defaults BEFORE issuers.
    owners = [_build_owner(case=case, document=document, tools=tools, check=check, live=live) for _ in range(3)]
    check()
    prepared = C.prepare(
        D.read(Path(document["instruction_declaration"]["path"]), target="gemmini"),
        target="gemmini",
        output=tmp_path / "live-admission",
    )
    check()
    for build, _ in owners:
        build.revalidate()
    with target_registry.observed_contract("gemmini", quiet, source_path=quiet_path):
        facts = selected_execution_firrtl(document["facts"]["path"], target="gemmini")
    live.update(prepared=prepared, facts={"public_effective": facts, "private_input": facts})
    prepared.require_facts(live["facts"])
    package = tmp_path / "public-compiler"
    shutil.copytree(document["public_compiler"]["path"], package)
    assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
    state, calls, bindings = case.reference.initial_state(), [], []
    for epoch, (build, owner_state) in enumerate(owners):
        check()
        prepared.require_facts(live["facts"])
        bound, arrays, expected = R.materialize(
            tmp_path / f"inputs-{epoch}",
            case=case,
            arm="baseline",
            epoch=epoch,
            state=state,
        )
        bindings.append(bound)
        input_pins = [pin(bound.root / name) for name in bound.verify(bound.root)]
        assert len(input_pins) == 8
        with target_registry.observed_contract("gemmini", quiet, source_path=quiet_path):
            placement = plan_before_build(
                "gemmini",
                case.source,
                datapath="i8",
                device_package=str(package),
                capture=bound.root,
                granularity="contraction",
                linked_elf_admission=prepared.service,
                source_kind="typed_static_integer_v1",
            )
            device = placement.get("device_routing")
            assert device is not None and device.linked_elf_admission is prepared.service
            result = compile_saved_model(
                capture=bound.root,
                package=document["host_package"]["path"],
                board_catalog=document["board_catalog"]["path"],
                board=document["board"],
                dts=document["dts"]["path"],
                output=tmp_path / "out" / f"run-{epoch}",
                target="gemmini",
                run="spike",
                arena_mb=16,
                timeout_s=300,
                device=device,
                materialized_inputs=bound,
                readback="full",
                tolerance=None,
                respect_captured_quantization_scope=True,
                readonly_input_indices=tuple(range(6)),
                source_recomputation=build,
            )
        check()
        assert result["status"] == "verified_complete_output"
        elf = Path(result["output"]["elf"])
        quiet_call_path, quiet_call = actual_observation(elf.parent, "selected_rocc_process")
        observation = result["output"]["readonly_input_observation"]
        manifest = revalidate_observation(observation["artifacts"])
        assert tuple(row["original_index"] for row in manifest["arguments"]) == tuple(range(6))
        assert [row["sha256"] for row in observation["arguments"]] == [
            hashlib.sha256(a.tobytes()).hexdigest() for a in arrays
        ]
        assert [row["bytes"] for row in observation["arguments"]] == [a.nbytes for a in arrays]
        outputs, _ = read_full_outputs(
            Path(quiet_call["stdout"]["path"]).read_bytes(),
            bound.root,
            result["output"]["build_hash"],
            readonly_input_observation=observation["artifacts"],
        )
        assert len(outputs) == 9 and all(
            value.dtype == other.dtype and value.shape == other.shape and value.tobytes() == other.tobytes()
            for value, other in zip(outputs, expected, strict=True)
        )
        actual = tmp_path / f"actual-{epoch}.npz"
        np.savez(actual, **dict(zip((row[0] for row in case.reference.OUTPUTS), outputs, strict=True)))
        products = _products(result, elf)
        original = (elf.parent / "model.prepared.mlir").read_bytes()
        lower = elf.parent / "lower"
        assert (lower / "source-recomputation-input.mlir").read_bytes() == original
        assert (lower / "source-recomputation-selected.mlir").read_bytes() == original
        assert owner_state["source"].encode() == original
        record = build.record()
        assert record == result["output"]["source_recomputation"]
        assert [row["stage"] for row in owner_state["final"]] == ["linked", "executed"]
        work = W.inspect_work(
            products["host_artifacts"],
            elf.parent / "model.prepared.mlir",
            bound.root,
            manifest,
            tuple(row["symbol"] for row in products["sidecar"]["routed"]),
            candidate=False,
            diagnostics=tmp_path / f"host-work-{epoch}",
        )
        for path, digest in work["import_observation"]["tools"].items():
            assert {"path": path, "sha256": digest} in observer_pins
        products["artifacts"].extend(
            {"path": path, "sha256": digest}
            for role in ("inputs", "tools", "products")
            for path, digest in work["import_observation"][role].items()
        )
        work_path = tmp_path / f"host-work-{epoch}.json"
        work_path.write_text(json.dumps(work, indent=2) + "\n")
        traced = _trace(elf, pin(elf), prepared, trace_path, trace, tmp_path / f"decode-{epoch}", monkeypatch)
        monkeypatch.setenv("MERLIN_TARGET_CONTRACT", str(quiet_path))
        build.revalidate()
        calls.append(
            dict(
                epoch=epoch,
                elf=pin(elf),
                actual_outputs=pin(actual),
                trace=traced,
                products=products,
                build=record,
                owner=json.loads(json.dumps(owner_state)),
                readonly_input_observation=observation,
                host_work=pin(work_path),
                inputs=input_pins,
                quiet=pin(quiet_call_path),
            )
        )
        state = outputs[0].copy()
    final_records = []
    for (build, _), bound, call in zip(owners, bindings, calls, strict=True):
        bound.verify(bound.root)
        prepared.require_facts(live["facts"])
        revalidate_observation(call["readonly_input_observation"]["artifacts"])
        final_records.append(build.record())
        assert final_records[-1] == call["build"]
        CR.verify_completed_recipe(
            Path(call["products"]["recipe"]["path"]),
            executable=Path(call["elf"]["path"]),
            expected_recipe_sha256=call["products"]["recipe"]["sha256"],
        )
    check()
    assert strict_tree_sha256(package) == {"sha256": PUBLIC_COMPILER_SHA, "n_files": 16}
    retained = [*original_pins, *[row for call in calls for row in _all_pins(call)]]
    recheck_artifacts(
        retained,
        invocations=[
            row
            for call in calls
            for row in (call["quiet"], call["trace"]["invocation"], *[r["policy"] for r in call["owner"]["final"]])
        ],
        decoder_receipts=[call["trace"]["decoder_receipt"] for call in calls],
    )
    (tmp_path / "ordinary-missing-cost.json").write_text(
        json.dumps(
            dict(
                case="ordinary-post-outline-recomputation-missing-cost",
                calls=calls,
                retained_files=retained,
                selection=selected_pin,
                cost="UNKNOWN",
                automatic_candidate_selected=False,
                scope="three original-source ordinary builds with actual state carry and final prepared binding; "
                "complete outputs, input bytes, custom execution and submitted-LLVM work checked separately",
                unknown=["target_profitability", "hardware_timing", "formal_phase2_admission", "resource_certificate"],
            ),
            indent=2,
        )
        + "\n"
    )


class _NoAdmissionProbe:
    """Pure preparation only: deliberately supplies no ELF admission service."""

    def require_facts(self, facts):
        assert facts == {"scope": "pure source/owner setup; unqualified"}


def _probe_owner(tmp_path, source):
    case = R.original_case()
    own = pin(Path(__file__).resolve())
    # These are explicit synthetic context labels, never a selected target contract.
    document = {name: own for name in ("facts", "board_catalog", "dts", "quiet_contract")}

    def check():
        case.revalidate()

    build, state = _build_owner(
        case=case,
        document=document,
        tools={"clang": own, "gcc": own},
        check=check,
        live={"prepared": _NoAdmissionProbe(), "facts": {"scope": "pure source/owner setup; unqualified"}},
    )
    capture = tmp_path / "capture"
    capture.mkdir()
    (capture / "model.mlir").write_text(case.source)
    build.capture(capture)
    work = tmp_path / "build"
    work.mkdir()
    recipe = CR.CompilationRecipe(work, producer=Path(__file__).resolve())
    build.bind_sources(recipe)
    prepared = work / "model.prepared.mlir"
    prepared.write_text(source)
    build.prepare(prepared, work / "lower", {"targets": (), "index_bits": 64}, {"scope": "pure preparation"})
    return build, state


def test_original_build_owner_constructs_with_explicit_fixed_source_premises(tmp_path):
    case = R.original_case()
    own = pin(Path(__file__).resolve())

    def check():
        case.revalidate()

    document = {name: own for name in ("facts", "board_catalog", "dts", "quiet_contract")}
    build, state = _build_owner(case=case, document=document, tools={"clang": own, "gcc": own}, check=check, live={})
    build.revalidate()
    assert state["source"] is None and state["final"] == []
    assert build.policy.comparison_for is None


def test_original_build_owner_binds_current_outlined_syntax_without_file_hash_retagging(tmp_path):
    from xdsl.dialects import func
    from xdsl.dialects.linalg.ops import GenericOp

    from merlin.transforms.certificates import ir_text

    module = R.parse_mlir_text(R.original_case().source)
    reductions = [
        op
        for op in module.walk()
        if isinstance(op, GenericOp) and any(it.data.value == "reduction" for it in op.iterator_types)
    ]
    assert len(reductions) == 2
    # Construct typed opaque-call syntax only. No execution semantics/admission
    # are assigned to these external symbols, and no compiler/backend is invoked.
    for index, operation in enumerate(reductions):
        symbol = f"unqualified_syntax_probe_{index}"
        types = tuple(value.type for value in operation.results)
        module.body.block.add_op(func.FuncOp.external(symbol, [v.type for v in operation.operands], types))
        call = func.CallOp(symbol, tuple(operation.operands), types)
        operation.parent.insert_op_before(call, operation)
        for old, new in zip(operation.results, call.results, strict=True):
            old.replace_all_uses_with(new)
        operation.parent.erase_op(operation)
    module.verify()
    source = "// Deliberately noncanonical file spelling for this pure setup probe.\n" + ir_text(module)
    build, state = _probe_owner(tmp_path, source)
    selection = build.options()["source_recomputation"]
    assert selection.comparison is None and state["source"] == source
    assert selection.plan.stamp.source_sha256 == ir_digest(module)
    assert selection.plan.stamp.source_sha256 != ir_digest(source)
