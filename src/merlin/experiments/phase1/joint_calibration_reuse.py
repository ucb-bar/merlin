"""Reuse original public correctness observations as exact calibration labels.

This is an operator-owned evidence adapter, not an execution cache or admission
surface. The normal fit/split/identifiability gates still precede label access.
A missing component is a denial, never inferred from a passing cycle scalar.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

from merlin.common import invocation_record as I
from merlin.runtime import _native_files as F

from .feedback import joint_performance as PERF
from ..phase2.contracts import document_sha256

SCHEMA = "merlin.correctness_calibration_reuse.v1"


class _Unavailable(ValueError):
    pass


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pin(path, *, maximum=64 << 20):
    path = F.canonical(Path(path))
    _require(path.stat().st_nlink == 1, "retained evidence has multiple hard links")
    return F.pin(path, max_bytes=maximum)


def _member(root, pin):
    _require(isinstance(pin, dict) and isinstance(pin.get("path"), str), "missing original member pin")
    relative = PurePosixPath(pin["path"])
    _require(not relative.is_absolute() and ".." not in relative.parts and str(relative) == pin["path"],
             "retained member escaped its original owner")
    path = Path(root) / relative
    checked = _pin(path)
    _require(checked["sha256"] == pin.get("sha256"), "retained member bytes changed")
    return path


def _runtime_components(process, *, runtime, backend, elf, expected):
    """Reopen a closed original process without manufacturing an issuance tag.

    The native launcher source and all original namespace bytes remain pinned.
    Comparing image content and actual argv/environment proves component
    equivalence only; it never asserts that a legacy record had a live owner.
    Direct unclosed host processes remain unavailable.
    """
    _require(process.get("kind") == "subprocess" and process.get("returncode") == 0,
             "original physical process did not complete")
    if process.get("stage") != "native_runtime_process":
        raise _Unavailable("original process has no closed native runtime/config component receipt")
    # This adapter supports only the ordinary BoundRoCCBackend selection. Its
    # pinned native owner maps that backend to false/UTF-8, rather than taking
    # these non-argv settings from an unauthenticated legacy claim.
    _require(expected.get("unlimited_stack") is False and expected.get("text_mode") == "utf8",
             "legacy component proof does not cover selected process mode")
    argv = process.get("argv")
    prefix = [str(runtime.bwrap), "--unshare-all", "--unshare-user", "--as-pid-1", "--cap-drop", "ALL",
              "--disable-userns", "--assert-userns-disabled", "--die-with-parent", "--new-session", "--clearenv",
              "--ro-bind"]
    _require(isinstance(argv, list) and argv[:len(prefix)] == prefix,
             "original native process namespace prefix differs")
    tail = argv[len(prefix):]
    _require(len(tail) >= 21 and tail[1:3] == ["/", "--ro-bind"]
             and tail[4:6] == ["/run/merlin/input.elf", "--bind"]
             and tail[7:9] == ["/run/work", "--bind"]
             and tail[10:12] == ["/tmp", "--chdir"]
             and tail[13:20] == ["--", "/bin/bash", "--noprofile", "--norc", "-c", tail[18], "native-runtime"],
             "original native process mount/dispatch layout differs")
    old_image, old_elf = Path(tail[0]), Path(tail[3])
    _require(_pin(old_elf)["sha256"] == _pin(elf)["sha256"], "physical process consumed another ELF")
    _require(tail[12] == expected["cwd"], "physical process cwd differs")
    script = tail[18]
    descriptor = script.removeprefix("printf READY >&").partition(";")[0]
    _require(script.startswith("printf READY >&") and descriptor.isascii() and descriptor.isdigit()
             and script == (f"printf READY >&{descriptor}; read -r native_go <&{descriptor}; "
                            f'[[ $native_go == GO ]] || exit 126; exec {descriptor}>&-; exec /usr/bin/env -- "$@"'),
             "original native process bootstrap differs")
    _require(process.get("environment") == I.environment_identity({}), "original native wrapper environment differs")
    identity = runtime.verify(backend)
    image = F.inventory(F.canonical(old_image), max_files=runtime.max_files, max_bytes=runtime.max_bytes)
    _require(image["sha256"] == identity["runtime_view_sha256"], "physical runtime filesystem/config differs")
    environment = backend._engine("gsim")["environment"]
    suffix = [*(f"{key}={value}" for key, value in sorted(environment.items())), *expected["guest_argv"]]
    _require(tail[20:] == suffix and expected["environment"] == I.environment_identity(environment),
             "physical process argv or memory-service environment differs")
    dependencies = {(p["path"], p["sha256"]) for p in process.get("dependencies", [])}
    for source in runtime.observation_sources():
        source = Path(source)
        # A separately issued owner may materialize the exact same guest image
        # under another private path. Reopen the original guest member at its
        # actual old path; do not require or invent a new dependency receipt.
        original = old_image / source.relative_to(runtime.image) if source.is_relative_to(runtime.image) else source
        _require((str(original), _pin(original)["sha256"]) in dependencies
                 and _pin(original)["sha256"] == _pin(source)["sha256"],
                 "original native process owner/source components differ")
    _require(any(p.get("sha256") == _pin(elf)["sha256"] for p in process.get("inputs", [])),
             "original native process has no exact ELF input binding")
    return dict(kind="legacy_component_equivalence", runtime_view_sha256=image["sha256"],
                selection_sha256=identity["selection_sha256"], reset_seed_closure="UNKNOWN")


def _selected_runtime(backend, elf):
    from merlin.runtime.backends.chipyard_rocc_bound import BoundRoCCBackend
    from merlin.runtime.native_runtime import active_native_runtime
    runtime = active_native_runtime()
    _require(type(backend) is BoundRoCCBackend and runtime is not None,
             "reuse requires the original selected native backend/runtime owner")
    return runtime, runtime.execution_identity(backend, elf, capture_bytes=True)


def _process_record(root, execution, *, console, runtime, backend, elf, expected):
    artifacts = execution["compiler_artifacts"]
    # A retained signed observation is stronger, when its exact live original
    # owner is still available. Never fabricate this record from old components.
    native = artifacts.get("native_observation")
    if native is not None:
        _require(isinstance(native, dict), "original native observation is malformed")
        try:
            runtime.reopen_observation(native["body"], native["signature"], elf=elf, console=console)
        except ValueError:
            # A new owner cannot authenticate an old live-owner signature.
            # Independently prove components below; never reissue that tag.
            pass
        else:
            body = native["body"]
            original = {k: v for k, v in body["selection"].items() if k != "capture_bytes"}
            selected = {k: v for k, v in expected.items() if k != "capture_bytes"}
            _require(original == selected, "issued native process selection differs")
            return dict(kind="same_issued_native_owner", observation=native,
                        process=body["record"])
    matches = []
    # Bounded discovery of original process records only; no generated benchmark
    # or external labels are discovered. Every candidate is independently reopened.
    for index, path in enumerate(Path(root).glob("invocations/*/invocation.json")):
        if index >= 256:
            raise ValueError("original process roster exceeds bounded evidence budget")
        _pin(path, maximum=1 << 20)
        row = json.loads(path.read_bytes())
        if row.get("kind") != "subprocess" or row.get("stage") != "native_runtime_process":
            continue
        if not any(p.get("sha256") == _pin(elf)["sha256"] for p in row.get("inputs", [])):
            continue
        row = I.verify(path)
        raw = Path(row["stdout"]["path"]).read_bytes()
        if raw != console:
            continue
        components = _runtime_components(row, runtime=runtime, backend=backend, elf=elf, expected=expected)
        matches.append(dict(components=components, process=_pin(path, maximum=1 << 20)))
    if len(matches) != 1:
        raise _Unavailable("no unique original physical process with complete matched runtime components")
    return matches[0]



def _retained_memory_outputs(root, execution, *, console, engine):
    """Reopen actual direct-process dump and decoder products, not saved values.

    This closes byte/producer joins only. The independent runtime/config check
    remains mandatory afterward; a direct process does not become a native view.
    """
    artifacts = execution["compiler_artifacts"]
    memory, observation = artifacts.get("readback_memory"), artifacts.get("readback_observation")
    if not isinstance(memory, dict) or not isinstance(observation, dict):
        raise _Unavailable("original full-value memory producer/decoder receipts are absent")
    def member(pin):
        _require(isinstance(pin, dict), "original decoder member pin is absent")
        path = F.canonical(Path(pin["path"]))
        _require(path.is_relative_to(root) and _pin(path)["sha256"] == pin["sha256"],
                 "original decoder member escaped or changed")
        return path
    record_path, product_path = member(observation["record"]), member(observation["product"])
    decoded = I.verify(record_path)
    _require(decoded.get("kind") == "python_call" and decoded.get("stage") == "coherent_memory_decode"
             and _pin(product_path) in decoded["outputs"]
             and decoded["stdout"]["sha256"] == _pin(product_path)["sha256"],
             "full-value decoder lost its original actual return")
    returned = json.loads(product_path.read_bytes())
    _require(returned.get("memory_evidence") == memory and memory.get("status") == "complete"
             and memory.get("source") == "coherent" and isinstance(returned.get("outputs"), dict)
             and returned["outputs"], "original full-value decoder return differs")
    original_elf = member(memory["original_elf"])
    payload, regions = member(memory["payload"]), member(memory["regions"])
    _require(_pin(original_elf)["sha256"] == execution["elf"]["sha256"]
             and _pin(original_elf) in decoded["inputs"]
             and _pin(payload) in decoded["outputs"] and _pin(regions) in decoded["inputs"]
             and memory.get("artifact_sha256") == _pin(payload)["sha256"]
             and any(p.get("sha256") == memory.get("console_sha256") for p in decoded["inputs"])
             and memory.get("console_sha256") == hashlib.sha256(console).hexdigest(),
             "full-value decoder lost original ELF/payload/console membership")
    outer_path = Path(artifacts["original_execution_invocation"]["path"])
    outer = I.verify(outer_path)
    processes = []
    for index, path in enumerate(Path(root).glob("invocations/*/invocation.json")):
        if index >= 256:
            raise ValueError("original process roster exceeds bounded evidence budget")
        _pin(path, maximum=1 << 20)
        row = json.loads(path.read_bytes())
        if row.get("stage") != "selected_rocc_process" or row.get("kind") != "subprocess":
            continue
        if row.get("executable", {}).get("sha256") != engine["binary"]["sha256"]:
            continue
        row = I.verify(path)
        if (_pin(original_elf) not in row["inputs"] or _pin(regions) not in row["inputs"]
                or _pin(payload) not in row["outputs"]):
            continue
        argv = row["argv"]
        _require(argv[-3:] == ["+dump-regions=" + str(regions), "+dump-out=" + str(payload), "+dump-mode=coherent"]
                 and Path(row["stdout"]["path"]).read_bytes() == console
                 and outer["started_ns"] <= row["started_ns"] <= row["finished_ns"] <= outer["finished_ns"],
                 "original coherent output lost its actual simulator command/window")
        processes.append(_pin(path, maximum=1 << 20))
    _require(len(processes) == 1, "no unique original simulator producer pins this complete dump and regions")
    return returned["outputs"], dict(kind="original_coherent_dump_and_decoder_products",
        decoder=_pin(record_path), product=_pin(product_path), payload=_pin(payload), regions=_pin(regions),
        process=processes[0], runtime_qualification="not_issued_by_byte_join")

def inspect(case, *, record, plan, backend, model):
    """Return explicit ready/denied evidence; never execute or grant admission.

    Called by the trusted round owner only after its native identifiability
    review and at the existing train/held label-open boundary.
    """
    try:
        return _inspect(case, record=record, plan=plan, backend=backend, model=model)
    except _Unavailable as error:
        return dict(schema=SCHEMA, status="denied", reason=str(error), fresh_observation_required=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        return dict(schema=SCHEMA, status="denied", reason=str(error), fresh_observation_required=False)


def _inspect(case, *, record, plan, backend, model):
    import numpy as np
    from merlin.benchharness import hash_tree
    from . import joint_calibration as C, joint_control
    from .joint_calibration_admission import assert_allowed
    from .joint_calibration_eligibility import require
    from .feedback.candidate_execution_products import reopen_operator_tier_products

    assert_allowed(plan)  # Runtime denial also applies to retained labels.
    _require(plan.get("schema") == C.SCHEMA and plan.get("gates") == C.GATES,
             "reuse requires the unchanged frozen calibration schema/gates")
    _require(any(case == c for c in plan["cases"]), "case is not an exact member of the frozen round")
    state = joint_control.selected()
    _require(state is not None, "no selected native calibration campaign")
    profile = PERF._read(state["policy"]["native_calibration"])
    cap = C._capsule(case["capsule"])
    member = profile["members"].get(cap["name"])
    _require(member is not None and member["source_group_sha256"] == case["source_group_sha256"],
             "case has no exact reviewed profile/source group")
    _require(cap["name"] in state["policy"]["gsim"]["witness_capsules"], "case is outside the physical witness roster")
    selection = PERF._read(plan["selection"])
    selected = selection.get("members", {}).get(cap["name"])
    _require(selected is not None, "case is outside selected public source roster")
    resolved = PERF.selected_capsule(selection, cap)
    _require(PERF.capsule_identity(resolved) == selected["capsule_sha256"], "selected capsule identity differs")
    grade = PERF._read(case["result"])
    require(grade, cap, compiler_sha256=case["compiler_sha256"])
    _require(hash_tree(Path(case["compiler"]))["sha256"] == case["compiler_sha256"], "compiler snapshot changed")
    tier = grade["tiers"].get("L3", {})
    if tier.get("status") != "pass" or tier.get("engine") != "gsim":
        raise _Unavailable("no original passed correctness gSIM observation")
    _require(tier.get("cycle_accurate") is True and tier.get("derived_from_rtl") is True,
             "original correctness observation is not cycle-accurate RTL")
    root = F.canonical(Path(case["generated_root"]))
    paths, custody = reopen_operator_tier_products(root, grade, tier="L3")
    elf = Path(record["elf"])
    _require(_pin(elf)["sha256"] == custody["elf_sha256"], "calibration replay and correctness ELF differ")
    observation = record["observation"]
    report = PERF._read(observation["report"])
    selected_runtime = PERF._read(plan["runtime"])
    selected_model = PERF._read(dict(path=selected_runtime["model"], sha256=selected_runtime["model_sha256"]))
    _require(selected_model == model, "selected frozen model differs")
    _require(report.get("timing_complete") is True and not report.get("unresolved")
             and report.get("elf_sha256") == custody["elf_sha256"]
             and observation.get("elf_sha256") == custody["elf_sha256"]
             and report.get("scope_sha256") == observation["scope_binding"].get("scope_sha256")
             and observation["scope_binding"].get("input_sha256") == custody["elf_sha256"]
             and report.get("scope_binding") == observation["scope_binding"],
             "native observation lost exact ELF/scope or complete timing")
    engine = backend._engine_identity("gsim")
    _require(engine.get("binary", {}).get("sha256") == model["domain"]["label_engine_sha256"],
             "selected physical engine differs from model domain")
    simulator = tier["sim_provenance"]
    _require(simulator.get("sha256") == engine["binary"]["sha256"], "original gSIM engine differs")
    executions = [x for x in grade["compiler_source"]["executed_programs"] if x.get("engine") == "gsim"]
    _require(len(executions) == 1, "original gSIM program is absent or ambiguous")
    execution = executions[0]
    oracle = execution["compiler_artifacts"].get("oracle") or {}
    memory = oracle.get("memory_engine") or {}
    if not memory.get("rtl_facts"):
        raise _Unavailable("original physical facts/config component is absent")
    facts = memory["rtl_facts"]
    _require(facts.get("sha256") == selection["facts"]["sha256"], "original facts/config differs")
    _require(_pin(facts["path"])["sha256"] == facts["sha256"], "original physical facts bytes changed")
    # The selected engine owner independently reopens the actual RTL build.
    receipt = (memory.get("engine") or {}).get("receipt") or {}
    _require(receipt.get("binary_sha256") == engine["binary"]["sha256"]
             and receipt.get("firrtl_sha256") == facts.get("firrtl_sha256"),
             "original build/facts/engine configuration binding differs")
    if not receipt.get("receipt_path") or not receipt.get("receipt_sha256"):
        raise _Unavailable("original physical build receipt cannot be reopened")
    _require(_pin(receipt["receipt_path"])["sha256"] == receipt["receipt_sha256"], "original physical build receipt changed")
    _require(backend._engine("gsim").get("receipt", {}).get("sha256") == receipt["receipt_sha256"],
             "selected physical build/config receipt differs")
    console_path = _member(root, execution["compiler_artifacts"]["members"]["console"])
    console = console_path.read_bytes()
    actual, metrics = backend.parse_output(console)
    output_custody = dict(kind="original_full_value_console")
    if not actual:
        actual, output_custody = _retained_memory_outputs(root, execution, console=console, engine=engine)
    processes = observation["processes"]
    _require(len(processes) == 2 and observation.get("functional_identical") is True
             and observation.get("complete_counts") is True,
             "native functional/annotated numerical observation is absent or ambiguous")
    reference_paths = []
    reference_bytes = []
    reference_values = []
    for process in processes:
        checked = I.verify(Path(process["path"]))
        _require(_pin(process["path"])["sha256"] == process["sha256"], "native numerical process changed")
        _require(checked["stage"] == "archive_screen_count"
                 and any(p.get("sha256") == custody["elf_sha256"] for p in checked["inputs"]),
                 "native numerical process does not consume this exact program")
        reference_path = Path(checked["stdout"]["path"])
        _require(_pin(reference_path)["sha256"] == process["stdout_sha256"], "native numerical output changed")
        reference_paths.append(reference_path)
        reference_bytes.append(reference_path.read_bytes())
        if observation.get("output_export") is not None:
            from ..phase2.integer_signature_export import reopen_values
            reference_values.append(reopen_values(observation, process))
        else:
            reference_values.append(backend.parse_output(reference_bytes[-1])[0])
        # Probe zero serializes the same frozen model into its own immutable
        # file. Reopen that actual process dependency, never substitute a saved
        # model digest or reject harmless JSON formatting changes.
        models = [p for p in checked["dependencies"] if p.get("sha256") == report.get("model_sha256")]
        _require(models and all(PERF._read(p) == model for p in models),
                 "native numerical process lost the exact replay model dependency")
        scope_inputs = [p for p in checked["inputs"] if Path(p["path"]).name == "scope.json"]
        _require(len(scope_inputs) == 1 and PERF._read(scope_inputs[0]) == observation["scope_binding"],
                 "native numerical process lost its actual counter scope input")
    _require(reference_bytes[0] == reference_bytes[1], "annotation changed the checked numerical console")
    expected_values = reference_values[0]
    _require(expected_values and set(expected_values) == set(reference_values[1])
             and all(np.array_equal(expected_values[k], reference_values[1][k], equal_nan=True) for k in expected_values),
             "functional and annotated full output values differ")
    _require(expected_values and set(actual) == set(expected_values)
             and all(np.array_equal(actual[k], expected_values[k], equal_nan=True) for k in actual),
             "retained physical full output values differ from checked native numerics")
    cycles = metrics.get("cycles")
    _require(type(cycles) in (int, float) and math.isfinite(cycles) and cycles > 0 and cycles == tier.get("cycles"),
             "retained physical counter cycles are absent or differ")
    # Reopen this observation's own counter review, including calibration
    # replays under joint_calibration/<round>/<probe>, not a convenience path.
    counter_path = F.canonical(Path(observation["report"]["path"]).parent / "counter_review.json")
    counter = json.loads(counter_path.read_bytes())
    binding = observation["scope_binding"]
    _require(counter.get("elf_sha256") == custody["elf_sha256"]
             and document_sha256(counter) == binding.get("evidence_sha256")
             and binding.get("scope_sha256") == model["domain"].get("scope_sha256")
             and all(counter.get(k) == binding.get(k) for k in ("begin_pc", "end_pc", "occurrence")),
             "physical/native counter scope differs")
    runtime, expected_runtime = _selected_runtime(backend, elf)
    provenance = _process_record(root, execution, console=console, runtime=runtime,
                                 backend=backend, elf=elf, expected=expected_runtime)
    proof = dict(case_id=case["id"], case=case, compiler_sha256=case["compiler_sha256"],
        facts=selection["facts"], source_group_sha256=case["source_group_sha256"],
        physical_program=execution, custody=custody, runtime=provenance, physical_output_custody=output_custody,
        console=_pin(console_path), native_outputs=[_pin(p) for p in reference_paths],
        counter_review=_pin(counter_path),
        scope_binding=binding, engine=engine, elf=_pin(elf), cycles=cycles)
    return dict(schema=SCHEMA, status="ready", reason="exact original correctness observation reopened",
        proof_sha256=document_sha256(proof), proof=proof, fresh_observation_required=False)


def measurement(review, *, plan_pin):
    """Translate a ready observation into an explicitly reused label receipt."""
    _require(review.get("schema") == SCHEMA and review.get("status") == "ready", "reuse evidence is not ready")
    proof = review["proof"]
    _require(document_sha256(proof) == review["proof_sha256"], "reuse evidence changed")
    return dict(schema="merlin.joint_calibration_measurement.v1", label_origin="retained_correctness_gsim",
        elf=proof["elf"],
        engine=proof["engine"], case_id=proof["case_id"], cycles=proof["cycles"], numeric="pass",
        scope_binding=proof["scope_binding"], console=proof["console"],
        reuse=dict(plan=plan_pin, case_id=proof["case_id"], proof_sha256=review["proof_sha256"]),
        label_scope="original same-ELF correctness gSIM; frozen-round membership and split preserved")


def revalidate(receipt, record, backend):
    """Reopen retained producer custody on resume; copied cycle values are insufficient."""
    plan = PERF._read(receipt["reuse"]["plan"])
    cases = [c for c in plan["cases"] if c["id"] == receipt["reuse"]["case_id"]]
    _require(len(cases) == 1, "reused label case changed")
    runtime = PERF._read(plan["runtime"])
    model = PERF._read(dict(path=runtime["model"], sha256=runtime["model_sha256"]))
    review = inspect(cases[0], record=record, plan=plan, backend=backend, model=model)
    _require(review["status"] == "ready" and review["proof_sha256"] == receipt["reuse"]["proof_sha256"],
             "retained correctness reuse custody changed: " + review.get("reason", "unknown"))
    _require(receipt == measurement(review, plan_pin=receipt["reuse"]["plan"]),
             "reused label receipt differs from the original reopened observation")
