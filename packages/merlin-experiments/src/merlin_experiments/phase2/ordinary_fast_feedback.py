"""Fixed normal Phase 1 compile/Spike observations before expensive timing.

Original required tiers and final grading remain unchanged. A materialized L2
execution view supplies functional screening observations, never a full grade.
Static sites and linked bytes are empirical features, not physical work or time.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.common import invocation_record as I
from merlin.common.digest import is_sha256
from merlin.perf import compiled_static_features as S
from merlin.runtime.backends.base import get_backend
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
from merlin.targetgen.rtl import facts as RTL_FACTS
from merlin_experiments.phase1.feedback.candidate_census import observe_passed_census
from merlin_experiments.phase1.feedback.candidate_census import unknown as unknown_census
from merlin_experiments.phase1.feedback.candidate_observations import (
    observe_generated_artifacts,
    reopen_native_products,
)

from . import campaign as PC
from .contracts import StageGateError, exact_tree_record, mapping_file, sha256_file, write_json
from .retained_revisions import executing_revisions

KIND = "corpus-compile-spike-feedback"
SCHEMA = "merlin.ordinary_fast_feedback.v1"
FEATURES = ("/compiled/emitted/action_sites", "/compiled/linked/executable_bytes")
LIMITS = S.StaticFeatureLimits(8 << 20, 64 << 20, 100000)


def _unique(root, name):
    paths = {}
    for path in root.rglob(name):
        if path.is_file() and not path.is_symlink():
            paths.setdefault(sha256_file(path), path)
    if len(paths) != 1:
        raise ValueError("ordinary compile has no unique " + name)
    return next(iter(paths.values()))


def compiled_observation(
    root, *, target, compiler=None, numeric_context=None, candidate_sha256=None, original_grade=None
):
    """Join fresh ELF/link invocation and emitted LLVM; unsupported bodies stay UNKNOWN."""
    try:
        entry = get_backend(target).harness_build_recipe().require_kernel_stack_frame().entry_symbol
    except (ValueError, RuntimeError, AttributeError, KeyError):
        entry = None
    diagnostics = observe_generated_artifacts(
        root,
        compiler=compiler,
        numeric_context=numeric_context,
        entry_symbol=entry,
        candidate_sha256=candidate_sha256,
        original_grade=original_grade,
    )
    try:
        native = (original_grade or {}).get("candidate_native_execution")
        if native is not None:
            if (
                not isinstance(native, dict)
                or not {"source", "candidate", "build_artifacts", "elf"} <= set(native)
                or any(not isinstance(native[key], dict) for key in ("source", "candidate", "build_artifacts"))
            ):
                raise ValueError("original native candidate receipt is incomplete")
            products, links = reopen_native_products(
                root,
                emission=original_grade.get("candidate_emission"),
                elf_pin=native.get("elf"),
                source=native.get("source"),
                build_artifacts=native.get("build_artifacts"),
                candidate=native.get("candidate"),
            )
            llvm, elf = products["emitted"], products["elf"]
            join = {
                "kind": "original_native_model_products",
                "source_sha256": sha256_file(products["source"]),
                "command_buffer_sha256": sha256_file(products["command_buffer"]),
                "producer_sha256s": [sha256_file(path) for path in links],
            }
        else:
            llvm, elf = _unique(root, "lowered.llvm.mlir"), _unique(root, PACKAGE_ELF_NAME)
            elf_pin = {"path": str(elf), "sha256": sha256_file(elf)}
            links = []
            for path in root.rglob("invocation.json"):
                row = mapping_file(path)
                if row.get("stage") == "elf" and elf_pin in row.get("outputs", []) and successful_record(row):
                    links.append(I.verify(path))
            if not links:
                raise ValueError("ordinary feature ELF lacks its actual native link invocation")
            join = {"kind": "original_operator_products"}
        if llvm.stat().st_size > LIMITS.max_source_bytes or elf.stat().st_size > LIMITS.max_elf_bytes:
            raise ValueError("ordinary compiled feature byte limit exceeded")
        elf_pin = {"path": str(elf), "sha256": sha256_file(elf)}
        report, features = None, dict.fromkeys(FEATURES)
        try:
            if entry is None:
                raise ValueError("ordinary feature entry selection is unavailable")
            data = elf.read_bytes()
            retained = S.linked_entry_extent(data, entry)
            # This observer supports ELF64 only; it supplies no source pointer premise.
            report = S.derive_compiled_static_features(
                llvm_mlir=llvm.read_text(),
                elf_bytes=data,
                entry_symbol=entry,
                pointer_bits=64,
                retained_entry=retained,
                limits=LIMITS,
            )
            features = {
                FEATURES[0]: float(sum(report["emitted"]["action_sites"].values())),
                FEATURES[1]: float(report["linked"]["executable_bytes"]),
            }
        except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as error:
            write_json(
                root / "static_feature_failure_private.json",
                {"category": type(error).__name__, "detail": str(error)[:500]},
            )
        return {
            "status": "observed",
            "elf_sha256": elf_pin["sha256"],
            "llvm_mlir_sha256": sha256_file(llvm),
            "artifact_join": join,
            "report": report,
            "static_status": "observed" if report is not None else "UNKNOWN",
            "candidate_diagnostics": diagnostics,
            "features": features,
        }
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        write_json(
            root / "compiled_failure_private.json", {"category": type(error).__name__, "detail": str(error)[:500]}
        )
        return {
            "status": "UNKNOWN",
            "reason": "ordinary compiled observation unavailable: " + type(error).__name__,
            "features": dict.fromkeys(FEATURES),
            "candidate_diagnostics": diagnostics,
        }


def successful_record(row):
    """Select completed candidates before reopening their exact original receipts.

    Ordinary build fallback keeps failed compiler/link/probe records. Those
    records cannot supply a product witness, and are not the final successful
    producer. This selection grants nothing; I.verify still checks every pin.
    """
    return row.get("status") == "completed" and (
        row.get("returncode") == 0 if row.get("kind") == "subprocess" else row.get("outcome") == "returned"
    )


def _member_arm(*, policy, revision, member, workspace, deadline, workers):
    remaining = int(deadline - time.monotonic())
    if remaining < 1:
        raise StageGateError("ordinary compile/Spike exhausted its original aggregate deadline")
    workspace.mkdir(parents=True, mode=0o700)
    before = revision.source_sha256, member.source_sha256
    capsule = dict(CR.load_capsule(member.source_dir, contract=str(policy._contract_root)))
    required = tuple(capsule.get("required_oracle_tiers") or ())
    # Keep the full original obligations. The ordinary runner records missing
    # deeper tiers as incomplete; consumers read actual L0/L1/L2 observations.
    capsule["oracle_tier_ceiling"] = "L2"
    from .public_program_measurement import selection as public_selection

    model_selection = public_selection(capsule, policy.feedback_evaluator.rtl_identity, simulator="spike")
    from merlin_experiments.phase1.feedback.compiler_profile_runtime import original_compiler_imports

    with original_compiler_imports(getattr(policy.feedback_evaluator, "_compiler_profile_service", None)):
        sandbox = PC.package_sandbox_policy(policy.target_experiment, workspace, revision.source)
        started = time.monotonic()
        with PC.boxed_entrypoints(sandbox):
            grade = CR.run_capsule(
                capsule,
                revision.source,
                runs_root=workspace / "capsule_runs",
                run_id="functional",
                contract=policy._contract_root,
                target=policy.target_experiment.target,
                oracle_adapters={"L2": CR.simulator_adapter("spike", policy.target_experiment.target)},
                adapter_managed_tiers=("L2",),
                timeout=remaining,
                workers=workers,
                **({"model_execution_selection": model_selection} if model_selection is not None else {}),
            )
    tiers = grade.get("tiers") or {}
    functional = all((tiers.get(tier) or {}).get("status") == "pass" for tier in ("L0", "L1", "L2"))
    if before != (str(hash_tree(revision.source)["sha256"]), exact_tree_record(member.source_dir)["sha256"]):
        raise StageGateError("ordinary cheap execution changed frozen compiler or input bytes")
    compiled = compiled_observation(
        workspace,
        target=policy.target_experiment.target,
        compiler=revision.source,
        numeric_context=capsule,
        candidate_sha256=revision.source_sha256,
        original_grade=grade,
    )
    facts_path = policy.feedback_evaluator.rtl_facts_path
    if facts_path.stat().st_size > 8 << 20:
        compiled["dynamic_census"] = unknown_census("original census facts exceed the descriptive input budget")
    # Reuse the already selected original facts, never an ambient target cache.
    else:
        with RTL_FACTS.observed_facts(policy.target_experiment.target, mapping_file(facts_path), facts_path):
            compiled["dynamic_census"] = observe_passed_census(
                workspace,
                target=policy.target_experiment.target,
                compiler=revision.source,
                remaining_s=max(0, deadline - time.monotonic()),
                numerical_pass=functional,
                policy=sandbox,
            )
    evidence = workspace / "ordinary_observation.json"
    result = {
        "functional_status": "PASS" if functional else "UNKNOWN_OR_FAILURE",
        "required_tiers": list(required),
        "pending_tiers": [tier for tier in required if tier not in ("L0", "L1", "L2")],
        "compiled": compiled,
        "elapsed_seconds": time.monotonic() - started,
        "physical_memory_traffic": None,
        "timing_cycles": None,
    }
    write_json(evidence, {**result, "original_grade": grade})
    return {**result, "evidence_sha256": sha256_file(evidence)}


def evaluate_retained(policy, revision, baseline_revision, *, work_root, call_index, timeout_s, workers):
    """Observe the exact complete public roster through original fixed Phase 1 owners."""
    policy.validate_search_inputs()
    revision.verify(authority=policy.edit_authority)
    baseline_revision.verify(authority=policy.edit_authority)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, mode=0o700)
    deadline = time.monotonic() + timeout_s
    members = tuple(sorted(policy.search_corpus.capsules, key=lambda row: (row.family, row.capsule)))
    tasks = [
        (index, arm, issued, member)
        for index, member in enumerate(members)
        for arm, issued in (("baseline", baseline_revision), ("candidate", revision))
    ]
    with executing_revisions(revision, baseline_revision), ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                copy_context().run,
                _member_arm,
                policy=policy,
                revision=issued,
                member=member,
                workspace=work_root / f"m{index:03d}_{arm}",
                deadline=deadline,
                workers=workers,
            )
            for index, arm, issued, member in tasks
        ]
        results = [future.result() for future in futures]
    cells = [
        {
            "family": member.family,
            "capsule": member.capsule,
            "source_sha256": member.source_sha256,
            "baseline": results[2 * index],
            "candidate": results[2 * index + 1],
        }
        for index, member in enumerate(members)
    ]
    document = {
        "schema": SCHEMA,
        "kind": KIND,
        "candidate_sha256": revision.source_sha256,
        "tuning_corpus_sha256": policy.search_corpus.capsules_sha256,
        "manifest_sha256": policy.search_corpus.manifest_sha256,
        "certificate_sha256": policy.feedback_evaluator.certificate.sha256,
        "configuration_sha256": policy.search_closure_sha256,
        "cells": cells,
        "promotion": "SCREENING_ONLY",
        "screening": {"status": "UNKNOWN", "reason": "no live ordinary calibration and held observations", "order": []},
    }
    screen = getattr(policy, "_ordinary_screen", None)
    if screen is not None:
        document["screening"] = screen.screen(policy, document)
    policy.validate_search_inputs()
    if time.monotonic() > deadline:
        raise StageGateError("ordinary cheap observation exceeded its aggregate deadline")
    write_json(work_root / "ordinary_fast_feedback.json", document)
    return validate_feedback(document, policy=policy, revision=revision)


def validate_feedback(document, *, policy, revision):
    """Validate transport against the live original selection, never grant model authority."""
    fields = {
        "schema",
        "kind",
        "candidate_sha256",
        "tuning_corpus_sha256",
        "manifest_sha256",
        "certificate_sha256",
        "configuration_sha256",
        "cells",
        "screening",
        "promotion",
    }
    if (
        type(document) is not dict
        or set(document) != fields
        or (
            document["schema"],
            document["kind"],
            document["candidate_sha256"],
            document["tuning_corpus_sha256"],
            document["manifest_sha256"],
            document["certificate_sha256"],
            document["configuration_sha256"],
            document["promotion"],
        )
        != (
            SCHEMA,
            KIND,
            revision.source_sha256,
            policy.search_corpus.capsules_sha256,
            policy.search_corpus.manifest_sha256,
            policy.feedback_evaluator.certificate.sha256,
            policy.search_closure_sha256,
            "SCREENING_ONLY",
        )
    ):
        raise StageGateError("ordinary cheap result differs from its original source/corpus/runtime selection")
    expected = {(row.family, row.capsule): row.source_sha256 for row in policy.search_corpus.capsules}
    rows = document["cells"]
    if type(rows) is not list or len(rows) != len(expected):
        raise StageGateError("ordinary cheap result omitted its complete frozen roster")
    seen = set()
    for row in rows:
        identity = row["family"], row["capsule"]
        if identity in seen or expected.get(identity) != row["source_sha256"]:
            raise StageGateError("ordinary cheap result changed public member identity")
        seen.add(identity)
        for arm in ("baseline", "candidate"):
            value = row[arm]
            if (
                type(value) is not dict
                or set(value)
                != {
                    "functional_status",
                    "required_tiers",
                    "pending_tiers",
                    "compiled",
                    "elapsed_seconds",
                    "physical_memory_traffic",
                    "timing_cycles",
                    "evidence_sha256",
                }
                or not is_sha256(value["evidence_sha256"])
            ):
                raise StageGateError("ordinary cheap observation must expose only its private evidence digest")
            if value["physical_memory_traffic"] is not None or value["timing_cycles"] is not None:
                raise StageGateError("ordinary cheap features cannot claim physical traffic or cycle timing")
            if not math.isfinite(value["elapsed_seconds"]) or value["elapsed_seconds"] < 0:
                raise StageGateError("ordinary cheap observation has invalid cost")
    return document
