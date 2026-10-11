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
from merlin.perf import compiled_static_features as S
from merlin.runtime.backends.base import get_backend
from merlin.targetgen import capsule_runner as CR
from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME

from . import campaign as PC
from .contracts import StageGateError, exact_tree_record, sha256_file, write_json
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


def compiled_observation(root, *, target):
    """Join fresh ELF/link invocation and emitted LLVM; unsupported bodies stay UNKNOWN."""
    try:
        llvm, elf = _unique(root, "lowered.llvm.mlir"), _unique(root, PACKAGE_ELF_NAME)
        if llvm.stat().st_size > LIMITS.max_source_bytes or elf.stat().st_size > LIMITS.max_elf_bytes:
            raise ValueError("ordinary compiled feature byte limit exceeded")
        elf_pin = {"path": str(elf), "sha256": sha256_file(elf)}
        links = [(path, I.verify(path)) for path in root.rglob("invocation.json")]
        if not any(row["stage"] == "elf" and elf_pin in row["outputs"] for _, row in links):
            raise ValueError("ordinary feature ELF lacks its actual native link invocation")
        recipe = get_backend(target).harness_build_recipe()
        entry = recipe.require_kernel_stack_frame().entry_symbol
        data = elf.read_bytes()
        retained = S.linked_entry_extent(data, entry)
        # linked_entry_extent supports ELF64 only. This is an observed ELF class,
        # not a claim about source pointer semantics or a selected target default.
        report = S.derive_compiled_static_features(
            llvm_mlir=llvm.read_text(),
            elf_bytes=data,
            entry_symbol=entry,
            pointer_bits=64,
            retained_entry=retained,
            limits=LIMITS,
        )
        return {
            "status": "observed",
            "elf_sha256": elf_pin["sha256"],
            "llvm_mlir_sha256": sha256_file(llvm),
            "report": report,
            "features": {
                FEATURES[0]: float(sum(report["emitted"]["action_sites"].values())),
                FEATURES[1]: float(report["linked"]["executable_bytes"]),
            },
        }
    except (OSError, ValueError, RuntimeError, KeyError, AttributeError) as error:
        return {"status": "UNKNOWN", "reason": str(error)[:500], "features": dict.fromkeys(FEATURES)}


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
        )
    tiers = grade.get("tiers") or {}
    functional = all((tiers.get(tier) or {}).get("status") == "pass" for tier in ("L0", "L1", "L2"))
    if before != (str(hash_tree(revision.source)["sha256"]), exact_tree_record(member.source_dir)["sha256"]):
        raise StageGateError("ordinary cheap execution changed frozen compiler or input bytes")
    compiled = compiled_observation(workspace, target=policy.target_experiment.target)
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
    return {**result, "evidence": {"path": str(evidence), "sha256": sha256_file(evidence)}}


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
            if value["physical_memory_traffic"] is not None or value["timing_cycles"] is not None:
                raise StageGateError("ordinary cheap features cannot claim physical traffic or cycle timing")
            if not math.isfinite(value["elapsed_seconds"]) or value["elapsed_seconds"] < 0:
                raise StageGateError("ordinary cheap observation has invalid cost")
    return document
