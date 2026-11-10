"""Original pre-authoring numerical/runtime evidence gate for frozen suites."""

from __future__ import annotations

import ast
from pathlib import Path

from merlin.benchharness import hash_tree, runs_root
from merlin.common.paths import contract_dir
from merlin.development.backend.optimization import campaign as PC
from merlin.development.backend.optimization import candidate_record as RECORD
from merlin.development.backend.optimization import checkpoint_admission as AD
from merlin.development.backend.optimization import contracts as P2_CONTRACTS
from merlin.development.backend.optimization import corpus as P2_CORPUS
from merlin.development.backend.optimization import development_feedback as DF
from merlin.development.backend.optimization import functional_inputs as FI
from merlin.development.backend.optimization.source_layout import IMPLEMENTATION_ROOT
from merlin.experiments import source_snapshot as SNAP
from merlin.targetgen.target_experiment import load_target_experiment


class SuiteError(RuntimeError):
    pass


def require_managed_wrapper(snapshot: Path) -> None:
    """Check the sealed parser, including the current explicit envelope adapter.

    This is CLI compatibility inspection only. Source sealing and service
    admission remain the caller's independent obligations; no module is executed.
    """
    wrapper = snapshot / IMPLEMENTATION_ROOT / "chia_envelope.py"
    try:
        tree = ast.parse(wrapper.read_text(), filename=str(wrapper))
        candidates = [tree]
        envelope_names = {
            item.asname or item.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module == "merlin.development.backend.optimization"
            for item in node.names
            if item.name == "chia_envelope"
        }
        forwards = any(
            isinstance(node, ast.Return)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "main"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id in envelope_names
            for function in tree.body
            if isinstance(function, ast.FunctionDef) and function.name == "main"
            for node in ast.walk(function)
        )
        if forwards:
            envelope = snapshot / "src/merlin/development/backend/optimization/chia_envelope.py"
            candidates.append(ast.parse(envelope.read_text(), filename=str(envelope)))
        for candidate in candidates:
            for function in candidate.body:
                if not isinstance(function, ast.FunctionDef) or function.name != "main":
                    continue
                for node in ast.walk(function):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "parser"
                        and node.func.attr == "add_argument"
                        and any(
                            isinstance(arg, ast.Constant) and arg.value == "--managed-native-endpoint"
                            for arg in node.args
                        )
                    ):
                        return
    except (OSError, SyntaxError) as exc:
        raise SuiteError("cannot inspect frozen managed wrapper; prepare and qualify a new suite") from exc
    raise SuiteError(
        "frozen wrapper lacks --managed-native-endpoint support; "
        "prepare and qualify a new suite without modifying the archive"
    )


def runtime_preflight(config: AD.Config, root: Path) -> Path:
    """Exercise every pre-authoring program arm the frozen claim actually owns.

    This is host-private execution evidence, not tuning feedback or a performance claim. Each call
    uses fresh immutable inputs and keeps failed attempts; no passing receipt bypasses a later run.
    The adapter may reuse only its own certificate-bound execution cache. A candidate-only claim has
    no candidate before authoring; its certificate/adapter/capability gates are resolved here and its
    execution is explicitly deferred to the first candidate feedback call. Running the functional
    ancestor and calling its expected capability failure a suite failure would test a program the
    claim does not measure.
    """
    root.mkdir(parents=True, exist_ok=False)
    record = {
        "schema": "merlin.performance-runtime-preflight.v1",
        "status": "NO_GO",
        "functional_submission_sha256": config.functional_submission_sha256,
        "certificate_sha256": config.gsim_certificate_sha256,
        "cells": [],
    }
    try:
        target = load_target_experiment(config.descriptor)
        functional = FI.inspect_stage_functional_run(
            runs_root(target.target, "capsule-bench"),
            config.functional_run_id,
            config.functional_submission_sha256,
            waive=frozenset(config.waive_functional_gate or ()),
        )
        baseline = PC.materialize_perf_workspace(functional, root / "baseline")
        discovered = P2_CORPUS.discover_performance_corpus(
            target, families=config.perf_families, capsules=config.perf_capsules
        )
        corpus = P2_CORPUS.freeze_performance_corpus(discovered, root / "corpus")
        if not corpus.capsules:
            raise SuiteError("runtime preflight has no baseline members")
        record["corpus_sha256"] = corpus.capsules_sha256
        formal = RECORD.prepare_formal_claim(corpus.capsules)
        declaration = formal.get("declaration") if isinstance(formal, dict) else None
        if not isinstance(declaration, dict):
            raise SuiteError("runtime preflight has no frozen formal declaration")
        program_arm = declaration.get("program_arm", "baseline")
        if program_arm not in ("baseline", "candidate"):
            raise SuiteError(f"runtime preflight does not understand program arm {program_arm!r}")
        stated = {
            row.get("program_arm")
            for row in formal.get("expected_identities", [])
            if isinstance(row, dict) and row.get("program_arm") is not None
        }
        if stated and stated != {program_arm}:
            raise SuiteError("formal measurement identities disagree with their declared program arm")
        if declaration.get("program_arm") is not None and not stated:
            raise SuiteError("formal declaration names a program arm but its identities omit it")
        record["program_arm"] = program_arm
        record["formal_claim_sha256"] = P2_CONTRACTS.document_sha256(formal)
        evaluator = DF.prepare_development_feedback(
            contract_root=contract_dir(),
            certificate_path=config.gsim_certificate,
            certificate_sha256=config.gsim_certificate_sha256,
            rtl_facts_path=config.rtl_facts,
            corpus=corpus,
            baseline=baseline,
            baseline_sha256=functional.digest,
            target_experiment=target,
            work_root=root / "measurements",
            functional_run_dir=functional.run_dir,
        )
        for index, member in enumerate(corpus.capsules):
            decision = evaluator.decisions[(member.family, member.capsule)]
            if program_arm == "candidate":
                record["cells"].append(
                    {
                        "family": member.family,
                        "capsule": member.capsule,
                        "source_sha256": member.source_sha256,
                        "arm": "candidate",
                        "execution_status": "gated_until_candidate_exists",
                        "correct": None,
                        "gsim_cycles": None,
                        "qualification": decision.to_dict(),
                    }
                )
                continue
            raw = evaluator._execute(
                arm="baseline",
                package=baseline,
                package_sha256=functional.digest,
                member=member,
                decision=decision,
                workspace=root / f"measurement_{index:03d}",
                timeout_s=config.measurement_timeout,
            )
            evidence = SNAP.seal(root, f"execution.{index:03d}", raw)
            row = evaluator._redact_execution(
                raw,
                decision,
                arm="baseline",
                family=member.family,
                capsule=member.capsule,
                required_tiers=tuple(member.descriptor.get("required_oracle_tiers") or ()),
            )
            record["cells"].append(
                {
                    "family": member.family,
                    "capsule": member.capsule,
                    "source_sha256": member.source_sha256,
                    "evidence": str(evidence),
                    **row,
                }
            )
            if row["correct"] is not True:
                raise SuiteError(f"runtime preflight baseline is incorrect: {member.capsule}")
        if hash_tree(baseline)["sha256"] != functional.digest:
            raise SuiteError("runtime preflight mutated its frozen baseline")
        record["live_execution"] = (
            "completed"
            if program_arm == "baseline"
            else "required_on_first_candidate_feedback_before_any_measurement_can_pass"
        )
        record["status"] = "GO"
    except Exception as exc:
        record["failure"] = {"type": type(exc).__name__, "reason": str(exc)}
        SNAP.seal(root, "runtime_preflight", record)
        raise
    return SNAP.seal(root, "runtime_preflight", record)
