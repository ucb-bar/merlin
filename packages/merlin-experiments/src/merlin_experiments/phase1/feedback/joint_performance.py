"""Opt-in Phase 1 roofline and boxed native timing, without a second agent loop.

The operator selects immutable facts, capsule geometry and runtime bytes. The
author can change its compiler, never those inputs. Missing timing inputs retain
useful structural feedback. No structural ratio supplies attainable utilization.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from merlin.common.tree_hash import hash_tree
from merlin.perf import capsule_roofline as R
from merlin.runtime import _native_files as F

from ...phase2.contracts import document_sha256

ENV = "MERLIN_JOINT_PERFORMANCE_SELECTION"
SCHEMA = "merlin.phase1.joint_performance_selection.v1"


def capsule_identity(capsule):
    """Native discovery stamps a movable directory; bind the declaration bytes."""
    return document_sha256({k: v for k, v in (capsule or {}).items() if k != "__dir__"})


def _read(pin):
    path = Path(pin["path"])
    before = F.pin(path, max_bytes=16 << 20)
    if before["sha256"] != pin["sha256"]:
        raise ValueError("joint performance input changed")
    value = json.loads(path.read_bytes())
    if F.pin(path, max_bytes=16 << 20) != before:
        raise ValueError("joint performance input changed while reading")
    return value


def selected():
    """Read only the explicitly pinned operator selection; no fallback discovery."""
    raw = os.environ.get(ENV)
    if not raw:
        return None
    pin = json.loads(raw)
    value = _read(pin)
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported joint performance selection")
    budget = value.get("per_request_seconds", 30)
    if type(budget) not in (int, float) or not 0 < budget <= 900:
        raise ValueError("joint performance observation budget must be in (0, 900]")
    return pin, value


def deadline(timeout):
    selection = selected()
    return time.monotonic() + min(timeout, selection[1].get("per_request_seconds", 30)) if selection else 0


def _runtime(pin):
    from ...phase2.archive_spike_runtime import COUNT_OBSERVATION_900S_PROTOCOL, freeze_runtime

    spec = _read(pin)
    # Every pricing update gets a new selection and runtime identity. No inherited
    # historical coefficient digest is implicitly approved by this adapter.
    runtime = freeze_runtime(
        review_id=spec["review_id"], functional_argv=tuple(spec["functional_argv"]),
        observer=Path(spec["observer"]), disassembler=Path(spec["disassembler"]),
        model_path=Path(spec["model"]), source_manifest=Path(spec["source_manifest"]),
        libraries=tuple(map(Path, spec["libraries"])), dependencies=tuple(map(Path, spec["dependencies"])),
        entry_symbol=spec["entry_symbol"], completion_symbol=spec["completion_symbol"],
        timing_selection=spec["timing_selection"],
        observation_resource_protocol=COUNT_OBSERVATION_900S_PROTOCOL,
    )
    expected = spec["model_sha256"]
    if F.digest(runtime.model_path) != expected:
        raise ValueError("joint runtime pricing differs from pinned model")
    return runtime


def structural_feedback(selection, capsule, *, target):
    """Reopen source-specific geometry and derive arithmetic capacity from facts."""
    facts = _read(selection["facts"])
    machine = R.machine_bounds(target, facts=facts)
    member = selection.get("members", {}).get(capsule.get("name"), {})
    if member and member.get("capsule_sha256") != capsule_identity(capsule):
        raise ValueError("joint roofline geometry belongs to different capsule bytes")
    return R.partial_roofline(
        member.get("contractions"), machine,
        logical_read_bytes=member.get("logical_read_bytes"),
        logical_write_bytes=member.get("logical_write_bytes"),
    )


def attach(row, root, *, result, context, compiler, capsule, remaining_s):
    """Attach a public projection after the original complete numerical screen.

    No gSIM is launched. Exact native ELF producer custody, one-ELF namespace,
    plain/annotated console equivalence and bounded shutdown are inherited from
    the Phase 2 observer. Cleanup uncertainty propagates to the owning job.
    """
    from .joint_performance_report import project

    choice = selected()
    if choice is None:
        return
    pin, selection = choice
    report = None
    observation = {}
    structural = None
    reason = "timing runtime not selected"
    try:
        structural = structural_feedback(selection, capsule or {}, target=context.target)
        numeric = result.get("numeric") or {}
        tier = (result.get("tiers") or {}).get("L2") or {}
        passed = numeric.get("status") == "pass" and tier.get("status") == "pass"
        if not passed:
            reason = "original complete numerical screen did not pass"
        elif tier.get("engine") != "spike":
            reason = "timing observation only follows original Spike execution"
        elif remaining_s < 1:
            reason = "joint observation request budget exhausted"
        elif selection.get("runtime"):
            runtime = _runtime(selection["runtime"])
            report, observation = _observe(
                root, context=context, compiler=compiler, runtime=runtime,
                remaining_s=min(remaining_s, selection.get("per_request_seconds", 30)),
            )
            reason = observation.get("reason", "native timing is provisional pending independent qualification")
        if selected() != choice:
            raise ValueError("joint selection changed during observation")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        report = None
        # Details and paths are operator-only. Do not expose reference filenames.
        reason = "joint performance unavailable: " + type(error).__name__
    row["performance"] = project(
        report, observation, structural, selection_sha256=pin["sha256"], reason=reason,
    )
    row["performance"]["capsule_sha256"] = capsule_identity(capsule)


def _observe(root, *, context, compiler, runtime, remaining_s):
    from merlin.targetgen.elf_lanes import PACKAGE_ELF_NAME
    from merlin.targetgen.target_experiment import load_target_experiment

    from ...phase2 import archive_count_observation as OBS
    from ...phase2 import campaign as PC

    root = F.canonical(Path(root))
    # Bound discovery; do not accept a convenient executable basename from an
    # unrelated build. OBS independently reopens its completed native producer.
    images = [p for p in F.members(root, max_files=4096) if p.name == PACKAGE_ELF_NAME and p.is_file()]
    if len(images) != 1:
        raise ValueError("no unique original capsule executable")
    elf = images[0]
    compiler = Path(compiler)
    candidate = hash_tree(compiler)["sha256"]
    elf_before = F.pin(elf, max_bytes=64 << 20)
    target = load_target_experiment(context.descriptor)
    if target.target != context.target:
        raise ValueError("joint observation target mismatch")
    policy = PC.package_sandbox_policy(target, root, compiler)
    # These self-contained capsule ELFs embed their complete stimulus. The input
    # identity is explicitly ELF-bound, not a fabricated external tensor hash.
    observation = OBS.observe(
        elf, runtime=runtime, policy=policy, directory=root / "joint_timing",
        input_sha256=elf_before["sha256"], remaining_s=remaining_s, numerical_pass=True,
    )
    observation["input_identity_basis"] = "self-contained numerically checked ELF, including embedded stimulus"
    report = _read(observation["report"]) if observation.get("status") == "observed" else None
    if (F.pin(elf, max_bytes=64 << 20) != elf_before
            or hash_tree(compiler)["sha256"] != candidate):
        raise ValueError("candidate changed during joint observation")
    runtime.verify()
    if report is not None:
        model = json.loads(runtime.model_json)
        if report.get("model_sha256") != F.digest(runtime.model_path):
            raise ValueError("joint report model differs from selected runtime")
        if report.get("domain") != dict(model["domain"], target_model_sha256=F.digest(runtime.model_path)):
            raise ValueError("joint report comparison domain differs from selected runtime")
    return report, observation
