"""Common observed runtime contract for versioned author profiles.

This stages public case identities and resource facts, without optimization advice
or access to protected source, fixtures, answers or prior compiler results.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from merlin.common.process.cpu_pool import grading_cpus, parse_roster, shared_compute_enabled
from merlin.development.backend.authoring import author_subagents
from merlin.development.backend.authoring import profiles as author_profiles
from merlin.development.backend.campaign import control as joint_control

# ---- profile_guidance --------------------------------------------------------------------------

PUBLIC_CASES = "PUBLIC_CASES.json"

HEADER = "## Selected public scope and runtime limits"


def _write_once(path, data):
    path = Path(path)
    if path.is_symlink():
        raise ValueError("profile guidance cannot replace a symlink")
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError("selected public profile roster changed")
    else:
        path.write_bytes(data)
    path.chmod(0o444)


def append_profile_guidance(workspace, run_dir, options, *, context, public_roots=None, contract=None):
    """Append the same public runtime facts to every matched profile, before sealing."""
    profile = author_profiles.resolve(getattr(options, "author_profile", ""))
    if profile is None:
        raise ValueError("profile runtime guidance requires an explicit author profile")
    suites = _metrics_roster(context, public_roots=public_roots, contract=contract)
    cases = sorted(name for names in suites.values() for name in names)
    counts = {suite: len(names) for suite, names in suites.items()}
    roster = dict(
        schema="merlin.author_public_cases.v1",
        scope="selected_public_capsules",
        counts=counts,
        total=len(cases),
        suites=suites,
        cases=cases,
    )
    encoded = (json.dumps(roster, sort_keys=True, indent=2) + "\n").encode()
    state = joint_control.selected()
    policy = {} if state is None else state["policy"]
    wall_s = policy.get("duration_s", options.max_wall_s)
    drain_s = policy.get("drain_s", 0)
    if wall_s != options.max_wall_s or not 0 <= drain_s < wall_s:
        raise ValueError("profile runtime clock differs from the selected launch")
    author_s = wall_s - drain_s
    author = parse_roster(os.environ.get("MERLIN_CODEX_CPUS", ""))
    compute = grading_cpus()
    shared = shared_compute_enabled()
    if shared and set(author) != set(compute):
        raise ValueError("profile author and shared compute CPU selections disagree")
    helper_selection = author_subagents.selection(options)
    helper_limit = None if helper_selection is None else helper_selection["max_concurrent_threads_per_session"]
    lines = [
        "\n\n" + HEADER + "\n",
        f"- `{PUBLIC_CASES}` lists the actual selected public case IDs and their suites: "
        + ", ".join(f"{counts[suite]} {suite}" for suite in suites)
        + f"; {len(cases)} public cases in total. Every listed case retains its declared numerical, "
        "placement, output and required evaluation obligations.",
        f"- Authoring budget: {author_s / 3600:g} hours ({author_s} seconds), followed by at most "
        f"{drain_s} seconds for the selected final drain; total run ceiling {wall_s} seconds. "
        "Provider waiting and retries count within the original clock.",
        f"- The author and its helpers share the assigned {len(author)}-CPU pool. "
        + (
            f"Preparation, compilation and simulation share that same {len(compute)}-CPU pool; "
            if shared
            else "Harness compute uses its separately declared CPU allocation; "
        )
        + "helpers receive no additional CPUs. "
        + (
            f"At most {helper_limit} helper agents may be active concurrently."
            if helper_limit is not None
            else "Helper concurrency follows the selected provider policy."
        ),
        (
            "- `mlir_oot_backend_contract.yaml` defines the submitted manifest, build and entrypoint contract. "
            + (
                "`neutral_input_contract.md` specifies the numerical input programs. "
                if profile.raw_rtl_study and profile.level == 1
                else "`interface_grammar.md` and `interface_dialect_contract.yaml` specify the input format. "
            )
            + "`command_buffer_abi.yaml`, `target_dialect_contract.yaml` and `schemas/` specify "
            "the external output and runtime protocol. "
            "The selected raw hardware files are under `raw-rtl/`. Derive target instruction behavior "
            "from those files; extracted facts and the derived target contract are not author inputs. "
            "The ordinary LLVM/MLIR installation is under `install/`; `MERLIN_MLIR_INSTALL` selects "
            "its prefix when supplied to an authored build."
            if profile.raw_rtl_study and not profile.disclosure["rtl_facts"]
            else "- `mlir_oot_backend_contract.yaml` defines the submitted manifest, build and entrypoint contract. "
            "Read `interface_grammar.md`, `interface_dialect_contract.yaml`, `command_buffer_abi.yaml`, "
            "`target_dialect_contract.yaml`, `target_contract.yaml`, `schemas/` and `isa/` for the public "
            "source, output, ABI and instruction declarations. The common LLVM/MLIR installation is exposed "
            "under `install/`; `MERLIN_MLIR_INSTALL` selects the installation prefix when supplied "
            "to an authored build."
        ),
        "- `AUTHOR_PROFILE.json` records the exact selected author tools and source boundary. "
        "Use only that selection and the granted public inputs. This runtime block adds no "
        "mandatory tool call, discovery sequence, construction order or optimization policy.",
    ]
    block = "\n".join(lines) + "\n"
    task = Path(workspace) / "TASK.md"
    body = task.read_text()
    if HEADER in body:
        raise ValueError("profile runtime guidance was already staged")
    for root in (Path(workspace), Path(run_dir)):
        _write_once(root / PUBLIC_CASES, encoded)
    task.write_text(body + block)
    (Path(run_dir) / "TASK.md").write_bytes(task.read_bytes())
    return roster


def _metrics_roster(context, *, public_roots=None, contract=None):
    """Read the same admitted public core, scale and program owners as grading."""
    from merlin.common.paths import resolve_grant
    from merlin.development.backend.feedback.certification import model_checks as public_programs
    from merlin.development.backend.feedback.scalability import public_scalability
    from merlin.development.targetgen.corpus.capsule_io import discover_capsules
    from merlin.target.registry.descriptor_loading import load_target_experiment

    experiment = load_target_experiment(context.descriptor)
    roots = experiment.graded_roots() if public_roots is None else public_roots
    core = discover_capsules(
        roots, labels={"public", "dev"}, contract=contract or resolve_grant("merlin/contract", root=context.repo)
    )
    excluded = set(experiment.effective_exclusions(cap["name"] for cap in core))
    suites = {"core": sorted(cap["name"] for cap in core if cap["name"] not in excluded)}
    scalability = public_scalability.selected(context)
    suites["scalability"] = sorted(scalability[1]["cases"]) if scalability else []
    programs = public_programs.selected(context)
    suites["programs"] = sorted(programs[2]) if programs else []
    names = [name for values in suites.values() for name in values]
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("study metrics require a nonempty actual public roster")
    if len(names) != len(set(names)):
        raise ValueError("study metric case identities must be unique across suites")
    return suites
