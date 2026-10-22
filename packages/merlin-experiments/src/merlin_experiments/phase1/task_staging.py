"""Actual Phase 1 task composition and observed bundle tool selection.

Public prompt templates are package resources owned by generate_prompt. Authored
tasks, bundle documents and descriptor policy are explicit operator inputs, never
guessed from this module's installation path. Import and callback creation do not
read those inputs; admission chooses when staging and tool observation occur.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml

from merlin.targetgen import tool_registry as _TR
from merlin.targetgen.generalization_prompt import append_general_compiler_contract
from merlin.targetgen.target_experiment import load_capability_manifest, load_target_experiment

from . import authoring as A
from . import run_inputs as RI
from .context import InvocationContext
from .feedback import loop_grading as LG


@dataclass(frozen=True)
class TaskStagingConfig:
    context: InvocationContext
    bundle_id: str
    bundle_dir: Path
    experiment: str
    language: str = ""
    add_tools: tuple[str, ...] = ()
    drop_tools: tuple[str, ...] = ()
    shared_pass_requirement: dict | None = None
    shared_pass_library: object | None = None


@dataclass(frozen=True)
class TaskCallbacks:
    stage_task: Callable
    resolved_tools: Callable[[], tuple]


def callbacks(config: TaskStagingConfig) -> TaskCallbacks:
    """Bind invocation inputs, retaining inventoried function identities.

    Warnings are deduplicated within this invocation only. Tools and grants are
    reread when called, including after trusted staging; no declaration cache.
    """
    warned: set[Path] = set()

    def stage_task(arm, ws, run_dir, *, sandbox, task_scope, policy_root):
        return build_task(arm, ws, run_dir, sandbox, config=config, task_scope=task_scope, policy_root=policy_root)

    def resolve_selected_tools():
        return resolved_tools(
            config.bundle_id,
            bundle_dir=config.bundle_dir,
            add_tools=config.add_tools,
            drop_tools=config.drop_tools,
            warned=warned,
        )

    return TaskCallbacks(stage_task, resolve_selected_tools)


def granted_merlin_tools(bundle_dir: Path) -> set:
    """The merlin tool paths THIS arm's bundle grants (its ``allowed_files.txt``) — the authoritative,
    per-arm set the enforced-workflow prompt block is derived from, so arm-4's RTL-facts grant in the
    selected rtlchecks bundle distinguishes it from arm-3. Empty when the file is
    absent (render_prompt then falls back to its coarse arm-string gate). Target-agnostic."""
    f = bundle_dir / "allowed_files.txt"
    if not f.is_file():
        return set()
    allowed = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
    tools_file = bundle_dir / "tools.txt"
    if tools_file.is_file():
        from merlin.common.paths import python_source_dir

        names = tuple(line.strip() for line in tools_file.read_text().splitlines() if line.strip())
        return _TR.selected_bundle_tool_paths(allowed, names, python_source_root=python_source_dir())
    # Old authored bundles have no explicit tool inventory. Preserve their
    # historical prompt input, while new releases use the selected-tool set.
    return {path for path in allowed if path.startswith(("merlin/", "experiments/"))}


def resolved_tools(
    bundle_id: str, *, bundle_dir: Path, add_tools: tuple = (), drop_tools: tuple = (), warned: set[Path] | None = None
) -> tuple:
    """The ARM-GATED tools this run actually carries — its ablation cell.

    Resolution order, most authoritative first:

    1. the bundle's generated ``tools.txt`` — a cell is self-describing, so a run launched with
       ``--bundle`` on a pre-generated cell gets the right brokers without repeating the flags;
    2. the bundle-id STEM (longest match against the ladder's stems).

    Not from the arm: the CIRCT driver runs under ``arm == "merlin_assisted"`` with its own bundle,
    so an arm-name test cannot see it. Not from the manifest's ``arm:`` field either — the legacy
    ``merlin_assisted_rtlchecks_public_v0`` declares ``arm: merlin_assisted``, so that field is stale on
    exactly the bundles where being wrong costs the most.

    The ``--with-tool``/``--without-tool`` flags apply last, so an ad-hoc cell needs no regeneration.
    """
    bdir = bundle_dir
    tools_f = bdir / "tools.txt"
    if tools_f.is_file():
        base = tuple(ln.strip() for ln in tools_f.read_text().splitlines() if ln.strip())
    else:
        base = _TR.ARM_TOOLS.get(arm_from_bundle_id(bundle_id), ())
        _warn_if_grants_disagree(bdir, base, warned=warned)
    drop = set(drop_tools)
    out = [t for t in base if t not in drop]
    out += [t for t in add_tools if t not in out]
    manifest_file = bdir / "input_bundle_manifest.yaml"
    if manifest_file.is_file():
        from merlin.targetgen.tool_profile import verify_tool_profile

        manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8")) or {}
        verify_tool_profile(manifest, tools=tuple(out))
    elif set(out) & {"isa_dialect_generation", "lean_proof_authoring"}:
        raise ValueError("optional ISA/Lean tools require an explicit versioned bundle manifest")
    return tuple(out)


def arm_from_bundle_id(bundle_id: str) -> str:
    """The ladder rung a bundle id names, by LONGEST matching stem.

    Longest wins because the stems nest: ``merlin_assisted_rtlchecks_*`` also starts with
    ``merlin_assisted_``, and picking the shorter one silently downgrades the CIRCT arm to the xDSL arm.

    RESOLVED AGAINST ``_ALL_ARMS``, NOT ``_ARMS``. ``_ARMS`` is the DEFAULT ladder -- the arms that get
    materialized into every target's bundles unless ``--arms`` says otherwise -- while ``_OPT_IN_ARMS``
    holds arms that exist and are emitted only when named. Resolving against the default ladder alone
    does not refuse an opt-in bundle; it MIS-RESOLVES it, because every opt-in stem deliberately
    contains ``merlin_assisted`` so the arm inherits the assisted prompt. Measured 2026-09-05:
    ``merlin_assisted_verify_hwbringup_v0`` resolved to ``merlin_assisted`` and would have run with
    arm-3's tool grants under the verify arm's name -- an arm gaining nothing while its result was
    attributed to the seam it was supposed to be testing. A gemmini verify bundle already exists on
    disk, so this was live rather than hypothetical.
    """
    from merlin.targetgen.generate_bundles import _ALL_ARMS

    best, best_len = "", -1
    for arm, stem in _ALL_ARMS.items():
        if bundle_id.startswith(stem + "_") and len(stem) > best_len:
            best, best_len = arm, len(stem)
    if not best:
        raise KeyError(f"bundle id {bundle_id!r} matches no ladder rung (stems: {sorted(_ALL_ARMS.values())})")
    return best


def task_tool_inventory_block(config: TaskStagingConfig) -> str:
    """Describe the actual selected treatment, including explicit tool ablations."""
    names = resolved_tools(
        config.bundle_id,
        bundle_dir=config.bundle_dir,
        add_tools=config.add_tools,
        drop_tools=config.drop_tools,
    )
    lines = [
        "\n\n## Selected Phase 1 tool inventory (launch-specific)\n",
        "Use the common feedback clients staged for this launch: "
        + ", ".join(f"`{filename}`" for _, filename in _TR.COMMON_CLIENTS)
        + ". Read their help for the admitted commands and scope.",
        "The treatment tools below come from the selected bundle and explicit add/drop choices. "
        "Their selection grants no access beyond the admitted mounts. A declared tool is not proof "
        "of runtime readiness; report failed or unavailable checks rather than claiming they passed.",
    ]
    for name in names:
        tool = _TR.spec(name)
        access = (
            "Broker client: " + ", ".join(f"`{filename}`" for _, filename in tool.broker.shims)
            if tool.broker is not None
            else "Use its admitted library/source paths and documentation in the bundle inventory."
        )
        lines.append(f"- `{name}`: {tool.blurb} {access}")
    if not names:
        lines.append("No additional treatment tools are selected; use the common substrate and admitted toolchain.")
    manifest = config.bundle_dir / "input_bundle_manifest.yaml"
    if manifest.is_file():
        from merlin.targetgen.tool_profile import tool_profile_prompt

        lines.append(tool_profile_prompt(yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}))
    return "\n".join(lines) + "\n"


def _warn_if_grants_disagree(bdir: Path, tools: tuple, *, warned: set[Path] | None = None) -> None:
    """Surface a bundle whose NAME, BINDINGS and PROMPT disagree, instead of running on the mismatch.

    Three surfaces must agree about which tools an arm has, and they are written by different code:

      * the rung (this bundle's stem)      -> decides which brokers the driver starts;
      * ``input_bundle_manifest.yaml``     -> what the sandbox actually binds (fully regenerated);
      * ``allowed_files.txt``              -> what the PROMPT tells the agent it has, via
        :func:`granted_merlin_tools`. It is written only-when-absent, so on a hand-authored bundle it
        goes stale while the manifest moves on.

    When the prompt and the bindings disagree the arm is either told about a tool it cannot open, or
    silently handed one it is never told about — and in both cases the treatment is not the one the
    run's label claims. That is not hypothetical: an arm once ran a whole campaign without executing a
    single one of its generators because its prompt never mentioned them.

    Advisory, not fatal, and reported once per bundle: several legacy bundles predate the generator.
    """
    if warned is not None:
        identity = bdir.resolve()
        if identity in warned:
            return
        warned.add(identity)
    man_f, files_f = bdir / "input_bundle_manifest.yaml", bdir / "allowed_files.txt"
    if not man_f.is_file():
        return
    man = yaml.safe_load(man_f.read_text()) or {}
    bound = {e.get("path") for e in (man.get("allowed") or []) if isinstance(e, dict)}
    for name in _TR.known_tools():
        t = _TR.spec(name)
        if not t.bundle_paths:
            continue  # brokered: staged, never bound -- nothing to compare
        if (set(t.bundle_paths) <= bound) != (name in tools):
            state = "binds" if set(t.bundle_paths) <= bound else "does not bind"
            print(
                f"  note: bundle {bdir.name} {state} {name!r} but its rung says otherwise; "
                f"the rung decides which brokers start",
                file=sys.stderr,
            )
    if files_f.is_file():
        told = {ln.strip() for ln in files_f.read_text().splitlines() if ln.strip()}
        unadvertised = {p for p in bound - told if p.startswith("merlin/python/")}
        if unadvertised:
            print(
                f"  WARNING: bundle {bdir.name} BINDS {len(unadvertised)} merlin tool path(s) that its "
                f"allowed_files.txt does not list, so the prompt will not mention them: "
                f"{sorted(unadvertised)[:4]}{' ...' if len(unadvertised) > 4 else ''}",
                file=sys.stderr,
            )


def task_runtime_scope(
    te, sandbox: str, *, context: InvocationContext, public_roots=None, hidden_roots=None, contract: Path | None = None
) -> dict:
    """Native context adapter for the authoritative admission task-scope policy."""
    from merlin_experiments.phase1.session import task_scope

    return task_scope(
        te, sandbox, repo=context.repo, public_roots=public_roots, hidden_roots=hidden_roots, contract=contract
    )


def task_runtime_scope_block(te, sandbox: str, *, context: InvocationContext, scope: dict | None = None) -> str:
    """Agent-facing launch facts; authoritative over static bundle prose."""
    from . import joint_control

    scope = task_runtime_scope(te, sandbox, context=context) if scope is None else scope
    joint = joint_control.selected()
    cohort = "Core public/dev capsules" if joint else "Required public/dev capsules"
    access = ""
    if joint:
        held = joint["policy"].get("final_only_capsules", ())
        access = (
            "- Core and separately selected scalability cases are distinct public rosters; "
            "the scalability section gives its additional count and source index. Use the default "
            "core suite for core IDs and `--suite scalability` for scalability IDs.\n"
            f"- **{len(held)} core capsules are final-only** (already included in the core count): "
            + (", ".join(f"`{name}`" for name in held) or "none") + ". "
            "They remain pending in development grades. Full-model execution, including small "
            "model capsules, is gated to the selected final Spike window; full target program "
            "obligations are separate from the capsule count.\n"
            "- Automatic grading uses a rolling unresolved queue with bounded snapshot admission time. "
            "The optional 1–16-ID impact hint sets priorities; it is "
            "not an author access limit. Name any admitted public development IDs in focused "
            "requests through the appropriate suite; hidden members remain inaccessible and "
            "final-only/model and engine gates still apply.\n"
        )
    if sandbox == "bwrap":
        isolation = "deny-by-default bwrap; allowed inputs are the frozen run snapshot and answer surfaces are masked"
    else:
        isolation = "unsandboxed diagnostic override; this launch cannot support a trusted isolation claim"
    return (
        "\n\n## Runtime scope (generated for this launch; authoritative)\n"
        f"- {cohort}: **{scope['required_public_dev_capsules']}**, derived from "
        "the descriptor's graded roots, label filter, and exclusions. Completion is non-vacuous only "
        "when every required member passes.\n"
        f"- Held-out capsules: **{scope['held_out_capsules']}**, derived from the descriptor's hidden "
        "roots; their contents remain sealed.\n"
        + access
        + f"- Active sandbox: **`{sandbox}`** ({isolation}).\n"
        "- The harness re-grades a snapshot of your workspace on its own schedule and refreshes "
        "`qa/verdict.json` underneath you; you are NOT relaunched between grades. To wait for the "
        "next one instead of checking repeatedly, run **`python await_verdict.py`** — it blocks until "
        "a new grade lands and prints its score and failing capsules (`--timeout <s>` to bound the "
        "wait; a timeout is reported, not an error). Polling `qa/verdict.json` in a loop costs you a "
        "turn every time you look.\n"
        "- If an older bundled document states a fixed capsule count, a different isolation mode, or "
        "that you are relaunched each round, this launch-generated block wins.\n"
    )


def full_model_task(config: TaskStagingConfig, sandbox: str, scope: dict) -> str:
    """The Gemmini Phase 1 charter, with no unenforced process checklist."""
    names = resolved_tools(
        config.bundle_id, bundle_dir=config.bundle_dir, add_tools=config.add_tools, drop_tools=config.drop_tools
    )
    tools = "\n".join(f"- `{name}`: {_TR.spec(name).blurb}" for name in names)
    from . import joint_control

    joint = joint_control.selected()
    body = """# Phase 1: Gemmini compiler for complete models

Build the compiler in `submission/` from the empty starting directory. The required programs are:

| Model | Programs |
| --- | --- |
| ResNet50 | `model` |
| SmolVLA | `prefix_encode`, `flow_denoise`, `action_decode` |
| TinyLlama | `prefill`, `decode` |

All six programs must compile and link. During Phase 1, numerical checks require every device kernel
to replay bit-exactly on Spike, including sealed inputs and sealed shape variants, a fixed-denominator
offload census, and public capsule L3 gSIM qualification. Whole-model Spike chain validation runs
on the frozen compiler after the experiment, outside its authoring clock;
per-kernel model gSIM replay is not requested.
`qa/verdict.json` reports your
visible-input results. Sealed-input results appear only in the official grade.

The frozen `task/PHASE2_TARGET.md` describes the planned FireSim deployment
target and its relationship to this release's selected RTL facts. Read the
selected facts and contract for the exact Phase 1 compilation target; retain
their identities when reporting any timing observation.

## Deliverable

Write a self-contained `submission/` with `manifest.yaml`, compiler sources, the four required
parse/lower/emit/LLVM entrypoints defined by the admitted contract, and `REPORT.md` stating
measured results and limitations. The model-check contract uses `numerics.approximations` in the
manifest for declared approximate kernels. Undeclared device kernels must match the integer
reference exactly. Declared approximate kernels and host f32 work must meet the contract's
error policy.

## Available tools

- `python agent_selfcheck.py --submission submission --capsules A,B`: focused public capsule checks.
  Without `--sim` it runs every tier each capsule declares mandatory (Spike, L3 gSIM, provided host)
  and reports a certified pass/fail per capsule; `--sim spike` is a faster screen that cannot certify.
  Add `--wait-budget 5` to get a request id back at once, keep working, and collect the verdict with
  `--attach ID`; `--list` shows your requests. Several requests may be in flight. Put every capsule an
  edit affects in one request: its capsules run in parallel and per-request overhead dominates.
  Check edits by naming the affected capsules; `--capsules changed` rebuilds every capsule (a slow
  sweep for occasional regression checks); `--capsules failing` re-checks your last check's failures.
- `--submission variants/NAME`: self-check an alternative kept in `variants/NAME/` (same layout as
  `submission/`) beside it. Variant results are labelled and never graded officially; only
  `submission/` counts, so copy the winner there.
- `python simjob.py --help`: asynchronous public capsule jobs (`--sim required` runs every tier).
- Before expensive simulation, stress applicable public accelerator members with
  `python agent_selfcheck.py --compile-only --scale-probe --capsules PUBLIC_ID --timeout 120`.
  Replace `PUBLIC_ID` with one or two declared public member IDs. The harness derives source-only
  scale/tail cases from those members and reports lowering failures, timeouts and emitted LLVM growth.
  Include public convolution and output-readout families when applicable; small matmuls alone do not
  test their compilation cost. This mode allocates no tensor values and reads no private model shapes.
- Then use `python agent_selfcheck.py --compile-only --capsules PUBLIC_ID --timeout 120` for bounded
  object compilation, linking and ABI/host-work audits. Inspect the reported failure stage before
  retrying. A successful emission or link grants no numerical, offload, trace or full-model credit.
- `python await_verdict.py --help`: wait for a new harness verdict.
- `python agent_model_check.py check --model MODEL --out REPORT.json`: visible model check through the broker.
- `python agent_model_check.py replay-kernel --model MODEL --program PROGRAM`
  `--group SOURCE_REGION_ID --engine spike`: visible kernel replay through the broker.
  A pending reply includes a `wait --job-id` command.

Selected treatment tools:
"""  # target-ok: the selected Gemmini full-model task names its target
    body += tools or "- No additional treatment tools are selected."
    body += """

## Sandbox and enforced rules

- Use only the frozen inputs and tools granted in this workspace. Do not access hidden inputs,
  goldens, prior backend submissions, or operator-private paths.
- `integrity_exempt: false`; the compiler must be agent-authored and use the selected xDSL/Python
  language and CLI contract. Do not import Merlin's reference or simulator implementation into
  the submission, or substitute C compute kernels for emitted device code.
- Keep `manifest.yaml` and the four entrypoints valid. The harness checks package integrity,
  public capsules (including L3 gSIM), exact model-kernel replays, and the offload census.
  A pending or unavailable check is not a pass.
- The harness grades immutable snapshots. You may continue editing while a self-check or grade runs.
  A verdict names the snapshot it graded; check its source digest before treating it as current.

## Final status

Lead with a table of all six programs: model, program, compile/link, numerics, and census.
Report capsule RTL status separately. Then name the submission path, test artifacts, and remaining failures. Report only
measured results.
"""  # target-ok: this is the selected Gemmini full-model task text
    if joint:
        body = body.replace(
            "offload census, and public capsule L3 gSIM qualification. Whole-model Spike chain validation runs\n"
            "on the frozen compiler after the experiment, outside its authoring clock;",
            "offload census, and separately tracked physical obligations. Development defaults to Spike;\n"
            "only selected small gSIM witnesses are admitted. Full-model Spike prevalidation is gated\n"
            "to the selected final model window inside the authoring clock;",
        ).replace(
            "Without `--sim` it runs every tier each capsule declares mandatory (Spike, L3 gSIM, provided host)\n"
            "  and reports a certified pass/fail per capsule; `--sim spike` is a faster screen that cannot certify.",
            joint_control.selfcheck_routing_guidance(),
        ).replace(
            "(`--sim required` runs every tier)",
            "(`--sim required` also selects Spike in this joint campaign)",
        ).replace(
            "public capsules (including L3 gSIM), exact model-kernel replays, and the offload census.",
            "public development capsules on Spike, admitted gSIM witnesses, exact model-kernel replays,\n"
            "  and the offload census. Unrun declared physical tiers remain pending.",
        )
    body += task_runtime_scope_block(
        load_target_experiment(config.context.descriptor), sandbox, context=config.context, scope=scope
    )
    if len(body.encode("utf-8")) > 10_000:
        raise RuntimeError(
            "Gemmini full-model task exceeds its 10 KB pre-resource budget"  # target-ok: selected full-model path
        )
    return body


def _shared_pass_record(config, run_dir):
    if config.shared_pass_requirement is None:
        return None
    from merlin.targetgen.compiler_library import CompilerLibrarySelection
    from merlin.targetgen.generate_prompt import shared_pass_prompt_record

    from .compiler_library import _check_contract

    library = config.shared_pass_library
    if type(library) is not CompilerLibrarySelection:
        raise ValueError("shared-pass task requires the original selected compiler library")
    original = library.record()
    snapshot = CompilerLibrarySelection(library.contract, run_dir / "compiler_library/python")
    _check_contract(snapshot, run_dir / "compiler_library/contract.json")
    # These exact additional leaves were admitted by the existing original
    # library transport before task staging. A bundle prefix grants no extras.
    granted = granted_merlin_tools(config.bundle_dir) | {
        member.path for member in snapshot.contract.members if member.module in snapshot.contract.public_modules
    }
    result = shared_pass_prompt_record(config.shared_pass_requirement, granted_tools=granted, library=snapshot)
    if library.record() != original:
        raise ValueError("original shared-pass library changed during task staging")
    return result


def verify_shared_pass_prompt(config, workspace, run_dir):
    """Reconstruct from live original inputs, never from saved successful JSON."""
    from merlin.targetgen.generate_prompt import shared_pass_prompt_block

    from .feedback.public_scalability import prompt_block

    from . import joint_control
    block = prompt_block(load_target_experiment(config.context.descriptor)) + joint_control.prompt_block()
    if block and (
        (workspace / "TASK.md").read_text().count(block) != 1
        or (workspace / "TASK.md").read_bytes() != (run_dir / "TASK.md").read_bytes()
    ):
        raise ValueError("required public scalability instructions changed")
    selected = _shared_pass_record(config, run_dir)
    files = {
        run_dir / "shared_pass_requirement.json": None if selected is None else selected["requirement"],
        run_dir / "shared_pass_selection.json": selected,
    }
    for path, value in files.items():
        if value is None:
            if path.exists() or path.is_symlink():
                raise ValueError("resume refused: shared-pass requirement was removed")
        elif (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or any(parent.is_symlink() for parent in path.parents)
            or path.stat().st_mode & 0o222
            or path.read_bytes() != (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        ):
            raise ValueError("shared-pass requirement or original library binding changed; cannot resume")
    if selected is None:
        for root in (workspace, run_dir):
            task = root / "TASK.md"
            if task.is_file() and "## Selected shared-pass compiler requirements (v" in task.read_text():
                raise ValueError("resume refused: shared-pass requirement was removed from the launch")
    if selected is not None:
        body = (workspace / "TASK.md").read_text()
        if body.count(shared_pass_prompt_block(selected)) != 1 or (run_dir / "TASK.md").read_text() != body:
            raise ValueError("shared-pass requirement is absent or changed in the actual author task")
    return None if selected is None else selected["requirement"]


def _build_selected_task(arm, ws, run_dir, sandbox="bwrap", *, config, task_scope=None, policy_root=None):
    """Keep every legacy task byte; opt in equally for every original task body."""
    if config.shared_pass_requirement is None:
        return _build_task(arm, ws, run_dir, sandbox, config=config, task_scope=task_scope, policy_root=policy_root)
    from merlin.targetgen.generate_prompt import shared_pass_prompt_block

    selected = _shared_pass_record(config, run_dir)
    _build_task(arm, ws, run_dir, sandbox, config=config, task_scope=task_scope, policy_root=policy_root)
    for name, value in (
        ("shared_pass_requirement.json", selected["requirement"]),
        ("shared_pass_selection.json", selected),
    ):
        path = run_dir / name
        with path.open("xb") as stream:
            stream.write((json.dumps(value, sort_keys=True, indent=2) + "\n").encode())
        path.chmod(0o444)
    task = ws / "TASK.md"
    task.write_text(task.read_text() + shared_pass_prompt_block(selected))
    (run_dir / "TASK.md").write_bytes(task.read_bytes())
    # Validate after the complete development-policy block has been staged.
    # A joint campaign can select that block even without a scalability suite.


def _build_task(
    arm: str,
    ws: Path,
    run_dir: Path,
    sandbox: str = "bwrap",
    *,
    config: TaskStagingConfig,
    task_scope: dict | None = None,
    policy_root: Path | None = None,
) -> None:
    """Stage the workspace TASK.md. Both arms get the IDENTICAL graded contract (TASK_pilot.md). The
    merlin arm appends TASK_ADDENDUM.md (merlin-specific tool guidance + provenance ask) and stages
    the merlin-only docs into the workspace so the agent can read them. The graded pilot contract is
    never altered — the addendum only adds allowances, so grading stays apples-to-apples."""
    context = config.context
    bundle_dir, experiment = config.bundle_dir, config.experiment
    _te = partial(load_target_experiment, context.descriptor)
    _manifest = partial(load_capability_manifest, context.target)
    if (
        context.target == "gemmini"  # target-ok: This is the explicitly selected Gemmini full-model Phase 1 path
        and arm == "merlin_assisted"
        and experiment == "full"
    ):
        body = full_model_task(config, sandbox, task_scope)
        (ws / "TASK.md").write_text(body)
        shutil.copy(ws / "TASK.md", run_dir / "TASK.md")
        for doc in RI.MERLIN_WS_DOCS:
            src = bundle_dir / doc
            if src.exists():
                shutil.copy(src, ws / doc)
        return
    if experiment == "realistic":
        # whole-repo + self-check tool + self-paced READY marker; TASK_realistic is self-contained,
        # so skip the full experiment's grading-tier addendum appended below.
        ws_task = ws / "TASK.md"
        # A target that ships a hand-authored realistic task uses it; a descriptor-only target
        # has none, so fall back to the GENERATED target-agnostic prompt — exactly what the
        # 'full' branch below already does. render_prompt is the COMPLETE per-arm task (incl. the seam menu
        # for the assisted/CIRCT arms), so the bundle STARTER_PROMPT is not re-appended in that case.
        _task_md = _te().resource_path("task/TASK_realistic.md")
        _generated_task = not _task_md.is_file()
        if _generated_task:
            from merlin.targetgen.generate_prompt import render_prompt

            body = render_prompt(_te(), _manifest(), "realistic", arm, granted_tools=granted_merlin_tools(bundle_dir))
        else:
            body = _task_md.read_text()
        if config.language.strip().lower() == "cpp":
            _opt = f"{context.target}-opt"  # the OOT MLIR tool name (derived from the active target)
            body += (
                "\n\n## Language mandate: C++ out-of-tree MLIR (REQUIRED for this run)\n"
                "- `manifest.yaml` MUST declare `language: cpp` and a `build` block that builds a real "
                f"out-of-tree MLIR tool `mlir_oot/build/bin/{_opt}` against the provided LLVM/MLIR-23 "
                "(`-DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR`); the runner builds it before grading.\n"
                "- Implement the 4 entrypoints as real MLIR passes in a C++ OOT package (input dialect + "
                f"{context.target} target dialect + conversions + the `{_opt}` tool). A Python tool is NOT "
                "acceptable for this run. All integrity rules still apply.\n"
            )
        bdir = bundle_dir
        # STARTER_PROMPT.md carries the ARM-SPECIFIC guidance (CIRCT generators for the +CIRCT arm,
        # verified-IR/kit for merlin, C++ method for baseline). It MUST be delivered or the arm's whole
        # approach is invisible to the agent (abc8: the CIRCT arm never ran a single generator because this
        # was missing). Append it to TASK.md (the agent always reads TASK.md) for every arm.
        # When the task body was GENERATED (render_prompt above), it already carries the arm's full
        # approach — appending the bundle STARTER_PROMPT (also render_prompt) would just duplicate it.
        starter = (
            ""
            if _generated_task
            else ((bdir / "STARTER_PROMPT.md").read_text() if (bdir / "STARTER_PROMPT.md").exists() else "")
        )
        if starter:
            body += "\n\n---\n\n# Starter plan / approach for THIS arm (read this)\n\n" + starter
        body += task_runtime_scope_block(_te(), sandbox, context=context, scope=task_scope)
        body += task_tool_inventory_block(config)
        body = append_general_compiler_contract(body)
        if arm == "merlin_assisted":
            add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
            ws_task.write_text(body + ("\n\n---\n\n" + add if add else ""))
            for doc in RI.MERLIN_WS_DOCS:
                src = bdir / doc
                if src.exists():
                    shutil.copy(src, ws / doc)
        else:
            ws_task.write_text(body)
        shutil.copy(ws_task, run_dir / "TASK.md")
        return
    # The graded contract is now the GENERATED (target-agnostic) prompt: ONE shared skeleton + slots
    # DERIVED from {descriptor + RTL fact bundle + endpoint}, so a per-target committed TASK_full.md is no
    # longer the source of truth (its content is covered by the generated body — proven by the dry-run
    # diff). The runtime-only operational blocks below (language mandate + descriptor-derived scope and
    # grading tiers) are still appended: they carry run-specific facts the target-agnostic template
    # deliberately omits.
    from merlin.targetgen.generate_prompt import render_prompt

    pilot = render_prompt(_te(), _manifest(), experiment, arm, granted_tools=granted_merlin_tools(bundle_dir))
    # Language mandate (env PILOT_LANG=cpp|python). The C++ arms (baseline/cpp_merlininfra) are forced to
    # the status-quo C++ OOT MLIR; the merlin arms (arm-3 merlin_assisted + arm-4 merlin_assisted_rtlchecks,
    # both carry "merlin_assisted") DEFAULT to the xDSL/Python path — the whole point of those arms is to
    # build the dialect with the granted xDSL kit, NOT a hand C++/TableGen tool. Arm-driven so it holds
    # however the run is launched (a merlin arm never "chooses C++"). tool stem is target-agnostic.
    _stem = f"{_te().target}-opt"
    _lang = config.language.strip().lower() or ("python" if "merlin_assisted" in arm else "")
    if _lang == "cpp":
        pilot += (
            "\n\n## Language mandate: C++ out-of-tree MLIR (REQUIRED for this run)\n"
            "- `manifest.yaml` MUST declare `language: cpp` and a `build` block "
            "(`configure`/`command`/`tool_output`) that builds a real out-of-tree MLIR tool "
            f"`mlir_oot/build/bin/{_stem}` against the provided LLVM/MLIR-23 "
            "(`-DMLIR_DIR=$MLIR_DIR -DLLVM_DIR=$LLVM_DIR`); the runner builds it before grading.\n"
            "- Implement the 4 entrypoints as real MLIR passes in a C++ OOT package "
            f"(input dialect + target dialect + conversions + the `{_stem}` tool). "
            "A Python tool is NOT acceptable for this run.\n"
            "- Every integrity rule above still applies unchanged (no C compute kernels, no copied "
            "reference kernels, no high-level device libs, no hardcoded outputs). `integrity_exempt: false`.\n"
        )
    elif _lang == "python":
        pilot += (
            "\n\n## Language mandate: xDSL / Python (REQUIRED for this arm)\n"
            "- Build the dialect + the 4 entrypoints with the granted **xDSL kit** "
            "(`oot_starterkit/` — dialect.py / transforms.py / verify.py / plan.py — + `xdsl_dialects/`): define the "
            "target dialect as xDSL ops with verifiers, and the interface->target lowering as xDSL rewrite "
            "passes. This is the approach this arm exists to exercise.\n"
            f"- `manifest.yaml` MUST declare `language: python`; the tool is an executable Python `{_stem}` "
            "exposing the 4 entrypoints. Do NOT author a C++/TableGen tool or a `build` block that compiles "
            "one (no cmake/`mlir-tblgen`/`*-opt` C++ binary) — a hand C++ backend is NOT acceptable for this "
            "arm. All integrity rules still apply (`integrity_exempt: false`).\n"
        )
    # Tier wording is TARGET-AGNOSTIC: name the target's own oracle tiers from the manifest, not one
    # target's spike/verilator literals (a self-hosted ISA's loop tier is the arc program-oracle, its
    # checkpoint the cycle-accurate RTL cosim/Verilator). `_loop`/`_ckpt` are the tier keys the runner resolves.
    from merlin.targetgen import capsule_runner as CR

    _loop = min(
        CR.qa_loop_adapters(
            _te().target, _te().sim_via, declared_tiers=LG.declared_loop_tiers(A.policy_roots(context, policy_root))
        )
        or {"L3": 1}
    )
    _ckpt = max(CR.qa_checkpoint_adapters(_te().target, _te().sim_via) or {"L3": 1})
    pilot += task_runtime_scope_block(_te(), sandbox, context=context, scope=task_scope)
    pilot += task_tool_inventory_block(config)
    pilot = append_general_compiler_contract(pilot)
    pilot += (
        "\n\n## Grading tiers (READ THIS)\n"
        f"- Each round the QA gate runs **L0+L1+trace + your fast RTL oracle tier ({_loop})** and returns "
        "a redacted verdict (pass/fail + failure plane, never goldens). Use it to fix failures.\n"
        f"- When your public capsules pass the loop tier, the harness runs the **cycle-accurate RTL "
        f"checkpoint ({_ckpt})**. If any capsule fails only there, you get **up to {A.VERILATOR_ATTEMPTS} "
        "checkpoint attempts** (a fix round between each) to make it cycle-accurate-correct. Treat RTL "
        "checkpoint failures as real bugs to fix, not noise.\n"
    )
    ws_task = ws / "TASK.md"
    if arm == "merlin_assisted":
        bdir = bundle_dir
        add = (bdir / "TASK_ADDENDUM.md").read_text() if (bdir / "TASK_ADDENDUM.md").exists() else ""
        ws_task.write_text(pilot + "\n\n---\n\n" + add)
        for doc in RI.MERLIN_WS_DOCS:
            src = bdir / doc
            if src.exists():
                shutil.copy(src, ws / doc)
    else:
        ws_task.write_text(pilot)
    shutil.copy(ws_task, run_dir / "TASK.md")  # archive the exact task served, for the record


def build_task(arm, ws, run_dir, sandbox="bwrap", *, config, task_scope=None, policy_root=None):
    _build_selected_task(arm, ws, run_dir, sandbox, config=config, task_scope=task_scope, policy_root=policy_root)
    from .feedback.public_scalability import prompt_block
    from . import joint_control

    block = prompt_block(load_target_experiment(config.context.descriptor)) + joint_control.prompt_block()
    if block:
        task = ws / "TASK.md"
        task.write_text(task.read_text() + block)
        (run_dir / "TASK.md").write_bytes(task.read_bytes())
    verify_shared_pass_prompt(config, ws, run_dir)
