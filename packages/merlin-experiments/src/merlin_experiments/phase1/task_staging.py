"""Actual Phase 1 task composition and observed bundle tool selection.

Public prompt templates are package resources owned by generate_prompt. Authored
tasks, bundle documents and descriptor policy are explicit operator inputs, never
guessed from this module's installation path. Import and callback creation do not
read those inputs; admission chooses when staging and tool observation occur.
"""

from __future__ import annotations

import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import yaml

from merlin.targetgen import tool_registry as _TR
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
    scope = task_runtime_scope(te, sandbox, context=context) if scope is None else scope
    if sandbox == "bwrap":
        isolation = "deny-by-default bwrap; allowed inputs are the frozen run snapshot and answer surfaces are masked"
    else:
        isolation = "unsandboxed diagnostic override; this launch cannot support a trusted isolation claim"
    return (
        "\n\n## Runtime scope (generated for this launch; authoritative)\n"
        f"- Required public/dev capsules: **{scope['required_public_dev_capsules']}**, derived from "
        "the descriptor's graded roots, label filter, and exclusions. Completion is non-vacuous only "
        "when every required member passes.\n"
        f"- Held-out capsules: **{scope['held_out_capsules']}**, derived from the descriptor's hidden "
        "roots; their contents remain sealed.\n"
        f"- Active sandbox: **`{sandbox}`** ({isolation}).\n"
        "- The harness re-grades a snapshot of your workspace on its own schedule and refreshes "
        "`qa/verdict.json` underneath you; you are NOT relaunched between grades. To wait for the "
        "next one instead of checking repeatedly, run **`python await_verdict.py`** — it blocks until "
        "a new grade lands and prints its score and failing capsules (`--timeout <s>` to bound the "
        "wait; a timeout is reported, not an error). Polling `qa/verdict.json` in a loop costs you a "
        "turn every time you look.\n"
        "- If an older bundled document states a fixed capsule count, a different isolation mode, or "
        "that you are relaunched each round, this launch-generated block wins.\n"
    )


def build_task(
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
    if experiment == "realistic":
        # whole-repo + self-check tool + self-paced READY marker; TASK_realistic is self-contained,
        # so skip the full experiment's grading-tier addendum appended below.
        ws_task = ws / "TASK.md"
        # A target that ships a hand-authored realistic task uses it (gemmini); a descriptor-only target
        # (e.g. atlas) has none, so fall back to the GENERATED target-agnostic prompt — exactly what the
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
            "(`oot_starterkit/` — dialect.py / transforms.py / verify.py — + `xdsl_dialects/`): define the "
            "target dialect as xDSL ops with verifiers, and the interface->target lowering as xDSL rewrite "
            "passes. This is the approach this arm exists to exercise.\n"
            f"- `manifest.yaml` MUST declare `language: python`; the tool is an executable Python `{_stem}` "
            "exposing the 4 entrypoints. Do NOT author a C++/TableGen tool or a `build` block that compiles "
            "one (no cmake/`mlir-tblgen`/`*-opt` C++ binary) — a hand C++ backend is NOT acceptable for this "
            "arm. All integrity rules still apply (`integrity_exempt: false`).\n"
        )
    # Tier wording is TARGET-AGNOSTIC: name the target's own oracle tiers from the manifest, not the
    # gemmini spike/verilator literals (atlas's loop tier is the arc program-oracle, its checkpoint the
    # cycle-accurate RTL cosim/Verilator). `_loop`/`_ckpt` are the tier keys the runner resolves.
    from merlin.targetgen import capsule_runner as CR

    _loop = min(
        CR.qa_loop_adapters(
            _te().target, _te().sim_via, declared_tiers=LG.declared_loop_tiers(A.policy_roots(context, policy_root))
        )
        or {"L3": 1}
    )
    _ckpt = max(CR.qa_checkpoint_adapters(_te().target, _te().sim_via) or {"L3": 1})
    pilot += task_runtime_scope_block(_te(), sandbox, context=context, scope=task_scope)
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
