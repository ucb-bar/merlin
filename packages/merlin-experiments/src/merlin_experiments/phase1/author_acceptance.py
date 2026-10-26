"""Pre-launch acceptance: stage an author workspace as a run does, then do what the author does in it.

Every mid-run surprise of one campaign came from a harness component tested alone and never through an
author workflow: withheld packages silently emptied library fusion in the staged tree, multi-MiB results
failed grades, helpers forked full history. This suite runs once on a FROZEN framework export, before a
launch, and answers one question per workflow with PASS/FAIL and the cause.

Two interpreters are involved and never mixed:

* the OPERATOR interpreter imports the export's own harness (``<export>/src`` and
  ``<export>/packages/*/src``) and stages with the owners a run uses -- ``full_compiler_workspace``
  (prepare + stage), ``task_staging.build_task``, ``task_disclosure.stage`` and ``skill_staging.stage``,
  in the order ``session._prepare`` calls them;
* the AUTHOR interpreter is ``python -I`` with only the author-visible roots on ``sys.path``, in the
  sandbox order (``submission/compiler_library`` first, then the frozen public SDK root, then the
  workspace). Unmounted operator roots that the sandbox ``PYTHONPATH`` also names are not added; whether
  they are reachable is answered separately inside a real bwrap sandbox built by the harness's own argv
  composer.

The launch's own pre-spend gate is replayed too: ``tooling-readiness`` freezes the sealed bundle with the
bwrap transport and runs ``tooling_readiness.assess`` in the export's interpreter, so the suite refuses
exactly when ``--preflight-only`` would print "NO_GO: selected authoring tools are not ready" (pass the
launch's ``--compiler-execution`` and ``--compiler-library-*`` selection).

Nothing here edits the export or the inputs; everything is written below one fresh work directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

SCHEMA = "merlin.phase1_author_acceptance.v1"
PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
#: Checks that need neither bwrap, the Codex CLI nor a full module walk; the pytest gate runs these.
FAST = (
    "withheld-imports",
    "fusion",
    "skills",
    "scheduling-example",
    "library-inventory",
    "capacity-schedule-example",
    "heavy-pass-recipes",
    "pass-pipeline",
    "tuning-knobs",
    "large-results",
    "task-disclosure",
)


@dataclass
class Check:
    name: str
    status: str
    cause: str = ""
    detail: dict = field(default_factory=dict)
    seconds: float = 0.0


# ---------------------------------------------------------------------------------------------------
# Child interpreters
# ---------------------------------------------------------------------------------------------------

#: Prepended to every child: argv[1] = sys.path roots (JSON), argv[2] = refused prefixes (JSON).
_PRELUDE = r"""
import importlib, importlib.abc, json, sys
_ROOTS, _REFUSED = json.loads(sys.argv[1]), json.loads(sys.argv[2])
_ATTEMPTS = set()

class _Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == p or name.startswith(p + ".") for p in _REFUSED):
            _ATTEMPTS.add(name)
            raise ModuleNotFoundError("withheld from the author: " + name, name=name)
        return None

if _REFUSED:
    sys.meta_path.insert(0, _Refuse())
# A sandbox mounts the interpreter's own prefix, not the checkouts its editable .pth files name.
_PREFIXES = tuple({sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix})
sys.path[:] = [p for p in sys.path if not p or p.startswith(_PREFIXES)]
sys.path[:0] = _ROOTS
del sys.argv[1:3]

def _outside(roots):
    return sorted(n for n, m in list(sys.modules.items())
                  if (n == "merlin" or n.startswith(("merlin.", "merlin_experiments")))
                  and getattr(m, "__file__", None) and not any(str(m.__file__).startswith(r) for r in roots))

def _emit(value):
    print("\n" + json.dumps(value, default=str), flush=True)
"""


def _child_env(extra: dict | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "MERLIN_"))}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def run_child(
    script: str,
    roots,
    refused=(),
    *args,
    cwd: Path,
    timeout: int = 600,
    env: dict | None = None,
    python: str | None = None,
):
    """``python -I -B`` with exactly ``roots`` prepended to ``sys.path``; returns (process, last JSON line)."""
    command = [
        python or sys.executable,
        "-I",
        "-B",
        "-c",
        _PRELUDE + script,
        json.dumps([str(r) for r in roots]),
        json.dumps(list(refused)),
        *map(str, args),
    ]
    try:
        proc = subprocess.run(
            command, capture_output=True, text=True, env=_child_env(env), cwd=str(cwd), timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(command, -9, "", f"timed out after {exc.timeout}s"), None
    result = None
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            try:
                result = json.loads(line)
                break
            except ValueError:
                continue
    return proc, result


def _tail(proc, n: int = 600) -> str:
    text = (proc.stderr or "").strip() or (proc.stdout or "").strip()
    return text[-n:]


def operator_roots(framework: Path) -> list[Path]:
    """The export's own harness import roots: core ``src`` and every optional distribution's ``src``."""
    roots = [framework / "src"]
    packages = framework / "packages"
    if packages.is_dir():
        roots += sorted(p / "src" for p in packages.iterdir() if (p / "src").is_dir())
    return roots


# ---------------------------------------------------------------------------------------------------
# Staging (operator interpreter, the export's own owners)
# ---------------------------------------------------------------------------------------------------

_STAGE = r"""
import json, sys
from pathlib import Path
framework, inputs, work, driver = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
from merlin_experiments.phase1 import full_compiler_workspace as FCW
from merlin_experiments.phase1 import skill_staging as SK
from merlin_experiments.phase1 import task_disclosure as TD
from merlin_experiments.phase1 import task_staging as TS
from merlin_experiments.phase1.context import load_context
from merlin_experiments.phase1.options import parse_options

record = {"framework": str(framework), "driver": driver}
ws, run_dir = work / "workspace", work / "run"
ws.mkdir(parents=True)
run_dir.mkdir()
# The operator preparation of the public compiler, from the export's own sources (fresh-inputs/full-compiler
# is the same call, made when the inputs were sealed).
selection = FCW.prepare(framework, work / "full-compiler", review_id="author acceptance")
record["full_compiler"] = selection.record()
record["denied_prefixes"] = list(FCW.DENIED_PREFIXES)
record["refused"] = sorted({*FCW.DENIED_PREFIXES, *(m for e in FCW.MODULE_ACCESS for m in e.modules)})
staging = FCW.stage(selection, ws / "submission")
record["staged_members"] = len(staging["copied_missing_members"])
record["full_compiler_modules"] = sorted(m for _p, _d, m in selection.members if m)
if inputs:
    inputs = Path(inputs)
    sealed = json.loads((inputs / "full-compiler-selection.json").read_text()) if (inputs / "full-compiler-selection.json").is_file() else {}
    record["sealed_full_compiler_manifest_sha256"] = sealed.get("manifest_sha256")
    record["fresh_manifest_sha256"] = selection.manifest_sha256
    seal = json.loads((inputs / "seal-result.json").read_text())
    descriptor = Path(seal["descriptor"])
    bundles = sorted(p for p in (descriptor.parent / "input_bundles").iterdir() if p.is_dir())
    record["bundles"] = [p.name for p in bundles]
    bundle = bundles[0]
    counts = seal.get("counts") or {}
    scope = {"required_public_dev_capsules": counts.get("public_admitted"),
             "held_out_capsules": counts.get("hidden_admitted"), "sandbox": "bwrap",
             "scope_source": "sealed input counts"}
    context = load_context(descriptor, repo=framework)
    config = TS.TaskStagingConfig(context, bundle.name, bundle, "full")
    # The runner arm whose name the bundle carries (the launcher selects it with --arm/--level).
    from merlin_experiments.phase1.options import build_parser
    arms = next(a.choices for a in build_parser()._actions if "--arm" in a.option_strings)
    arm = max((a for a in arms if bundle.name.startswith(a)), key=len, default=TS.arm_from_bundle_id(bundle.name))
    TS.build_task(arm, ws, run_dir, "bwrap", config=config, task_scope=scope)
    record.update(task_staged=True, arm=arm, bundle=bundle.name, descriptor=str(descriptor),
                  target=context.target)
else:
    # No sealed inputs: a neutral sectioned task keeps the disclosure owner exercised.
    (ws / "TASK.md").write_text("# Task\n\nShort.\n\n## Budgets\n\n" + "Budget line.\n" * 120)
    record["task_staged"] = False
options = parse_options(["--run-id", "acceptance", "--driver", driver, "--sandbox", "bwrap"])
skills = SK.selection(options) or {}
from merlin_experiments.phase1.providers import codex_context as CTX
context_record = CTX.selection(options) or {}
record["task_skills"] = bool(context_record.get("task_skills"))
if context_record.get("task_skills"):
    record["disclosure"] = TD.stage(ws, run_dir, feedback=not skills.get("enabled"))
if skills.get("enabled"):
    record["skills"] = SK.stage(ws, run_dir, skills)
record["skill_roots"] = {k: v.as_posix() for k, v in SK.ROOTS.items()}
record["feedback_skill"] = SK.FEEDBACK_SKILL
record["workspace"], record["run_dir"] = str(ws), str(run_dir)
_emit(record)
"""


def stage(framework: Path, inputs: Path | None, work: Path, driver: str) -> dict:
    proc, record = run_child(
        _STAGE, operator_roots(framework), (), framework, inputs or "", work, driver, cwd=work, timeout=1800
    )
    if proc.returncode != 0 or record is None:
        raise RuntimeError(f"staging failed: {_tail(proc, 3000)}")
    return record


def author_roots(staged: dict, inputs: Path | None) -> list[Path]:
    """The author-visible import roots in sandbox order (see ``toolchain.sandbox_env``/``paths_for_bundle``
    and ``full_compiler_workspace.apply_policy``): editable library, frozen public SDK, workspace."""
    ws = Path(staged["workspace"])
    roots = [ws / "submission" / "compiler_library"]
    sdk = inputs / "author-library" / "base-sdk" if inputs is not None else None
    if sdk is not None and sdk.is_dir():
        roots.append(sdk)
    return [*roots, ws]


# ---------------------------------------------------------------------------------------------------
# a. Withheld imports
# ---------------------------------------------------------------------------------------------------

_WITHHELD = r"""
import importlib, importlib.util
prefixes, candidates = json.loads(sys.argv[1]), json.loads(sys.argv[2])
reachable = {}
for name in [*prefixes, *candidates]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        continue
    origin = spec.origin or (list(spec.submodule_search_locations or []) or [None])[0]
    try:
        importlib.import_module(name)
        reachable[name] = {"origin": origin, "imports": True}
    except BaseException as exc:
        reachable[name] = {"origin": origin, "imports": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
_emit({"reachable": reachable, "path": sys.path[:6]})
"""


def _withheld_candidates(roots, prefixes) -> list[str]:
    """Every module file under an author root that a withheld prefix names (derived from the tree)."""
    found = set()
    for root in roots:
        base = Path(root) / "merlin"
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            parts = list(path.relative_to(root).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            name = ".".join(parts)
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                found.add(name)
    return sorted(found)


def withheld_names(staged: dict) -> list[str]:
    """Every identity withheld from the author: the package prefixes AND the access policy's module-level
    grader/oracle denials (``MODULE_ACCESS``, every reviewed spelling). A record from an older stage
    script without ``refused`` falls back to the prefixes alone."""
    return sorted({*staged["denied_prefixes"], *staged.get("refused", ())})


def check_withheld(ctx) -> Check:
    names = withheld_names(ctx.staged)
    candidates = _withheld_candidates(ctx.roots, names)
    proc, result = run_child(_WITHHELD, ctx.roots, (), json.dumps(names), json.dumps(candidates), cwd=ctx.ws)
    if result is None:
        return Check("withheld-imports", FAIL, f"probe failed: {_tail(proc)}")
    reachable = {k: v for k, v in result["reachable"].items() if v.get("imports")}
    detail = {
        "author_sys_path": [str(r) for r in ctx.roots],
        "candidates": candidates,
        "resolvable": result["reachable"],
    }
    if reachable:
        where = sorted({str(Path(v["origin"] or "?").parent) for v in reachable.values()})
        return Check(
            "withheld-imports",
            FAIL,
            f"{len(reachable)} withheld module(s) import for the author, e.g. {sorted(reachable)[:4]} from {where[:2]}",
            detail,
        )
    if candidates:
        # Source on the author path is exposure even when its own imports fail: it can be read.
        return Check(
            "withheld-imports",
            FAIL,
            f"{len(candidates)} withheld module source file(s) on the author path, e.g. {candidates[:4]}",
            detail,
        )
    return Check(
        "withheld-imports",
        PASS,
        f"{len(ctx.staged['denied_prefixes'])} prefixes and {len(names)} withheld identities refused "
        "on the author path",
        detail,
    )


_SANDBOX_PROBE = (
    "import importlib, json, sys\n"
    "out = {}\n"
    "for n in json.loads(sys.argv[1]):\n"
    "    try:\n"
    "        m = importlib.import_module(n); out[n] = getattr(m, '__file__', None) or str(list(m.__path__))\n"
    "    except BaseException as e:\n"
    "        out[n] = None\n"
    "print(json.dumps({'imported': {k: v for k, v in out.items() if v}, 'path': sys.path}))\n"
)

_SANDBOX_ARGV = r"""
from merlin.targetgen.sandbox import bwrap as BW, toolchain as TC
from merlin.targetgen.target_experiment import load_target_experiment
from merlin_experiments.phase1 import full_compiler_workspace as FCW
from pathlib import Path
ws, descriptor, repo = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])

class _Policy:
    def __init__(self, argv, env_prefix, workspace):
        self.argv, self.env_prefix, self.workspace = argv, env_prefix, workspace

te = load_target_experiment(descriptor)
argv = BW.base_argv(ws, {}, repo=repo, include_claude_home=False, inherit_environment=False)
argv += TC.toolchain_binds(te)
argv = BW.apply_final_answer_masks(argv, te, ws, {}, repo=repo)
from merlin_experiments.phase2.compiler_libraries import _environment_prefix
env = TC.sandbox_env(te, ws) + " " + _environment_prefix("", ws / "submission" / FCW.DIRECTORY)
_emit({"argv": argv, "env": env})
"""


def check_withheld_sandbox(ctx) -> Check:
    if shutil.which("bwrap") is None:
        return Check("withheld-in-bwrap", SKIP, "bwrap is not installed")
    if not ctx.staged.get("descriptor"):
        return Check("withheld-in-bwrap", SKIP, "needs --inputs (the descriptor selects toolchain binds)")
    proc, built = run_child(
        _SANDBOX_ARGV, operator_roots(ctx.framework), (), ctx.ws, ctx.staged["descriptor"], ctx.framework, cwd=ctx.work
    )
    if built is None:
        return Check("withheld-in-bwrap", FAIL, f"sandbox argv composition failed: {_tail(proc)}")
    from shlex import quote

    withheld = withheld_names(ctx.staged)
    names = sorted({*withheld, *_withheld_candidates(operator_roots(ctx.framework), withheld)})
    payload = ctx.work / ".bwrap-acceptance.args"
    payload.write_bytes(b"\0".join(a.encode() for a in built["argv"][1:]) + b"\0")
    inner = f"{built['env']} python3 -c {quote(_SANDBOX_PROBE)} {quote(json.dumps(names))}"
    command = f"exec {{fd}}<{quote(str(payload))} && bwrap --args $fd bash -c {quote(inner)}"
    run = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=600, env=_child_env())
    lines = [line for line in run.stdout.splitlines() if line.startswith("{")]
    if run.returncode != 0 or not lines:
        return Check("withheld-in-bwrap", FAIL, f"sandbox probe did not run: {_tail(run)}")
    seen = json.loads(lines[-1])
    exports = [p for p in built["env"].split(";") if "PYTHONPATH" in p]
    detail = {
        "sandbox_pythonpath_exports": [p.strip() for p in exports],
        "sandbox_sys_path": seen["path"],
        "imported": seen["imported"],
        "probed": len(names),
    }
    if seen["imported"]:
        return Check(
            "withheld-in-bwrap",
            FAIL,
            f"{len(seen['imported'])} withheld module(s) import inside bwrap: {sorted(seen['imported'])[:4]}",
            detail,
        )
    return Check(
        "withheld-in-bwrap",
        PASS,
        f"{len(names)} withheld modules unreachable inside bwrap; operator roots on PYTHONPATH are not mounted",
        detail,
    )


def audit_sandbox_args(path: Path, prefixes) -> Check:
    """Read-only audit of a recorded ``--args`` payload: which mounted files are withheld modules."""
    try:
        argv = [a.decode() for a in path.read_bytes().split(b"\0") if a]
    except OSError as exc:
        return Check("sandbox-args-audit", FAIL, f"cannot read {path}: {exc}")
    mounted = set()
    for i, arg in enumerate(argv):
        if arg in ("--ro-bind", "--bind", "--ro-bind-try", "--bind-try") and i + 2 < len(argv):
            source, destination = argv[i + 1], argv[i + 2]
            if source == "/dev/null" or not destination.endswith(".py"):
                continue
            parts = Path(destination).with_suffix("").parts
            if "merlin" not in parts:
                continue
            tail = list(parts[len(parts) - 1 - list(reversed(parts)).index("merlin") :])
            if tail[-1] == "__init__":
                tail.pop()
            name = ".".join(tail)
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                mounted.add((name, destination))
    detail = {"args_file": str(path), "withheld_mounted": sorted(f"{n} at {d}" for n, d in mounted)}
    if mounted:
        return Check(
            "sandbox-args-audit",
            FAIL,
            f"{len(mounted)} withheld module file(s) mounted, e.g. {sorted(n for n, _ in mounted)[:4]}",
            detail,
        )
    return Check("sandbox-args-audit", PASS, "no withheld module file is mounted", detail)


# ---------------------------------------------------------------------------------------------------
# b. Every staged module imports
# ---------------------------------------------------------------------------------------------------

_IMPORTS = r"""
modules = json.loads(sys.argv[1])
rows = []
for module in modules:
    error = None
    before = set(_ATTEMPTS)
    try:
        importlib.import_module(module)
    except BaseException as exc:  # noqa: BLE001 - every failure is reported
        error = f"{type(exc).__name__}: {exc}"[:300]
    rows.append({"module": module, "error": error, "refused": sorted(_ATTEMPTS - before)})
_emit({"rows": rows, "outside": _outside(_ROOTS)})
"""


def check_imports(ctx) -> Check:
    from concurrent.futures import ThreadPoolExecutor

    modules = ctx.staged["full_compiler_modules"]
    jobs = max(1, min(8, len(os.sched_getaffinity(0))))
    chunks = [modules[i::jobs] for i in range(jobs)]
    library = [Path(ctx.ws) / "submission" / "compiler_library"]

    def run(chunk):
        return run_child(
            _IMPORTS, library, ctx.staged["refused"], json.dumps(chunk), cwd=ctx.ws, timeout=1500, env=ctx.data_env
        )

    with ThreadPoolExecutor(jobs) as pool:
        results = list(pool.map(run, chunks))
    rows, outside = [], set()
    for proc, result in results:
        if result is None:
            return Check("staged-imports", FAIL, f"import walk stopped: {_tail(proc)}")
        rows += result["rows"]
        outside.update(result["outside"])
    failed = {r["module"]: r["error"] for r in rows if r["error"]}
    withheld = {
        r["module"]: r["refused"]
        for r in rows
        if r["refused"]
        and any(any(n == p or n.startswith(p + ".") for p in ctx.staged["denied_prefixes"]) for n in r["refused"])
    }
    detail = {"modules": len(rows), "failed": failed, "reaches_withheld": withheld, "outside": sorted(outside)}
    if withheld or outside or len(failed) > ctx.allowed_import_failures:
        cause = (
            f"{len(withheld)} reach withheld code"
            if withheld
            else f"{len(outside)} resolved outside the staged tree"
            if outside
            else f"{len(failed)} module(s) fail to import, e.g. {sorted(failed)[:3]}"
        )
        return Check("staged-imports", FAIL, cause, detail)
    return Check(
        "staged-imports", PASS, f"{len(rows)} modules import ({len(failed)} declared oracle-dependent)", detail
    )


# ---------------------------------------------------------------------------------------------------
# c. Library fusion
# ---------------------------------------------------------------------------------------------------

_FUSION = r"""
fixtures = json.loads(sys.argv[1])
sys.path[len(_ROOTS):len(_ROOTS)] = fixtures
from merlin.common import mlir_query as mq
from merlin.compile.analysis.contraction_shapes import contraction_extents
from merlin.xdsl_dialects.lowering import compute_groups as CG
from merlin.xdsl_dialects.lowering import output_fusion as OF
import test_output_fusion as T

module = mq.parse(T._module())
extents = contraction_extents(module)
first = CG.form_groups(module, "synthetic", oracle=T._Oracle())
decided = OF.decisions(first, module, T._path(), OF.Values())
groups = CG.form_groups(module, "synthetic", oracle=T._Oracle(), fusion=decided)
device = [g for g in groups if g.placement != CG.HOST]
_emit({"extents": [list(v) for v in extents.values()], "cuts": [g.fusion["cut"] for g in device],
       "refusals": [g.fusion.get("refusals") for g in device], "outside": _outside(_ROOTS)})
"""


def check_fusion(ctx) -> Check:
    tests = ctx.framework / "merlin" / "tests"
    fixtures = [str(tests / "fixtures"), str(tests / "ir")]
    if not (tests / "ir" / "test_output_fusion.py").is_file():
        return Check("fusion", SKIP, "the export carries no output-fusion fixture")
    library = [Path(ctx.ws) / "submission" / "compiler_library"]
    proc, result = run_child(
        _FUSION, library, ctx.staged["refused"], json.dumps(fixtures), cwd=ctx.ws, env=ctx.data_env
    )
    if result is None:
        return Check("fusion", FAIL, f"fusion probe failed: {_tail(proc)}")
    if not result["extents"] or not any(result["extents"]):
        return Check("fusion", FAIL, "contraction_extents is empty on a dense fixture", result)
    if not result["cuts"] or all(cut == "bare" for cut in result["cuts"]):
        return Check("fusion", FAIL, f"output fusion refused every cut but bare: {result['refusals']}", result)
    if result["outside"]:
        return Check("fusion", FAIL, f"modules resolved outside the staged tree: {result['outside'][:4]}", result)
    return Check("fusion", PASS, f"extents {result['extents'][0]}, cut {result['cuts'][0]}", result)


# ---------------------------------------------------------------------------------------------------
# d. Skills: both drivers, identical bytes, every script answers --help
# ---------------------------------------------------------------------------------------------------


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def check_skills(ctx) -> Check:
    roots = {driver: ctx.ws / rel for driver, rel in ctx.staged["skill_roots"].items()}
    if "skills" not in ctx.staged:
        return Check("skills", FAIL, "author skills were not staged for this driver selection")
    names = ctx.staged["skills"]["skills"]
    trees = {driver: {n: _tree(root / n) for n in names if (root / n).is_dir()} for driver, root in roots.items()}
    missing = {d: sorted(set(names) - set(t)) for d, t in trees.items() if set(names) - set(t)}
    if missing:
        return Check("skills", FAIL, f"skills missing under a driver root: {missing}")
    reference = next(iter(trees.values()))
    differ = [d for d, t in trees.items() if t != reference]
    if differ:
        return Check("skills", FAIL, f"skill bytes differ between driver roots: {differ}")
    scripts = sorted(f"{n}/{rel}" for n, files in reference.items() for rel in files if rel.startswith("scripts/"))
    first = next(iter(roots.values()))
    env = {"PYTHONPATH": os.pathsep.join(str(r) for r in ctx.roots), **ctx.data_env}
    failures, outside = {}, {}
    for script in scripts:
        proc = subprocess.run(
            [sys.executable, "-P", "-B", str(first / script), "--help"],
            capture_output=True,
            text=True,
            cwd=str(ctx.ws),
            env=_child_env(env),
            timeout=300,
        )
        if proc.returncode != 0 or "usage" not in proc.stdout.lower() + proc.stderr.lower():
            failures[script] = _tail(proc, 300)
    detail = {"skills": names, "scripts": scripts, "help_failures": failures, "drivers": sorted(roots)}
    if failures:
        return Check("skills", FAIL, f"{len(failures)} script(s) fail --help, e.g. {sorted(failures)[:3]}", detail)
    return Check(
        "skills",
        PASS,
        f"{len(names)} skills x {len(roots)} drivers identical; {len(scripts)} scripts answer --help",
        detail,
    )


_LIBRARY_ORIGIN = r"""
import runpy
script, argv = sys.argv[1], json.loads(sys.argv[2])
sys.argv = [script, *argv]
code = 0
try:
    runpy.run_path(script, run_name="__main__")
except SystemExit as exc:
    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
library = _ROOTS[0]
loaded = sorted({n for n, m in list(sys.modules.items()) if n.startswith("merlin.") and getattr(m, "__file__", None)})
foreign = sorted(n for n in loaded if not str(sys.modules[n].__file__).startswith(library))
_emit({"code": code, "loaded": len(loaded), "foreign": foreign})
"""


def _run_skill_script(ctx, skill: str, script: str, argv: list[str], timeout: int = 900):
    path = ctx.ws / ctx.staged["skill_roots"]["codex"] / skill / "scripts" / script
    return run_child(
        _LIBRARY_ORIGIN,
        ctx.roots,
        ctx.staged["refused"],
        path,
        json.dumps(argv),
        cwd=ctx.ws,
        timeout=timeout,
        env=ctx.data_env,
    )


def check_scheduling_example(ctx) -> Check:
    proc, result = _run_skill_script(ctx, "scheduling-api", "dense_schedule_example.py", ["example"])
    if result is None or result["code"] != 0:
        return Check("scheduling-example", FAIL, f"example did not verify: {_tail(proc)}")
    if result["foreign"]:
        return Check(
            "scheduling-example",
            FAIL,
            f"merlin modules came from outside the staged library: {result['foreign'][:4]}",
            result,
        )
    return Check(
        "scheduling-example", PASS, f"verified against the staged library ({result['loaded']} modules)", result
    )


def check_library_inventory(ctx) -> Check:
    """The inventory script imports ``scheduling_capabilities`` from the staged library and writes its file."""
    out = ctx.work / "library_inventory.md"
    proc, result = _run_skill_script(ctx, "scheduling-api", "library_inventory.py", ["--output", str(out)])
    if result is None or result["code"] != 0:
        return Check("library-inventory", FAIL, f"inventory did not run: {_tail(proc)}")
    if result["foreign"]:
        return Check("library-inventory", FAIL,
                     f"merlin modules came from outside the staged library: {result['foreign'][:4]}", result)
    text = out.read_text(encoding="utf-8") if out.is_file() else ""
    rows = sum(1 for line in text.splitlines() if line.startswith("| capability |"))
    if "capabilities: scheduling_capabilities(" not in text or not rows or "## Per-family library passes" not in text:
        return Check("library-inventory", FAIL, f"inventory lacks the staged capability catalog or the family "
                     f"table: {text[:300]!r}", result)
    return Check("library-inventory", PASS,
                 f"{rows} capabilities from the staged library ({result['loaded']} modules) and a family table", result)


def _stdout_json(proc):
    start = proc.stdout.find("{")
    try:
        return json.JSONDecoder().raw_decode(proc.stdout[start:])[0] if start >= 0 else None
    except ValueError:
        return None


def _shape_choice_inputs(root: Path) -> list[str]:
    """A minimal knob file, three cases, a neutral facts body and results for the choice-table script: the
    winner changes exactly where the output stops fitting the accumulator."""
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "knobs.json": {"schema": "merlin.tuning_knobs.v1", "knobs": {"plan": {"values": ["a", "b"]}}},
        "cases.json": [{"case": "wide", "m": 4, "k": 8, "n": 32}, {"case": "tall", "m": 32, "k": 8, "n": 4},
                       {"case": "small", "m": 4, "k": 8, "n": 8}],
        "facts.json": {"arrays": [{"name": "array", "rows": 4, "cols": 4}],
                       "memories": [{"name": "operand_sram", "banks": 1, "depth": 64, "bytes": 256},
                                    {"name": "acc_sram", "banks": 1, "depth": 16, "bytes": 256}],
                       "datapaths": [{"name": "operand_sram", "dtype": "i8"}, {"name": "acc_sram", "dtype": "i32"}]},
        "results.json": {"rows": [{"case": c, "settings": {"plan": p}, "cycles": cycles, "domain": "acceptance"}
                                  for c, p, cycles in (("wide", "a", 20), ("wide", "b", 10), ("tall", "a", 30),
                                                       ("tall", "b", 15), ("small", "a", 5), ("small", "b", 9))]},
    }
    for name, value in files.items():
        (root / name).write_text(json.dumps(value))
    return ["--knobs", str(root / "knobs.json"), "--cases", str(root / "cases.json"), "--results",
            str(root / "results.json"), "--facts", str(root / "facts.json"), "--json"]


#: The plans capacity_pipeline_example.py realizes by default (its PIPELINES).
CAPACITY_PLANS = ("compact", "retained", "segmented", "ready")


def check_capacity_schedule_example(ctx) -> Check:
    """The shape-schedule-selection skill: its capacity-derived dense example verifies every plan and
    its choice table ranks a held-out class, both against the staged library only."""
    name = "capacity-schedule-example"
    proc, result = _run_skill_script(ctx, "shape-schedule-selection", "capacity_pipeline_example.py", [])
    if result is None or result["code"] != 0:
        return Check(name, FAIL, f"example did not verify: {_tail(proc)}")
    out = _stdout_json(proc) or {}
    plans = out.get("pipelines") or {}
    detail = {"example": result, "facts": out.get("facts"), "shape": out.get("shape"),
              "blocks": {k: v.get("block") for k, v in plans.items()}}
    if set(plans) != set(CAPACITY_PLANS) or not plans["retained"].get("retained_pairs") or not (
            plans["segmented"].get("disjoint_readout_epochs")):
        return Check(name, FAIL, f"example did not realize every plan: {sorted(plans)}", detail)
    argv = _shape_choice_inputs(ctx.work / "shape-choice")
    table_proc, table = _run_skill_script(ctx, "shape-schedule-selection", "shape_choice_table.py", argv)
    report = _stdout_json(table_proc) or {}
    separating = report.get("separating_predicates", ())
    guards = [s.get("guards", {}).get("when_true", {}).get("predicate") for s in separating]
    detail.update(table=table, classes=[row.get("class") for row in report.get("classes", ())], guards=guards)
    if table is None or table["code"] != 0 or detail["classes"] != ["long-m", "long-n"]:
        return Check(name, FAIL, f"choice table did not rank the held-out classes: {_tail(table_proc)}", detail)
    if {"resource": "accumulator_output_copies", "at_least": 1} not in guards:
        return Check(name, FAIL, "choice table did not derive the accumulator-fit ResourcePredicate guard", detail)
    foreign = sorted({*result["foreign"], *table["foreign"]})
    if foreign:
        return Check(name, FAIL, f"merlin modules came from outside the staged library: {foreign[:4]}", detail)
    return Check(name, PASS, f"{len(plans)} plans from {out.get('facts')} capacities and a choice table with its capacity "
                 f"guard verified "
                 f"against the staged library ({result['loaded']} modules)", detail)


def check_heavy_pass_recipes(ctx) -> Check:
    """The heavy-pass-recipes skill: each recipe verifies against the staged library (see the module)."""
    from .heavy_recipe_acceptance import check_heavy_pass_recipes as check

    return check(ctx)


def check_pass_pipeline(ctx) -> Check:
    orders = _skill_orders(ctx)
    detail = {"documented_orders": orders}
    if not orders:
        return Check("pass-pipeline", FAIL, "the skill documents no concrete --check order", detail)
    failures, results = {}, {}
    for entry, order in orders:
        argv = ["--check", entry, *(["--order", ",".join(order)] if order else [])]
        proc, result = _run_skill_script(ctx, "compose-pass-pipeline", "pass_specs.py", argv)
        verdict = _stdout_json(proc)
        results[entry + (":" + ",".join(order) if order else ":source-order")] = verdict
        if result is None or result["code"] != 0 or result["foreign"]:
            failures[entry] = (verdict or {}).get("reason") or (result or {}).get("foreign") or _tail(proc, 400)
    detail.update(results=results, failures=failures)
    if failures:
        return Check("pass-pipeline", FAIL, f"documented order rejected: {failures}", detail)
    return Check("pass-pipeline", PASS, f"{len(results)} documented order(s) valid against the staged library", detail)


def _skill_orders(ctx) -> list[tuple[str, list[str]]]:
    """``--check ENTRY [--order a,b,c]`` documented in the staged skill, with concrete pass names."""
    root = ctx.ws / ctx.staged["skill_roots"]["codex"] / "compose-pass-pipeline"
    text = "\n".join(p.read_text() for p in sorted(root.rglob("*")) if p.suffix in (".md", ".py"))
    found = []
    for line in text.replace("`", "\n").splitlines():
        words = line.split()
        if "--check" not in words:
            continue
        i = words.index("--check")
        if i + 1 >= len(words) or words[i + 1].startswith("-") or not words[i + 1].isidentifier():
            continue
        entry = words[i + 1]
        order = words[words.index("--order") + 1].split(",") if "--order" in words[:-1] else []
        if order and not all(len(name) > 1 for name in order):  # a,b,c is a placeholder, not an order
            order = []
        if (entry, order) not in found:
            found.append((entry, order))
    # An entry documented only with placeholders is checked in its declared source order.
    concrete = {entry for entry, order in found if order}
    return [(entry, order) for entry, order in found if order or entry not in concrete]


# ---------------------------------------------------------------------------------------------------
# g. Tuning knobs
# ---------------------------------------------------------------------------------------------------

_KNOB_DECLARE = r"""
from merlin.compile import tuning_knobs as K
submission = sys.argv[1]
K.declare(submission, {"acceptance_choice": K.Knob(values=("off", "on"), default="off")})
_emit({"value": K.knob("acceptance_choice", default="off", values=("off", "on")),
       "root": str(K.current_root()), "module": K.__file__})
"""

_KNOB_APPLY = r"""
from merlin_experiments.phase1.brokers import selfcheck_tuning as T
submission = sys.argv[1]
applied = T.apply_request(submission, {T.REQUEST_KEY: {"acceptance_choice": "on"}}, digest=lambda root: "acceptance")
_emit({"applied": applied.get(T.APPLIED_KEY)})
"""

_KNOB_READ = r"""
from merlin.compile import tuning_knobs as K
_emit({"value": K.knob("acceptance_choice", default="off", values=("off", "on")), "module": K.__file__})
"""


def check_tuning(ctx) -> Check:
    copy = ctx.work / "tuning-copy" / "submission"
    shutil.copytree(ctx.ws / "submission", copy, symlinks=True)
    library = [copy / "compiler_library"]
    proc, declared = run_child(_KNOB_DECLARE, library, ctx.staged["refused"], copy, cwd=copy.parent)
    if declared is None:
        return Check("tuning-knobs", FAIL, f"declare/read failed: {_tail(proc)}")
    proc, applied = run_child(_KNOB_APPLY, operator_roots(ctx.framework), (), copy, cwd=ctx.work)
    if applied is None:
        return Check("tuning-knobs", FAIL, f"selfcheck_tuning.apply_request failed: {_tail(proc)}")
    proc, read = run_child(_KNOB_READ, library, ctx.staged["refused"], cwd=copy.parent)
    detail = {"declared": declared, "applied": applied, "read": read}
    if read is None:
        return Check("tuning-knobs", FAIL, f"override read failed: {_tail(proc)}", detail)
    if declared["value"] != "off" or read["value"] != "on":
        return Check("tuning-knobs", FAIL, f"default {declared['value']!r}, override {read['value']!r}", detail)
    if not str(read["module"]).startswith(str(copy)):
        return Check("tuning-knobs", FAIL, f"tuning_knobs came from {read['module']}", detail)
    return Check(
        "tuning-knobs", PASS, "default 'off' read; selfcheck override 'on' read by the copy's compiler", detail
    )


# ---------------------------------------------------------------------------------------------------
# h. Large results; i. Codex config; j. Task disclosure
# ---------------------------------------------------------------------------------------------------


def _pytest(ctx, test_file: Path, selector: str, timeout: int = 900):
    env = {"PYTHONPATH": os.pathsep.join(str(r) for r in operator_roots(ctx.framework)), "TMPDIR": str(ctx.work)}
    for key in ("MERLIN_OUT_ROOT", "MERLIN_BUNDLE_CAS"):
        env[key] = str(ctx.work / key.lower())
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            str(test_file),
            "-k",
            selector,
            "--rootdir",
            str(ctx.framework),
        ],
        capture_output=True,
        text=True,
        cwd=str(ctx.work),
        env=_child_env(env),
        timeout=timeout,
    )


def _completion_guarded(source: Path) -> tuple[bool, str]:
    """Whether every ``*.completed(...)`` advisory observation inside ``process_completed`` sits in a ``try``."""
    import ast

    tree = ast.parse(source.read_text())
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "process_completed"]
    if not functions:
        return False, "loop_grading has no process_completed"
    for function in functions:
        guarded = {
            id(c)
            for t in ast.walk(function)
            if isinstance(t, ast.Try) and t.handlers
            for statement in t.body
            for c in ast.walk(statement)
        }
        for call in ast.walk(function):
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr == "completed"
                and id(call) not in guarded
            ):
                return False, f"line {call.lineno}: a completed() observation is not guarded"
    return True, "every completed() observation in process_completed is guarded"


def check_large_results(ctx) -> Check:
    tests = ctx.framework / "packages" / "merlin-experiments" / "tests"
    fixture = tests / "test_rolling_performance_feedback.py"
    if not fixture.is_file():
        return Check("large-results", FAIL, "the export carries no rolling-feedback fixture")
    proc = _pytest(ctx, fixture, "mebibyte")
    summary = (proc.stdout.strip().splitlines() or [""])[-1]
    detail = {"rolling_fixture": summary}
    if proc.returncode != 0 or "passed" not in summary:
        return Check("large-results", FAIL, f"a >=3 MiB result is not observed: {_tail(proc, 500)}", detail)
    grading = [
        r / "merlin_experiments" / "phase1" / "feedback" / "loop_grading.py" for r in operator_roots(ctx.framework)
    ]
    grading = next((p for p in grading if p.is_file()), None)
    if grading is None:
        return Check("large-results", FAIL, "the export carries no loop_grading", detail)
    guarded, why = _completion_guarded(grading)
    detail["completion_processing"] = why
    if not guarded:
        return Check("large-results", FAIL, f"a failed observation can fail the grade: {why}", detail)
    return Check("large-results", PASS, f"3 MiB result observed ({summary}); {why}", detail)


_CODEX_HOME = r"""
import tomllib
from pathlib import Path
from merlin_experiments.phase1.options import parse_options
from merlin_experiments.phase1.providers import codex_agent, codex_context as CTX
from merlin_experiments.phase1 import author_subagents as AS
home, ws, effort = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
options = parse_options(["--run-id", "acceptance", "--driver", "codex", "--sandbox", "bwrap",
                         "--author-helper-effort", effort])
record = CTX.selection(options)
codex_agent.prepare_codex_home(home, model="acceptance-model", effort="high", workspace=ws,
                               max_concurrent_subagents=4, helper_effort=effort, author_context=record)
parsed = tomllib.loads((home / "config.toml").read_text())
root_keys = AS.home_config(4, effort) + CTX.home_config(record)
_emit({"limit_key": CTX.LIMIT_KEY, "prompt_key": CTX.PROMPT_KEY, "helper_key": AS.HELPER_EFFORT_KEY,
       "parsed_keys": sorted(parsed), "agents": parsed.get("agents", {}),
       "has_limit": CTX.LIMIT_KEY in parsed, "has_prompt": CTX.PROMPT_KEY in parsed,
       "root_keys": root_keys})
"""


def _config_read(codex: str, home: Path, workspace: Path) -> dict:
    """The app-server's effective configuration over stdio JSON-RPC; no model request is made."""
    import select

    proc = subprocess.Popen(
        [codex, "app-server"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "CODEX_HOME": str(home)},
        cwd=workspace,
        text=True,
    )
    try:
        replies = {}
        for message in (
            {"id": 1, "method": "initialize", "params": {"clientInfo": {"name": "probe", "version": "0"}}},
            {"method": "initialized"},
            {"id": 2, "method": "config/read", "params": {"includeLayers": False, "cwd": str(workspace)}},
        ):
            proc.stdin.write(json.dumps(message) + "\n")
            proc.stdin.flush()
        deadline = time.monotonic() + 60
        while 2 not in replies and time.monotonic() < deadline:
            if select.select([proc.stdout], [], [], 1)[0]:
                line = proc.stdout.readline()
                if not line:
                    break
                reply = json.loads(line)
                replies[reply.get("id")] = reply
        return replies.get(2) or {}
    finally:
        proc.kill()
        proc.wait()


def check_codex_config(ctx) -> Check:
    codex = ctx.codex or shutil.which("codex")
    home = ctx.work / "codex-home"
    proc, written = run_child(
        _CODEX_HOME, operator_roots(ctx.framework), (), home, ctx.ws, ctx.helper_effort, cwd=ctx.work
    )
    if written is None:
        return Check("codex-config", FAIL, f"frozen config.toml not written: {_tail(proc)}")
    missing = [
        k
        for k, present in (
            ("model_auto_compact_token_limit", written["has_limit"]),
            ("compact_prompt", written["has_prompt"]),
            (f"agents.{written['helper_key']}", written["helper_key"] in written["agents"]),
        )
        if not present
    ]
    if missing:
        return Check("codex-config", FAIL, f"frozen config.toml lacks {missing}", written)
    if codex is None:
        return Check("codex-config", SKIP, "keys present; no Codex CLI to load them", written)
    # The full file's permission profile names a sandbox helper that exists only inside the launch sandbox;
    # the CLI loads the frozen root-level keys the way test_author_context does.
    probe = ctx.work / "codex-probe-home"
    probe.mkdir()
    (probe / "config.toml").write_text(
        written["root_keys"] + 'model = "acceptance-model"\nmodel_reasoning_effort = "high"\n'
    )
    reply = _config_read(codex, probe, ctx.ws)
    effective = (reply.get("result") or {}).get("config")
    detail = {**written, "codex": codex, "config_read": bool(effective)}
    if not effective:
        return Check("codex-config", FAIL, f"config/read did not answer: {str(reply)[:300]}", detail)
    agents = effective.get("agents") or {}
    loaded = {
        "model_auto_compact_token_limit": effective.get(written["limit_key"]),
        "compact_prompt": bool(effective.get(written["prompt_key"])),
        written["helper_key"]: agents.get(written["helper_key"]),
    }
    detail["effective"] = loaded
    if not all(loaded.values()):
        return Check("codex-config", FAIL, f"the CLI did not load every key: {loaded}", detail)
    if loaded[written["helper_key"]] != ctx.helper_effort:
        return Check("codex-config", FAIL, f"helper effort loaded as {loaded[written['helper_key']]!r}", detail)
    return Check("codex-config", PASS, f"offline config/read loaded {sorted(loaded)}", detail)


def check_task_disclosure(ctx) -> Check:
    task = (ctx.ws / "TASK.md").read_text()
    disclosure = ctx.staged.get("disclosure")
    detail = {"task_bytes": len(task.encode()), "task_staged": ctx.staged.get("task_staged")}
    if not ctx.staged.get("task_skills"):
        return Check("task-disclosure", FAIL, "the codex selection does not stage TASK.md section skills", detail)
    if not disclosure or not disclosure.get("skills"):
        return Check("task-disclosure", FAIL, "task_disclosure staged no section skill", detail)
    names = [row["name"] for row in disclosure["skills"]]
    detail["section_skills"] = names
    # Every long TASK.md section is staged; a budget section is no exception (derived from the headings).
    budget = [n for n in names if "budget" in n]
    headed = [line for line in task.splitlines() if line.startswith("## ") and "budget" in line.lower()]
    detail.update(budget_sections=budget, budget_headings=headed)
    for row in disclosure["skills"]:
        if not (ctx.ws / row["path"]).is_file():
            return Check("task-disclosure", FAIL, f"recorded section skill missing: {row['path']}", detail)
    proc, prompt = run_child(
        r"""
from merlin_experiments.phase1.providers import codex_context as CTX
from merlin_experiments.phase1.options import parse_options
record = CTX.selection(parse_options(["--run-id", "acceptance", "--driver", "codex", "--sandbox", "bwrap"]))
_emit({"round": CTX.round_note(record, 16), "continuation": CTX.continuation_note(record)})
""",
        operator_roots(ctx.framework),
        (),
        cwd=ctx.work,
    )
    if prompt is None:
        return Check("task-disclosure", FAIL, f"core prompt notes unavailable: {_tail(proc)}", detail)
    detail["round_note"] = prompt["round"][-1200:]
    wanted = {"section skills": "task-*", "feedback files": "qa/"}
    absent = [label for label, token in wanted.items() if token not in prompt["round"]]
    if absent:
        return Check("task-disclosure", FAIL, f"the round prompt does not name {absent}", detail)
    if headed and not budget:
        return Check("task-disclosure", FAIL, f"budget section(s) {headed} did not become section skills", detail)
    feedback = ctx.ws / ctx.staged["skill_roots"]["codex"] / ctx.staged["feedback_skill"] / "SKILL.md"
    if not feedback.is_file():
        return Check("task-disclosure", FAIL, f"feedback-file guide {ctx.staged['feedback_skill']} not staged", detail)
    return Check(
        "task-disclosure",
        PASS,
        f"{len(names)} section skills ({len(budget)} budget); round prompt "
        f"names task-* skills and qa/ feedback; {ctx.staged['feedback_skill']} staged",
        detail,
    )


# ---------------------------------------------------------------------------------------------------
# Tooling readiness: the launch preflight's own verdict on the staged bundle
# ---------------------------------------------------------------------------------------------------

#: What ``session._prepare`` prints, and refuses the launch with, when the readiness verdict is ``no_go``.
NO_GO = "NO_GO: selected authoring tools are not ready"


def assess_staged_bundle(descriptor, repo, manifest, work, *, library=None, with_tools=(), without_tools=()) -> dict:
    """The launch preflight's tooling-readiness assessment of one sealed bundle, operator side.

    Same owners and order as ``session._prepare`` under ``--sandbox bwrap``: the bwrap transport
    freezes the bundle's grants (``workspace_transport.assemble_workspace``), the frozen RTL facts are
    selected, the tool set is resolved from the bundle directory, and
    ``tooling_readiness.assess`` runs the registry grant checks, the frozen RTL check compilation and
    the selected-sandbox authoring probe (``_sandbox_probe`` with the bundle-derived flags, inside the
    argv ``bwrap.full_argv`` composes). The probe workspace never holds a candidate. Self-contained so
    the suite can ship this exact source to the export's own interpreter.
    """
    from pathlib import Path

    import yaml

    from merlin.common.bundle_snapshot import verified_snapshot_scope
    from merlin.targetgen.compiler_library import scoped_compiler_library
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.target_experiment import load_target_experiment
    from merlin_experiments.phase1 import task_staging as TS
    from merlin_experiments.phase1 import tooling_readiness as TRD
    from merlin_experiments.phase1 import workspace_transport as WT
    from merlin_experiments.phase1.context import load_context
    from merlin_experiments.phase1.frozen_facts import select as select_frozen_facts

    manifest = Path(manifest)
    context = load_context(Path(descriptor), repo=Path(repo))
    te = load_target_experiment(context.descriptor)
    bundle = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    tools = tuple(
        TS.resolved_tools(
            bundle.get("bundle_id", manifest.parent.name),
            bundle_dir=manifest.parent,
            add_tools=tuple(with_tools),
            drop_tools=tuple(without_tools),
        )
    )
    ws = Path(work) / "workspace"
    WT.assemble_workspace(bundle, ws, context=context)
    with verified_snapshot_scope(ws, bundle, repo=context.repo):
        select_frozen_facts(ws, BW.frozen_selected_rtl_facts(ws, bundle, repo=context.repo))
        # The public capsules the run corpus admits: the frozen copies of the descriptor corpus's granted
        # subtrees (hidden capsules are never granted), at their mirrored place in the snapshot.
        corpus = Path(te.capsule_corpus).absolute()
        sources = [BW.resolve_grant(row["path"], context.repo).absolute() for row in bundle.get("allowed", ())]
        granted = [source for source in sources if source != corpus and source.is_relative_to(corpus)]
        public = ws
        if granted:
            [frozen] = BW.snapshot_input_paths(ws, bundle, granted[:1], repo=context.repo)
            public = Path(*frozen.parts[: len(frozen.parts) - len(granted[0].relative_to(corpus).parts)])
        with scoped_compiler_library(library):
            result = TRD.assess(context, te, ws, bundle, tools, public, without_tools=tuple(without_tools))
    return {**result, "bundle_id": bundle.get("bundle_id"), "resolved_tools": list(tools)}


def readiness_check(result: dict) -> Check:
    """Map a readiness verdict to a row; ``no_go`` carries the launch's own refusal text."""
    checks = result.get("checks") or []
    failed = [f"{row['name']}: {row['detail']}" for row in checks if not row.get("ok")]
    detail = {
        "status": result.get("status"),
        "bundle_id": result.get("bundle_id"),
        "resolved_tools": result.get("resolved_tools"),
        "checks": checks,
    }
    if result.get("status") == "no_go":
        return Check("tooling-readiness", FAIL, f"{NO_GO}: {'; '.join(failed)}"[:1500], detail)
    if result.get("status") == "skipped":
        return Check("tooling-readiness", PASS, f"skipped by the launch too: {result.get('reason')}", detail)
    if result.get("status") != "ready" or failed:
        return Check("tooling-readiness", FAIL, f"unrecognized readiness verdict {result.get('status')!r}", detail)
    names = [row["name"] for row in checks]
    grants = sum(name.startswith("grant:") for name in names)
    withheld = sum("recorded denial" in str(row.get("detail")) for row in checks)
    return Check(
        "tooling-readiness",
        PASS,
        f"ready: {grants} grant checks ({withheld} withheld by recorded denial), "
        f"{len(names) - grants} assessment checks incl. selected_sandbox_authoring="
        f"{'selected_sandbox_authoring' in names}",
        detail,
    )


def _readiness_script() -> str:
    import inspect
    import textwrap

    return (
        textwrap.dedent(inspect.getsource(assess_staged_bundle))
        + r"""
args = json.loads(sys.argv[1])
library = None
if args["library"] is not None:
    from merlin_experiments.phase2.compiler_library_state import load_selection
    library = load_selection(**args["library"])
_emit(assess_staged_bundle(args["descriptor"], args["repo"], args["manifest"], args["work"], library=library,
                           with_tools=args["with_tools"], without_tools=args["without_tools"]))
"""
    )


def check_tooling_readiness(ctx) -> Check:
    if shutil.which("bwrap") is None:
        return Check("tooling-readiness", SKIP, "bwrap is not installed (the launch preflight requires it)")
    if not ctx.staged.get("descriptor") or not ctx.staged.get("bundle"):
        return Check("tooling-readiness", SKIP, "needs --inputs (the sealed bundle is what the launch assesses)")
    descriptor = Path(ctx.staged["descriptor"])
    manifest = descriptor.parent / "input_bundles" / ctx.staged["bundle"] / "input_bundle_manifest.yaml"
    work = ctx.work / "tooling-readiness"
    work.mkdir()
    args = {
        "descriptor": str(descriptor),
        "repo": str(ctx.framework),
        "manifest": str(manifest),
        "work": str(work),
        "library": ctx.compiler_library,
        "with_tools": [],
        "without_tools": [],
    }
    # Operator side, under the caller's launch environment (the preflight inherits its MERLIN_* settings),
    # with the export's harness importable by the brokers it starts, as the launch's PYTHONPATH makes it.
    roots = operator_roots(ctx.framework)
    env = {k: v for k, v in os.environ.items() if k.startswith("MERLIN_")}
    env["PYTHONPATH"] = os.pathsep.join(str(r) for r in roots)
    if ctx.compiler_execution is not None:
        env["MERLIN_COMPILER_EXECUTION_MODE"] = ctx.compiler_execution
    proc, result = run_child(_readiness_script(), roots, (), json.dumps(args), cwd=work, timeout=1800, env=env)
    if proc.returncode != 0 or result is None:
        return Check("tooling-readiness", FAIL, f"readiness assessment did not run: {_tail(proc, 1200)}")
    return readiness_check(result)


# ---------------------------------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------------------------------


@dataclass
class Context:
    framework: Path
    inputs: Path | None
    work: Path
    staged: dict
    roots: list
    data_env: dict
    codex: str | None
    helper_effort: str
    allowed_import_failures: int
    #: The launch's ``--compiler-execution`` and ``--compiler-library-*`` selection, for tooling readiness.
    compiler_execution: str | None = None
    compiler_library: dict | None = None

    @property
    def ws(self) -> Path:
        return Path(self.staged["workspace"])


CHECKS = {
    "withheld-imports": check_withheld,
    "withheld-in-bwrap": check_withheld_sandbox,
    "staged-imports": check_imports,
    "fusion": check_fusion,
    "skills": check_skills,
    "scheduling-example": check_scheduling_example,
    "library-inventory": check_library_inventory,
    "capacity-schedule-example": check_capacity_schedule_example,
    "heavy-pass-recipes": check_heavy_pass_recipes,
    "pass-pipeline": check_pass_pipeline,
    "tuning-knobs": check_tuning,
    "large-results": check_large_results,
    "codex-config": check_codex_config,
    "task-disclosure": check_task_disclosure,
    "tooling-readiness": check_tooling_readiness,
}


def run(
    framework: Path,
    inputs: Path | None,
    *,
    work_root: Path,
    keep: bool = False,
    only=None,
    sandbox_args: Path | None = None,
    codex: str | None = None,
    helper_effort: str = "medium",
    allowed_import_failures: int = 3,
    compiler_execution: str | None = None,
    compiler_library: dict | None = None,
) -> dict:
    framework = Path(framework).resolve()
    inputs = Path(inputs).resolve() if inputs else None
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="author-acceptance-", dir=work_root))
    started = time.monotonic()
    report = {
        "schema": SCHEMA,
        "framework": str(framework),
        "inputs": str(inputs) if inputs else None,
        "work": str(work),
        "checks": [],
    }
    try:
        staged = stage(framework, inputs, work, "codex")
    except Exception as exc:  # noqa: BLE001 - staging failure is the first row
        report["checks"].append(asdict(Check("staging", FAIL, str(exc)[-1500:])))
        report["status"] = FAIL
        return report
    report["checks"].append(
        asdict(
            Check("staging", PASS, f"{staged['staged_members']} compiler members, task staged={staged['task_staged']}")
        )
    )
    report["staged"] = {k: v for k, v in staged.items() if k not in ("full_compiler_modules", "skills", "disclosure")}
    schemas = framework / "merlin" / "schemas"
    ctx = Context(
        framework,
        inputs,
        work,
        staged,
        author_roots(staged, inputs),
        {"MERLIN_SCHEMAS_DIR": str(schemas)} if schemas.is_dir() else {},
        codex,
        helper_effort,
        allowed_import_failures,
        compiler_execution,
        compiler_library,
    )
    report["author_sys_path"] = [str(r) for r in ctx.roots]
    for name, check in CHECKS.items():
        if only and name not in only:
            continue
        began = time.monotonic()
        try:
            result = check(ctx)
        except Exception as exc:  # noqa: BLE001 - one broken check never hides the others
            result = Check(name, FAIL, f"{type(exc).__name__}: {exc}"[:600])
        result.seconds = round(time.monotonic() - began, 1)
        report["checks"].append(asdict(result))
    if sandbox_args is not None:
        report["checks"].append(asdict(audit_sandbox_args(Path(sandbox_args), withheld_names(staged))))
    report["status"] = FAIL if any(c["status"] == FAIL for c in report["checks"]) else PASS
    report["seconds"] = round(time.monotonic() - started, 1)
    path = work_root / f"{work.name}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    report["report_path"] = str(path)
    if not keep:
        _remove(work)
    return report


def _remove(path: Path) -> None:
    for item in path.rglob("*"):
        if item.is_dir() and not item.is_symlink():
            item.chmod(0o755)
    shutil.rmtree(path, ignore_errors=True)


def table(report: dict) -> str:
    rows = [(c["name"], c["status"], f"{c.get('seconds', 0):.0f}s", c["cause"]) for c in report["checks"]]
    width = max(len(r[0]) for r in rows)
    lines = [f"{'check'.ljust(width)}  result  time  cause", "-" * (width + 40)]
    lines += [f"{n.ljust(width)}  {s.ljust(6)}  {t.rjust(4)}  {c}" for n, s, t, c in rows]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--framework", required=True, type=Path, help="frozen framework export root")
    ap.add_argument("--inputs", type=Path, default=None, help="sealed fresh-inputs directory (read only)")
    ap.add_argument(
        "--work-root", type=Path, default=None, help="where the work directory and JSON report go (default: $TMPDIR)"
    )
    ap.add_argument("--keep", action="store_true", help="keep the staged work directory")
    ap.add_argument("--only", default="", help="comma-separated check names (default: all)")
    ap.add_argument("--fast", action="store_true", help="only the checks that need no bwrap, CLI or full walk")
    ap.add_argument(
        "--sandbox-args",
        type=Path,
        default=None,
        help="a recorded bwrap --args payload to audit read-only for mounted withheld modules",
    )
    ap.add_argument("--codex", default=None, help="Codex CLI to load the frozen config with (default: PATH)")
    ap.add_argument("--helper-effort", default="medium")
    ap.add_argument(
        "--compiler-execution",
        default=None,
        help="the launch's --compiler-execution, for tooling readiness (default: the inherited environment)",
    )
    ap.add_argument("--compiler-library-root", default=None, help="the launch's --compiler-library-root")
    ap.add_argument("--compiler-library-contract", default=None, help="the launch's --compiler-library-contract")
    ap.add_argument(
        "--compiler-library-contract-sha256", default=None, help="the launch's --compiler-library-contract-sha256"
    )
    a = ap.parse_args(argv)
    library = {
        "root": a.compiler_library_root,
        "contract_path": a.compiler_library_contract,
        "contract_sha256": a.compiler_library_contract_sha256,
    }
    only = set(FAST) if a.fast else {n for n in a.only.split(",") if n}
    unknown = only - set(CHECKS)
    if unknown:
        ap.error(f"unknown checks: {sorted(unknown)}; choose from {sorted(CHECKS)}")
    work_root = a.work_root or Path(tempfile.gettempdir())
    report = run(
        a.framework,
        a.inputs,
        work_root=work_root,
        keep=a.keep,
        only=only or None,
        sandbox_args=a.sandbox_args,
        codex=a.codex,
        helper_effort=a.helper_effort,
        compiler_execution=a.compiler_execution,
        compiler_library=library if any(library.values()) else None,
    )
    print(table(report))
    print(
        f"\n{report['status']}  report: {report.get('report_path', '(not written)')}  work: {report['work']}"
        + ("" if a.keep else " (removed)")
    )
    return 0 if report["status"] == PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
