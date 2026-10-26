"""Reproducible fresh (unseeded) open-compiler campaign launch from one reviewed spec.

    fresh_campaign.py export    --spec SPEC   # frozen framework export of the branch HEAD
    fresh_campaign.py inputs    --spec SPEC   # author library, full compiler, references, corpus
    fresh_campaign.py seal      --spec SPEC   # corpus inspect + reviewed seal
    fresh_campaign.py plan      --spec SPEC   # policy, pins and proposed-launch.json (no launch)
    fresh_campaign.py preflight --spec SPEC [--run] [--execute]
    fresh_campaign.py launch    --spec SPEC --confirm
    fresh_campaign.py check     --spec SPEC [--watch | --until HHMM] [--console]

Every path, id and CPU list comes from the spec (``${NAME}`` variables, overridable with
``--var NAME=VALUE``) or its run section (``--run-id``, ``--unit``, ``--cpus``). The campaign
settings derived from an earlier recorded launch are spelled out in the spec's ``plan`` section,
so a reviewer reads the exact argv and environment instead of a rewrite rule. Preparation and
reservation reuse the sibling launch modules of the export (prepare_author_library,
prepare_source_release, prepare_author_launch, resource_reservation) and
``merlin_experiments.phase1.full_compiler_workspace``. Only ``launch`` starts anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

SCHEMA = "merlin.phase1.fresh_campaign_spec.v1"
HERE = Path(__file__).resolve().parent
_NAME_START = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ_")
_NAME_REST = _NAME_START | set("0123456789")
# Run-section values exposed as variables; the spec may reference but not redefine them.
# HOME and USER are the operator's own account, taken from the invoking environment.
# RUN_TAG is the run id's last dash-separated word (fresh2 for ...-20261026-fresh2).
BUILTINS = (
    "RUN_ID",
    "RUN_TAG",
    "UNIT",
    "CPUS",
    "PREPARATION_CPUS",
    "EXPORT_ID",
    "PREFLIGHT_RUN_ID",
    "HOME",
    "USER",
)


class SpecError(ValueError):
    pass


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_name(text: str) -> bool:
    return bool(text) and text[0] in _NAME_START and all(char in _NAME_REST for char in text[1:])


def expand(text: str, lookup) -> str:
    """Replace each ``${NAME}`` through ``lookup``; anything else is literal."""
    pieces, cursor = [], 0
    while (start := text.find("${", cursor)) >= 0:
        end = text.find("}", start + 2)
        name = text[start + 2 : end] if end >= 0 else ""
        if not _is_name(name):
            raise SpecError(f"malformed variable reference in {text!r}")
        pieces += [text[cursor:start], lookup(name)]
        cursor = end + 1
    pieces.append(text[cursor:])
    return "".join(pieces)


def load_module(name: str, directory: Path):
    """Load a sibling operator module from the selected (export) tree, as the launch does."""
    source = Path(directory) / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"fresh_campaign_{name}", source)
    if spec is None or spec.loader is None:
        raise SpecError(f"launch module is unavailable: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def cpu_list(text: str) -> list[int]:
    cpus = [int(part) for part in str(text).split(",") if part.strip()]
    if not cpus or len(set(cpus)) != len(cpus):
        raise SpecError(f"CPU list must name distinct CPUs: {text!r}")
    return cpus


def merged(base, override):
    """``override`` on top of ``base``: mappings merge key by key, anything else replaces."""
    if isinstance(base, dict) and isinstance(override, dict):
        result = dict(base)
        for key, value in override.items():
            result[key] = merged(base[key], value) if key in base else value
        return result
    return override


def load_spec(path: Path, seen=()) -> dict:
    """A spec, with ``extends`` (a path relative to it) resolved by :func:`merged`."""
    if path in seen:
        raise SpecError(f"spec extends itself: {path}")
    document = yaml.safe_load(path.read_text())
    if not isinstance(document, dict):
        raise SpecError(f"{path} is not a mapping")
    parent = document.pop("extends", None)
    if parent is None:
        return document
    return merged(load_spec((path.parent / parent).resolve(), (*seen, path)), document)


def spec_chain(path: Path) -> list[Path]:
    """The spec file and every file it extends, child first."""
    chain = [path]
    while (parent := yaml.safe_load(chain[-1].read_text()).get("extends")) is not None:
        chain.append((chain[-1].parent / parent).resolve())
    return chain


def with_defaults(value: dict, defaults: dict) -> dict:
    """Fill absent keys from ``defaults`` recursively; present values always win."""
    for key, default in defaults.items():
        if isinstance(default, dict) and isinstance(value.get(key), dict):
            with_defaults(value[key], default)
        else:
            value.setdefault(key, json.loads(json.dumps(default)))
    return value


class Context:
    """One spec with its run section and variables resolved for this invocation."""

    def __init__(self, spec_path: str | Path, *, overrides=(), run_id=None, unit=None, cpus=None, export_id=None):
        self.spec_path = Path(spec_path).resolve()
        self.spec = load_spec(self.spec_path)
        self.spec_sources = spec_chain(self.spec_path)
        if not isinstance(self.spec, dict) or self.spec.get("schema") != SCHEMA:
            raise SpecError(f"{self.spec_path} is not a {SCHEMA} spec")
        run = dict(self.spec.get("run") or {})
        self.run_id = run_id or run.get("run_id")
        if not self.run_id:
            raise SpecError("this spec selects no run id: pass --run-id")
        self.unit = unit or run.get("unit")
        self.cpus = cpus or run["cpus"]
        self.preparation_cpus = run.get("preparation_cpus") or self.cpus
        cpu_list(self.cpus), cpu_list(self.preparation_cpus)
        self.run = run
        declared = dict(self.spec.get("variables") or {})
        clash = sorted(set(declared) & set(BUILTINS))
        if clash:
            raise SpecError(f"spec may not redefine built-in variables {clash}")
        for item in overrides:
            name, sep, value = item.partition("=")
            if not sep or name not in declared:
                raise SpecError(f"--var must name a declared variable as NAME=VALUE: {item!r}")
            declared[name] = value
        self._raw = declared
        self._export_id = export_id
        self._resolved: dict[str, str] = {
            "RUN_ID": self.run_id,
            "RUN_TAG": self.run_id.rsplit("-", 1)[-1],
            "PREFLIGHT_RUN_ID": self.run_id + "-preflight",
            "UNIT": self.unit or "",
            "CPUS": self.cpus,
            "PREPARATION_CPUS": self.preparation_cpus,
        }
        for name in ("HOME", "USER"):
            if os.environ.get(name):
                self._resolved[name] = os.environ[name]
        self.computed: dict[str, str] = {}

    def lookup(self, name: str, active=()) -> str:
        if name in self.computed:
            return self.computed[name]
        if name == "EXPORT_ID" and name not in self._resolved:
            self._resolved[name] = self._export_id or git(Path(self.lookup("REPO")), "rev-parse", "HEAD")[:12]
        if name in self._resolved:
            return self._resolved[name]
        if name not in self._raw:
            raise SpecError(f"undefined variable ${{{name}}}")
        if name in active:
            raise SpecError(f"variable cycle: {' -> '.join([*active, name])}")
        value = expand(str(self._raw[name]), lambda inner: self.lookup(inner, (*active, name)))
        self._resolved[name] = os.path.normpath(value) if value.startswith("/") else value
        return self._resolved[name]

    def x(self, text) -> str:
        return expand(str(text), self.lookup)

    def x_tree(self, value):
        """Expand every string inside a nested spec value."""
        if isinstance(value, str):
            return self.x(value)
        if isinstance(value, list):
            return [self.x_tree(item) for item in value]
        if isinstance(value, dict):
            return {key: self.x_tree(item) for key, item in value.items()}
        return value

    def path(self, text) -> Path:
        return Path(self.x(text))

    def section(self, name: str) -> dict:
        value = self.spec.get(name)
        if not isinstance(value, dict):
            raise SpecError(f"spec has no {name!r} section")
        return value


# --------------------------------------------------------------------------- export


def cmd_export(ctx: Context, args) -> int:
    section = ctx.section("export")
    repo = ctx.path("${REPO}")
    head = git(repo, "rev-parse", "HEAD")
    if ctx._export_id and ctx._export_id != head[:12]:
        raise SpecError("--export-id must be the current branch commit for a new export")
    ctx._resolved["EXPORT_ID"] = head[:12]
    if git(repo, "status", "--porcelain", "--untracked-files=no"):
        raise SpecError("uncommitted edits: commit before exporting")
    framework = ctx.path("${FRAMEWORK}")
    if framework.exists():
        raise SpecError(f"export exists: {framework}")
    framework.mkdir(parents=True)
    archive = subprocess.Popen(["git", "-C", str(repo), "archive", "HEAD"], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(framework)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    if archive.wait() != 0:
        raise SpecError("git archive failed")
    # Repository-local configuration that git archive cannot carry.
    source = ctx.path(section["config_source"])
    for item in section.get("copies") or ():
        relative = item["path"]
        origin, target = source / relative, framework / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if origin.is_dir():
            shutil.copytree(origin, target, symlinks=True)
        else:
            shutil.copy2(origin, target)
        if item.get("mode") is not None:
            os.chmod(target, int(str(item["mode"]), 8))
    print(json.dumps({"export": str(framework), "revision": head}))
    return 0


# --------------------------------------------------------------------------- inputs / seal


def preparation_environment(ctx: Context) -> dict[str, str]:
    """The campaign runtime with the preparation-time overrides on top of the caller's environment."""
    environment = dict(os.environ)
    template = dict(ctx.section("plan")["environment"])
    template.update(ctx.section("inputs").get("environment") or {})
    environment.update({key: ctx.x(value) for key, value in template.items()})
    return environment


def _prepared_runtime(ctx: Context) -> tuple[dict[str, str], Path]:
    environment = preparation_environment(ctx)
    Path(environment["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    framework = ctx.path("${FRAMEWORK}")
    load_module("prepare_author_launch", ctx.path(ctx.section("plan")["launch_modules"])).prepare_slot_roots(
        environment
    )
    return environment, framework


def _forward(ctx: Context, args) -> list[str]:
    forwarded = ["--spec", str(ctx.spec_path), "--run-id", ctx.run_id, "--cpus", ctx.cpus]
    forwarded += ["--export-id", ctx.lookup("EXPORT_ID")]
    for item in args.var or ():
        forwarded += ["--var", item]
    return forwarded


def cmd_inputs(ctx: Context, args) -> int:
    section = ctx.section("inputs")
    environment, framework = _prepared_runtime(ctx)
    command = ["taskset", "-c", ctx.preparation_cpus, "nice", "-n", str(section.get("nice", 5))]
    command += [ctx.x("${PYTHON}"), "-B", str(Path(__file__).resolve()), "_inputs", *_forward(ctx, args)]
    return subprocess.run(command, cwd=framework, env=environment).returncode


def forbidden_references(members, prefixes) -> list[str]:
    return [m["path"] for m in members if any(m["path"].startswith(prefix) for prefix in prefixes)]


def reference_members(document):
    """Publication-reviewed v2 destinations, or legacy historical manifest entries."""
    if document.get("schema") == "operator.allowed_reference_sources.v2":
        return [{"path": origin["name"] + "/" + member["destination"]}
                for origin in document["origins"] for member in origin["files"]]
    return document["members"]


def stage_launch_references(manifest, destination, prefixes):
    document = json.loads(manifest.read_text())
    forbidden = forbidden_references(reference_members(document), prefixes)
    if forbidden:
        raise SpecError(f"public references carry forbidden members: {forbidden[:3]}")
    if document.get("schema") == "operator.allowed_reference_sources.v2":
        from merlin_experiments.corpus.public_references import stage
        stage(manifest, destination)
        # The original reviewed origin containers are needed for release staging.
        return manifest
    for member in document["members"]:
        target = destination / member["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(manifest.parent / member["path"], target)
    staged = destination / "reference-manifest.json"
    staged.write_text(json.dumps(document, indent=1) + "\n")
    return staged


def extra_source_selections(ctx, section):
    selected = {}
    for key in ("public_program_suite", "host_capability_selection"):
        pin = section.get(key)
        if pin is None:
            continue
        path = ctx.path(pin["path"])
        if sha256(path) != pin["sha256"]:
            raise SpecError(f"{key} selection changed: {path}")
        selected[key] = path if key == "public_program_suite" else (path, pin["sha256"])
    return selected


def cmd_inputs_stage(ctx: Context, args) -> int:
    """Inside the preparation runtime: produce the four reviewed inputs from the export's own code."""
    section = ctx.section("inputs")
    framework, out = ctx.path("${FRAMEWORK}"), ctx.path(section["output"])
    revision = git(ctx.path("${REPO}"), "rev-parse", f"{ctx.lookup('EXPORT_ID')}^{{commit}}")
    if framework.name != revision[:12]:
        raise SpecError("export must be named by its source commit")
    modules = ctx.path(ctx.section("plan")["launch_modules"])
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    library = load_module("prepare_author_library", modules)
    base = ctx.path(section["previous_library"])
    previous = json.loads((base / "generic-library-selection.json").read_bytes())
    report = library.prepare(
        repo=framework,
        source_revision=revision,
        previous_root=Path(previous["root"]),
        previous_contract=base / "generic-library-contract.json",
        previous_sha256=previous["contract_sha256"],
        output=out / "author-library",
        library_version=int(section["library_version"]),
        reviewed_module=tuple(section.get("reviewed_modules") or ()),
    )
    contract = report["contract_sha256"]
    print(json.dumps({"stage": "author_library", "contract": contract, "members": report["members"]}), flush=True)

    from merlin_experiments.phase1.full_compiler_workspace import prepare as prepare_full

    full = prepare_full(framework, out / "full-compiler", review_id=ctx.x(section["full_compiler_review_id"]))
    full.verify()
    (out / "full-compiler-selection.json").write_text(json.dumps(full.record(), indent=2) + "\n")
    print(json.dumps({"stage": "full_compiler", "members": len(full.members), "manifest": full.manifest_sha256}))

    # Public references: the pinned base members only; a forbidden prefix (private measured kernels)
    # refuses the whole preparation before anything is copied.
    source_refs, refs = ctx.path(section["public_references"]), out / "public-references"
    reference_manifest = stage_launch_references(source_refs / "reference-manifest.json", refs,
        section.get("forbidden_reference_prefixes") or ())

    source = load_module("prepare_source_release", modules)
    corpus, budget = ctx.path(section["corpus"]), int(section["logical_storage_budget_bytes"])
    result = source.prepare(
        corpus,
        out / "source-release",
        out / "author-library/base-sdk",
        public_reference_manifest=reference_manifest,
        **extra_source_selections(ctx, section),
        logical_storage_budget_bytes=budget,
        common_substrate=section["common_substrate"],
    )
    selection = {
        "schema": "operator.fresh_run_inputs.v1",
        "source_revision": revision,
        "framework_export": str(framework),
        "author_library_selection": str(out / "author-library/compiler-library-selection.json"),
        "author_library_contract_sha256": contract,
        "library_inventory_version": int(section["library_version"]),
        "common_substrate": section["common_substrate"],
        "logical_storage_budget_bytes": budget,
        "full_compiler_selection": str(out / "full-compiler-selection.json"),
        "full_compiler": full.record(),
        "source_release": str(out / "source-release"),
        "previous_release": str(corpus),
        "requires_review_and_sealing": True,
        "wall_s": time.monotonic() - started,
    }
    (out / "inputs-selection.json").write_text(json.dumps(selection, indent=2, default=str) + "\n")
    print(json.dumps({"stage": "done", "release": str(out / "source-release"), "result": str(result)[:300]}))
    return 0


def cmd_seal(ctx: Context, args) -> int:
    section, inputs = ctx.section("seal"), ctx.path(ctx.section("inputs")["output"])
    environment, framework = _prepared_runtime(ctx)
    release = inputs / "source-release"
    corpus = ["taskset", "-c", ctx.preparation_cpus, ctx.x("${PYTHON}"), "-B", "-m", "merlin_experiments", "corpus"]
    inspected = subprocess.run(
        [*corpus, "inspect", str(release)], cwd=framework, env=environment, check=True, capture_output=True, text=True
    ).stdout
    (inputs / "inspect.json").write_text(inspected)
    digest = json.loads(inspected)["review_digest"]
    print(json.dumps({"inspect": str(inputs / "inspect.json"), "review_digest": digest}))
    if args.inspect_only:
        return 0
    reviewer = args.reviewed_by or section.get("reviewed_by")
    note = args.review_note or section.get("review_note")
    if not (reviewer and note):
        raise SpecError("sealing needs a reviewer and a review note (--reviewed-by, --review-note)")
    note = ctx.x(note)
    sealed = subprocess.run(
        [*corpus, "seal", str(release), "--expected-digest", digest, "--reviewed-by", reviewer, "--review-note", note],
        cwd=framework,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    (inputs / "seal-result.json").write_text(sealed)
    print(json.dumps({"sealed": str(release), "result": str(inputs / "seal-result.json")}))
    return 0


# --------------------------------------------------------------------------- plan


def edited_policy(base: dict, *, max_wall_s, final_model_s, workers, excluded, validate, defaults=None) -> dict:
    """The base policy without a reviewed compiler origin, on the selected clock."""
    policy = json.loads(json.dumps(base))
    policy.pop("compiler_origin", None)
    policy["duration_s"] = max(3600, int(max_wall_s))
    policy["final_model_s"] = min(int(final_model_s), policy["duration_s"] - 1800)
    policy["drain_s"] = min(policy.get("drain_s", 900), policy["final_model_s"])
    gsim = policy.setdefault("gsim", {})
    # The gSIM worker budget follows the clock: take the harness's own default for this duration.
    gsim.pop("worker_budget_s", None)
    gsim["worker_budget_s"] = validate(policy)["gsim"]["worker_budget_s"]
    # Settings a fresh launch adds unless the base policy already selects them.
    with_defaults(policy, defaults or {})
    if workers:
        for name in ("gsim", "gsim_correctness"):
            if isinstance(policy.get(name), dict) and "max_jobs" in policy[name]:
                policy[name]["max_jobs"] = min(policy[name]["max_jobs"], int(workers))
    if excluded is not None:
        if not (isinstance(excluded, list) and all(isinstance(name, str) for name in excluded)):
            raise SpecError("performance scope must list case ids under 'excluded'")
        policy["performance_scope"] = {"excluded": sorted(set(excluded))}
    return policy


def _validate_policy(policy: dict) -> dict:
    from merlin_experiments.phase1.joint_control import policy as load_policy

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(policy, handle)
    try:
        return load_policy(handle.name)
    finally:
        os.unlink(handle.name)


def restore_library(selection_path: Path):
    from merlin_experiments.phase2.compiler_library_state import restore_selection

    return restore_selection(json.loads(Path(selection_path).read_bytes()))


def set_wrapper_option(argv: list[str], flag: str, value: str) -> None:
    """Set an option of the reservation wrapper (before its ``--``), never of the campaign."""
    wrapper = argv[: argv.index("--")]
    argv[wrapper.index(flag) + 1] = value


WORKER_KEYS = (
    "MERLIN_DEVICE_BUILD_JOBS",
    "MERLIN_GRADE_WORKERS",
    "MERLIN_GSIM_MAX_SLOTS",
    "MERLIN_SPIKE_MAX_SLOTS",
    "MERLIN_PRIVATE_EXECUTION_WORKERS",
    "MERLIN_SELFCHECK_ORDINARY_JOBS",
    "MERLIN_SELFCHECK_VARIANT_JOBS",
)


def build_plan(ctx: Context, *, helper_effort=None, write=True) -> dict:
    plan, run = ctx.section("plan"), ctx.run
    output = ctx.path("${ARTIFACTS}")
    output.mkdir(parents=True, exist_ok=True)
    workers = run.get("workers")

    # 1. Policy.
    base_pin = plan["policy"]["base"]
    base_path = ctx.path(base_pin["path"])
    if sha256(base_path) != base_pin["sha256"]:
        raise SpecError(f"base policy changed: {base_path}")
    scope = plan["policy"].get("performance_scope")
    excluded = None
    if scope:
        excluded = json.loads(ctx.path(scope).read_text()).get("excluded") or []
    policy = edited_policy(
        json.loads(base_path.read_text()),
        max_wall_s=run["max_wall_s"],
        final_model_s=run["final_model_s"],
        workers=workers,
        excluded=excluded,
        validate=_validate_policy,
        defaults=plan["policy"].get("defaults"),
    )
    if "reviewed_compiler_seed" in policy or "compiler_origin" in policy:
        raise SpecError("a fresh launch must not carry a compiler seed")
    policy_path = output / "policy.json"
    policy_path.write_text(json.dumps(policy, indent=2) + "\n")
    _validate_policy(policy)

    # 2. Guidance and timing selection, copied beside the launch so the run pins exactly these bytes.
    guidance = output / "author-guidance.md"
    if ctx.path(plan["guidance"]).resolve() != guidance.resolve():
        shutil.copyfile(ctx.path(plan["guidance"]), guidance)
    selection = output / "performance-selection.json"
    shutil.copyfile(ctx.path(plan["timing_selection"]), selection)
    timing = {"path": str(selection), "sha256": sha256(selection)}

    # 3. Inputs: full compiler manifest, author library selection and public references.
    manifest = ctx.path(plan["full_compiler_manifest"])
    library = restore_library(ctx.path(plan["library_selection"]))
    contract = output / "selected-library-contract.json"
    contract.write_text(json.dumps(library.contract.record(), sort_keys=True, indent=2) + "\n")
    references = ctx.path(plan["public_reference_manifest"])
    prefixes = ctx.section("inputs").get("forbidden_reference_prefixes") or ()
    forbidden = forbidden_references(reference_members(json.loads(references.read_text())), prefixes)
    if forbidden:
        raise SpecError(f"public references carry forbidden members: {forbidden[:3]}")

    cap = (lambda value: str(min(int(value), int(workers)))) if workers else str
    ctx.computed.update(
        {
            "DURATION_S": str(policy["duration_s"]),
            "RATE_LIMIT_WAITS": str(run["rate_limit_waits"]),
            "SIM_MAX_JOBS": cap(run["sim_max_jobs"]),
            "LIBRARY_ROOT": str(library.root),
            "LIBRARY_CONTRACT_SHA256": library.contract.sha256,
            "FULL_COMPILER_MANIFEST_SHA256": sha256(manifest),
            "PERFORMANCE_SELECTION_PIN": json.dumps(timing),
        }
    )

    # 4. Campaign argv and environment, exactly as the spec spells them.
    argv = [ctx.x(item) for item in plan["argv"]]
    effort = helper_effort if helper_effort is not None else run.get("author_helper_effort")
    if effort and "--author-helper-effort" not in argv:
        argv += ["--author-helper-effort", str(effort)]
    environment = {key: ctx.x(value) for key, value in plan["environment"].items()}
    if workers:
        environment = {k: cap(v) if k in WORKER_KEYS else v for k, v in environment.items()}
    for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"):
        if environment.get(key) != ctx.cpus:
            raise SpecError(f"{key} must be the reserved CPU list")
    _check_campaign(ctx, argv, plan)

    # 5. Reservation command: the export's own wrapper, with the selected unit, state and log.
    launch = load_module("prepare_author_launch", ctx.path(plan["launch_modules"]))
    reservation = plan["reservation"]
    duration = policy["duration_s"]
    reserved = launch.reservation_argv(
        argv,
        environment,
        task=ctx.path("${TASK}"),
        output=output,
        run_id=ctx.run_id,
        cpus=ctx.cpus,
        duration_s=duration + int(reservation["margin_s"]),
    )
    for flag, key in (("--state", "state"), ("--log", "log"), ("--unit", "unit")):
        if reservation.get(key):
            set_wrapper_option(reserved, flag, ctx.x(reservation[key]))
    launch.prepare_slot_roots(environment)

    framework = ctx.path("${FRAMEWORK}")
    record = ctx.x_tree(dict(plan.get("record") or {}))
    # Never carry historical roster counts into a newly prepared campaign.
    record.pop("capsules", None)
    inspection = ctx.path("${INPUTS}") / "inspect.json"
    seal = ctx.path("${RELEASE}") / "private/seal.json"
    if inspection.is_file() and seal.is_file():
        inspected, sealed = json.loads(inspection.read_text()), json.loads(seal.read_text())
        if (Path(inspected.get("release", "")).resolve() != ctx.path("${RELEASE}").resolve()
                or inspected.get("review_digest") != sealed.get("review_digest")):
            raise SpecError("capsule counts must come from this release's sealed inspection")
        counts = inspected["counts"]
        core = counts["public_admitted"]
        scalability = counts.get("public_scalability", {}).get("source_cases", 0)
        programs = counts.get("public_programs", {}).get("source_cases", 0)
        record["capsules"] = {"total_public": core + scalability + programs,
            "core": core, "scalability": scalability, "connected_programs": programs,
            "basis": "sealed admission inventory; not execution results"}
    record.update(
        {
            "execution_started": False,
            "run_id": ctx.run_id,
            "cwd": str(framework),
            "source_revision": git(ctx.path("${REPO}"), "rev-parse", f"{ctx.lookup('EXPORT_ID')}^{{commit}}"),
            "framework_repository": str(framework),
            "full_compiler_manifest_sha256": ctx.computed["FULL_COMPILER_MANIFEST_SHA256"],
            "cpu_reservation": {"cpus": cpu_list(ctx.cpus), "shared_across_roles": True, "currently_acquired": False},
            "launched_at": None,
            "launcher_pid": None,
            "treatment": run["treatment"],
            "source_release": str(ctx.path("${RELEASE}")),
            "timing_selection": timing,
            "argv": argv,
            "reserved_argv": reserved,
            "environment": environment,
            "launch_spec": {
                "path": str(ctx.spec_path),
                "sha256": sha256(ctx.spec_path),
                "extends": [{"path": str(p), "sha256": sha256(p)} for p in ctx.spec_sources[1:]],
                "unit": ctx.unit,
            },
        }
    )
    for directory in (ctx.path("${RUN_STATE}"), ctx.path("${PRIVATE_ROOT}")):
        directory.mkdir(parents=True, exist_ok=True)
    private = ctx.path("${PRIVATE_ROOT}")
    if private.stat().st_mode & 0o077:
        os.chmod(private, 0o700)
    if write:
        (output / "proposed-launch.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _check_campaign(ctx: Context, argv: list[str], plan: dict) -> None:
    """Fresh-launch invariants: no seed, no stale origin, every read input present."""
    leftover = [item for item in argv if "${" in item]
    if leftover:
        raise SpecError(f"unexpanded variables: {leftover[:3]}")
    for flag in plan.get("required_flags") or ():
        if flag not in argv:
            raise SpecError(f"fresh launch requires {flag}")
    for flag in plan.get("forbidden_flags") or ():
        if flag in argv:
            raise SpecError(f"fresh launch must not pass {flag}")
    stale = [item for item in argv if any(part in item for part in plan.get("forbidden_fragments") or ())]
    if stale:
        raise SpecError(f"stale origin paths remain: {stale[:3]}")
    for flag in plan.get("input_flags") or ():
        value = argv[argv.index(flag) + 1]
        if not Path(value).exists():
            raise SpecError(f"missing input for {flag}: {value}")


def cmd_plan(ctx: Context, args) -> int:
    record = build_plan(ctx, helper_effort=args.author_helper_effort)
    print(
        json.dumps(
            {
                "artifacts": str(ctx.path("${ARTIFACTS}")),
                "run_id": ctx.run_id,
                "unit": ctx.unit,
                "duration_s": int(ctx.computed["DURATION_S"]),
                "cpus": record["environment"]["MERLIN_COMPUTE_CPUS"],
                "argv_tail": record["reserved_argv"][-12:],
            },
            indent=2,
        )
    )
    return 0


# --------------------------------------------------------------------------- preflight / launch


def preflight_script(plan: dict, *, run: bool) -> str:
    """The plan's exact command: its own run id and --preflight-only, or (run) unchanged and unreserved."""
    argv = plan["reserved_argv"]
    command = argv[argv.index("/usr/bin/env") :]
    if not run:
        name, preflight = plan["run_id"], plan["run_id"] + "-preflight"
        command = [item.replace(name, preflight) for item in command] + ["--preflight-only"]
    cpus = plan["environment"]["MERLIN_COMPUTE_CPUS"]
    slots = [
        item.split("=", 1)[1]
        for item in command
        if item.split("=", 1)[0] in ("MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT")
    ]
    lines = ["#!/bin/bash", "set -e", f"cd {shlex.quote(plan['cwd'])}"]
    lines += [f"mkdir -p {shlex.quote(slot)} && chmod 700 {shlex.quote(slot)}" for slot in slots]
    if run:
        # Beside another reservation: the open-file limit and wall-clock bound the reservation would set.
        bound = argv[argv.index("--max-seconds") + 1]
        lines.append("ulimit -n 65536")
        lines.append(f"exec timeout -s TERM -k 60 {bound} taskset -c {cpus} " + shlex.join(command))
    else:
        lines.append(f"exec taskset -c {cpus} " + shlex.join(command))
    return "\n".join(lines) + "\n"


def cmd_preflight(ctx: Context, args) -> int:
    section = ctx.section("preflight")
    plan = json.loads((ctx.path("${ARTIFACTS}") / "proposed-launch.json").read_text())
    script = ctx.path(section["run_script" if args.run else "script"])
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(preflight_script(plan, run=args.run))
    script.chmod(0o755)
    print(json.dumps({"script": str(script), "mode": "unreserved run" if args.run else "preflight"}))
    if not args.execute or args.run:
        return 0
    # A preflight starts from empty preflight state: earlier attempts move aside, never deleted.
    retired = ctx.path(section["retire_into"]) / str(int(time.time()))
    for item in section.get("retire") or ():
        source = ctx.path(item["path"])
        if source.exists():
            retired.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(retired / item["as"]))
    log = ctx.path(section["log"])
    with open(log, "wb") as handle:
        try:
            code = subprocess.run([str(script)], stdout=handle, stderr=subprocess.STDOUT, timeout=section["timeout_s"])
            code = code.returncode
        except subprocess.TimeoutExpired:
            code = 124
    lines = log.read_text(errors="replace").splitlines()
    if code == 0:
        print("PREFLIGHT OK")
    else:
        print(f"PREFLIGHT rc={code}")
        for line in lines:
            if "NO_GO" in line:
                print(line[:600])
    for line in lines[-3:]:
        print(line[:200])
    return code


def cmd_launch(ctx: Context, args) -> int:
    path = ctx.path("${ARTIFACTS}") / "proposed-launch.json"
    plan = json.loads(path.read_text())
    if not args.confirm:
        print(json.dumps({"would_execute": plan["reserved_argv"][:12], "cwd": plan["cwd"]}, indent=2))
        print("refusing to launch without --confirm", file=sys.stderr)
        return 2
    if plan.get("execution_started"):
        raise SpecError("this plan was already launched; plan again for a new run")
    plan["execution_started"] = True
    path.write_text(json.dumps(plan, indent=2) + "\n")
    with open(ctx.path("${ARTIFACTS}") / "reservation-launch.log", "ab") as log:
        process = subprocess.Popen(
            plan["reserved_argv"], cwd=plan["cwd"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    print("launched reservation pid", process.pid)
    return 0


# --------------------------------------------------------------------------- check


def error_lines(lines, markers) -> list[str]:
    """Lines matching any marker; a marker is a sequence of fragments that occur in order (case-folded)."""
    found = []
    for line in lines:
        folded = line.casefold()
        for marker in markers:
            parts = [marker] if isinstance(marker, str) else list(marker)
            cursor = 0
            for part in parts:
                cursor = folded.find(part.casefold(), cursor)
                if cursor < 0:
                    break
                cursor += len(part)
            else:
                found.append(line)
                break
    return found


def unit_active(unit: str) -> bool:
    return bool(unit) and subprocess.run(["systemctl", "is-active", "-q", unit]).returncode == 0


def planned_unit(ctx: Context) -> str:
    """The selected unit, else the one the written plan's reservation wrapper names."""
    if ctx.unit:
        return ctx.unit
    plan = ctx.path("${ARTIFACTS}") / "proposed-launch.json"
    if not plan.exists():
        return ""
    reserved = json.loads(plan.read_text())["reserved_argv"]
    wrapper = reserved[: reserved.index("--")]
    return wrapper[wrapper.index("--unit") + 1]


def status(ctx: Context) -> dict:
    section = ctx.section("check")
    unit = planned_unit(ctx)
    log, run_dir = ctx.path(section["log"]), ctx.path(section["run_dir"])
    lines = log.read_text(errors="replace").splitlines() if log.exists() else []
    errors = error_lines(lines, section["error_markers"])
    summaries = sorted(run_dir.glob("joint_campaign_summary.json")) + sorted(
        run_dir.glob("*/joint_campaign_summary.json")
    )
    free = shutil.disk_usage(ctx.path(section["disk_path"])).free >> 30
    return {
        "unit": unit,
        "unit_active": unit_active(unit),
        "summary": str(summaries[0]) if summaries else None,
        "errors": len(errors),
        "recent_errors": [line[:250] for line in errors[-3:]],
        "disk_free_gib": free,
        "disk_low": free < int(section["disk_min_gib"]),
        "log_tail": [line[:200] for line in lines[-3:]],
    }


def workspace_progress(ctx: Context) -> dict:
    workspace = ctx.path(ctx.section("check")["workspace"])
    responses = sorted(
        [*workspace.glob(".qa_channel/resp_*"), *workspace.glob(".qa_channel/simresp_*")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[:40]
    counts = {}
    for key in ctx.section("check").get("response_keys") or ():
        counts[key] = sum(1 for p in responses if key in p.read_text(errors="replace"))
    progress = workspace / "qa/progress.json"
    selected = {}
    if progress.exists():
        record = json.loads(progress.read_text())
        selected = {k: record.get(k) for k in ("state", "n_selected", "n_completed", "n_passed", "updated_at")}
    return {"recent_responses": len(responses), "response_keys": counts, "qa_progress": selected}


def console_dump(ctx: Context) -> dict:
    console = ctx.section("check")["console"]
    command = ["nice", "-n", "19", "timeout", str(console.get("timeout_s", 400)), ctx.x(console["python"]), "-B"]
    command += ["-m", "merlin_experiments.tracking.console.serve", *(ctx.x(item) for item in console["args"])]
    environment = dict(os.environ, PYTHONPATH=ctx.x(console["pythonpath"]))
    out = subprocess.run(command, cwd=ctx.x(console["cwd"]), env=environment, capture_output=True, text=True)
    dump = json.loads(out.stdout)
    sections = dump["sections"]
    return {
        "diagnostics": [(row["code"], row["severity"], row["title"][:90]) for row in dump["diagnostics"]],
        "sections": {
            name: "; ".join(f"{row['label']}={row['value']}" for row in (sections.get(name) or {}).get("summary", []))[
                :900
            ]
            for name in console.get("sections") or ()
        },
    }


def cmd_check(ctx: Context, args) -> int:
    section = ctx.section("check")
    if args.watch:
        baseline = status(ctx)["errors"]
        while True:
            current = status(ctx)
            stamp = dt.datetime.now(dt.UTC).strftime("%H:%M:%S")
            reason = None
            if not current["unit_active"]:
                reason = f"UNIT INACTIVE at {stamp}"
            elif current["summary"]:
                reason = f"SUMMARY {current['summary']} at {stamp}"
            elif current["errors"] > baseline:
                reason = f"NEW ERRORS ({current['errors']}) at {stamp}"
            elif current["disk_low"]:
                reason = f"DISK LOW {current['disk_free_gib']}G at {stamp}"
            if reason:
                print(json.dumps({"attention": reason, **current}, indent=2))
                return 0
            time.sleep(int(section.get("interval_s", 60)))
    if args.until:
        while dt.datetime.now(dt.UTC).strftime("%H%M") < args.until:
            if not unit_active(planned_unit(ctx)):
                print(json.dumps({"attention": "UNIT INACTIVE", **status(ctx)}, indent=2))
                return 1
            time.sleep(30)
    report = {**status(ctx), **workspace_progress(ctx)}
    if args.console:
        report["console"] = console_dump(ctx)
    print(json.dumps(report, indent=2))
    return 0


# --------------------------------------------------------------------------- CLI


def parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--spec", required=True, help="fresh campaign launch spec (YAML)")
    common.add_argument("--var", action="append", default=[], metavar="NAME=VALUE", help="override a spec variable")
    common.add_argument("--run-id", help="override the spec's run id")
    common.add_argument("--unit", help="override the spec's reservation unit name")
    common.add_argument("--cpus", help="override the spec's reserved CPU list (comma separated)")
    common.add_argument("--export-id", help="framework export (12-hex commit); default: the repo's HEAD")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("export", parents=[common], help="frozen framework export of the branch HEAD")
    sub.add_parser("inputs", parents=[common], help="prepare author library, full compiler and corpus")
    sub.add_parser("_inputs", parents=[common], help=argparse.SUPPRESS)
    seal = sub.add_parser("seal", parents=[common], help="inspect and seal the prepared corpus")
    seal.add_argument("--inspect-only", action="store_true")
    seal.add_argument("--reviewed-by")
    seal.add_argument("--review-note")
    plan = sub.add_parser("plan", parents=[common], help="write policy, pins and proposed-launch.json")
    plan.add_argument("--author-helper-effort", help="Codex helper reasoning effort (default: the spec's)")
    preflight = sub.add_parser("preflight", parents=[common], help="write (and run) the no-author preflight")
    preflight.add_argument("--run", action="store_true", help="write the unreserved full-run script instead")
    preflight.add_argument("--execute", action="store_true", help="run the preflight script now")
    launch = sub.add_parser("launch", parents=[common], help="start the reserved campaign")
    launch.add_argument("--confirm", action="store_true")
    check = sub.add_parser("check", parents=[common], help="read-only campaign status")
    check.add_argument("--watch", action="store_true", help="return once the campaign needs attention")
    check.add_argument("--until", metavar="HHMM", help="wait until this UTC time, then report")
    check.add_argument("--console", action="store_true", help="include the experiment console dump")
    return p


COMMANDS = {
    "export": cmd_export,
    "inputs": cmd_inputs,
    "_inputs": cmd_inputs_stage,
    "seal": cmd_seal,
    "plan": cmd_plan,
    "preflight": cmd_preflight,
    "launch": cmd_launch,
    "check": cmd_check,
}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ctx = Context(
        args.spec, overrides=args.var, run_id=args.run_id, unit=args.unit, cpus=args.cpus, export_id=args.export_id
    )
    return COMMANDS[args.command](ctx, args)


if __name__ == "__main__":
    raise SystemExit(main())
