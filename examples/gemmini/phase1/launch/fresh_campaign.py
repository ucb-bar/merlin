"""Reproducible fresh open-compiler campaign launch from one reviewed spec.

    fresh_campaign.py export    --spec SPEC   # frozen framework export of the branch HEAD
    fresh_campaign.py inputs    --spec SPEC   # author library, full compiler, references, corpus
    fresh_campaign.py seal      --spec SPEC   # corpus inspect + reviewed seal
    fresh_campaign.py ordering  --spec SPEC   # completion-order selection from the run's facts
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

A spec with a ``seed:`` section starts from an existing compiler instead of an empty backend;
``fresh_seed.py`` beside this file owns that path, and a spec without one never loads it.
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
# CPU_COUNT is the number of reserved CPUs (in CPUS), for settings sized from the reservation.
BUILTINS = (
    "RUN_ID",
    "RUN_TAG",
    "UNIT",
    "CPUS",
    "CPU_COUNT",
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

    def __init__(
        self,
        spec_path: str | Path,
        *,
        overrides=(),
        run_id=None,
        unit=None,
        cpus=None,
        export_id=None,
        reference_guidance=None,
    ):
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
        plan = self.spec.get("plan") or {}
        default_guidance = plan.get("reference_guidance", "on" if plan.get("reference_experiments") else "off")
        self.reference_guidance = reference_guidance if reference_guidance is not None else default_guidance
        if self.reference_guidance not in ("off", "on"):
            raise SpecError("reference guidance must be explicitly off or on")
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
            "CPU_COUNT": str(len(cpu_list(self.cpus))),
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
    # The export has no .git: record its own revision, or every provenance read inside it would walk up
    # to whichever checkout encloses it and report that branch's moving HEAD instead.
    from merlin.targetgen.provenance import write_source_record

    write_source_record(framework, head)
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


ORDERING_SCRIPT = """
import json, sys
from pathlib import Path
from merlin.targetgen.rtl import command_ordering_selection as owner
target, facts, contract, declarations, output = sys.argv[1:6]
pin = owner.prepare(target=target, facts=Path(facts), contract=Path(contract),
                    declarations=Path(declarations), output=Path(output))
print(json.dumps(pin))
"""


def ordering_inputs(ctx: Context) -> dict[str, Path]:
    """The facts and contract the campaign grades against, the declarations and the selection to write."""
    section = ctx.section("ordering")
    # Other plan entries contain bindings produced later by build_plan. Ordering
    # needs only these source identities and must be independently preparable.
    environment = ctx.section("plan")["environment"]
    return {
        "facts": ctx.path(environment["MERLIN_RTL_FACTS"]),
        "contract": ctx.path(environment["MERLIN_TARGET_CONTRACT"]),
        "declarations": ctx.path(section["declarations"]),
        "output": ctx.path(ctx.section("plan")["ordering_selection"]),
    }


def cmd_ordering(ctx: Context, args) -> int:
    """Derive the completion-order selection from this campaign's own facts and contract (idempotent)."""
    paths = ordering_inputs(ctx)
    output = paths["output"]
    if output.is_file():
        selection = json.loads(output.read_text())
        bound = all(sha256(paths[key]) == selection[f"base_{key}"]["sha256"] for key in ("facts", "contract"))
        if not bound:
            raise SpecError(f"{output} binds other facts or a different contract; remove it to derive again")
        print(json.dumps({"selection": str(output), "sha256": sha256(output), "reused": True}))
        return prepare_command_recorder(ctx)
    environment, framework = _prepared_runtime(ctx)
    command = [
        "taskset",
        "-c",
        ctx.preparation_cpus,
        ctx.x("${PYTHON}"),
        "-B",
        "-c",
        ORDERING_SCRIPT,
        ctx.x(ctx.section("ordering")["target"]),
    ]
    command += [str(paths[key]) for key in ("facts", "contract", "declarations", "output")]
    result = subprocess.run(command, cwd=framework, env=environment).returncode
    return result or prepare_command_recorder(ctx)


def prepare_command_recorder(ctx: Context) -> int:
    """Build the explicitly selected operator recorder using the frozen framework's sources."""
    config = (ctx.spec.get("ordering") or {}).get("command_recorder")
    if config is None:
        return 0
    environment, framework = _prepared_runtime(ctx)
    command = [
        "taskset",
        "-c",
        ctx.preparation_cpus,
        ctx.x("${PYTHON}"),
        "-B",
        "-m",
        "merlin.perf.command_recorder",
        "--engine",
        str(ctx.path(config["engine"])),
        "--include",
        str(ctx.path(config["include"])),
        "--out",
        str(ctx.path(config["output"])),
    ]
    return subprocess.run(command, cwd=framework, env=environment).returncode


def command_recorder_pin(ctx: Context) -> dict | None:
    """Reopen every recorder product and source pin before putting it in a launch environment."""
    config = (ctx.spec.get("ordering") or {}).get("command_recorder")
    if config is None:
        return None
    manifest = ctx.path(config["output"]) / "manifest.json"
    try:
        doc = json.loads(manifest.read_text())
        source = ctx.path("${FRAMEWORK}") / "src/merlin/perf/spike_command_recorder.cc"
        engine = ctx.path(config["engine"]).resolve()
        if (
            doc["schema"] != "merlin.command_recorder.v1"
            or Path(doc["engine"]["path"]).resolve() != engine
            or sha256(engine) != doc["engine"]["sha256"]
            or sha256(Path(doc["library"]["path"])) != doc["library"]["sha256"]
            or sha256(source) != doc["source_sha256"]
        ):
            raise ValueError("recorder source, engine or library changed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SpecError(f"command recorder unavailable; run ordering first: {exc}") from exc
    return {"path": str(manifest), "sha256": sha256(manifest)}


def forbidden_references(members, prefixes) -> list[str]:
    return [m["path"] for m in members if any(m["path"].startswith(prefix) for prefix in prefixes)]


def reference_members(document):
    """Publication-reviewed v2 destinations, or legacy historical manifest entries."""
    if document.get("schema") == "operator.allowed_reference_sources.v2":
        return [
            {"path": origin["name"] + "/" + member["destination"]}
            for origin in document["origins"]
            for member in origin["files"]
        ]
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


def profile_reference_manifest(ctx, section, destination, profile):
    """Stage reviewed examples, or require an explicit profile-bound absence.

    A no-reference treatment passes ``None`` into ordinary corpus preparation;
    it never weakens positive-origin review by inventing an empty review bank.
    """
    selected = section.get("public_references")
    if getattr(profile, "raw_kernel_references", True) is False:
        if selected is not None:
            raise SpecError("the no-reference author profile cannot select public references")
        if ctx.reference_guidance != "off" or ctx.section("plan").get("reference_map") is not None:
            raise SpecError("the no-reference author profile cannot select reference guidance or a map")
        if destination.exists() or destination.is_symlink():
            raise SpecError("the no-reference input destination must be absent")
        return None
    if selected is None:
        raise SpecError("this author profile requires its reviewed public references")
    return stage_launch_references(
        ctx.path(selected) / "reference-manifest.json", destination,
        section.get("forbidden_reference_prefixes") or (),
    )


def planned_reference_manifest(ctx, plan, profile):
    """Bind a reviewed reference manifest or the explicit no-reference treatment."""
    selected = plan.get("public_reference_manifest")
    if getattr(profile, "raw_kernel_references", True) is False:
        if (selected is not None or ctx.section("inputs").get("public_references") is not None
                or ctx.reference_guidance != "off" or plan.get("reference_map") is not None):
            raise SpecError("the no-reference plan cannot select reference material, guidance or a map")
        return None
    if selected is None:
        raise SpecError("this plan requires its reviewed public reference manifest")
    references = ctx.path(selected)
    prefixes = ctx.section("inputs").get("forbidden_reference_prefixes") or ()
    forbidden = forbidden_references(reference_members(json.loads(references.read_text())), prefixes)
    if forbidden:
        raise SpecError(f"public references carry forbidden members: {forbidden[:3]}")
    return references


def extra_source_selections(ctx, section):
    """Forward explicit native source choices; a run pin hashes its resolved-plan.json.

    The native release owner independently validates that completed producer's
    frozen inputs, output and execution receipts. This pin selects its plan,
    without hashing a potentially large run tree or carrying a release seal.
    """
    selected = {}
    tuple_selections = {"host_capability_selection", "operand_protocol_selection", "operand_protocol_revision"}
    for key in ("public_program_suite", "host_capability_selection", "phase1_policy_descriptor", "source_run",
                "operand_protocol_selection", "operand_protocol_revision"):
        pin = section.get(key)
        if pin is None:
            continue
        if not isinstance(pin, dict) or set(pin) != {"path", "sha256"}:
            raise SpecError(f"{key} requires an explicit path/sha256 selection")
        path = ctx.path(pin["path"])
        member = path / "resolved-plan.json" if key == "source_run" else path
        if sha256(member) != pin["sha256"]:
            raise SpecError(f"{key} selection changed: {path}")
        selected[key] = (path, pin["sha256"]) if key in tuple_selections else path
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

    from merlin_experiments.phase1.author_profiles import prepare_sources, resolve
    from merlin_experiments.phase1.full_compiler_workspace import prepare as prepare_full

    profile = resolve(ctx.section("plan").get("author_profile", ""))
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
    if profile is not None and profile.name.startswith("fresh10-tooling-v1/"):
        from merlin_experiments.phase1.fresh10_sources import project_author_library

        report = {**report, **project_author_library(out / "author-library", profile.name, repo=framework)}
    contract = report["contract_sha256"]
    print(json.dumps({"stage": "author_library", "contract": contract, "members": report["members"]}), flush=True)

    full = (prepare_full(framework, out / "full-compiler", review_id=ctx.x(section["full_compiler_review_id"]))
            if profile is None else prepare_sources(framework, out / "full-compiler", profile=profile,
                                                    review_id=ctx.x(section["full_compiler_review_id"]),
                                                    source_selection=ctx.x_tree(section.get("author_profile_source_selection"))))
    if full is not None:
        full.verify()
    (out / "full-compiler-selection.json").write_text(json.dumps(None if full is None else full.record(), indent=2) + "\n")
    print(json.dumps({"stage": "full_compiler", "members": len(full.members) if full else 0,
                      "manifest": full.manifest_sha256 if full else None}))

    # Public references: the pinned base members only; a forbidden prefix (private measured kernels)
    # refuses the whole preparation before anything is copied.
    reference_manifest = profile_reference_manifest(
        ctx, section, out / "public-references", profile,
    )

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
        **(dict(author_profile=profile.name, author_profile_sources=full,
                author_profile_common_inputs=section.get("author_profile_common_inputs")) if profile else {}),
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
        "full_compiler": None if full is None else full.record(),
        **({"author_profile": profile.record()} if profile else {}),
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
    policy["duration_s"] = int(max_wall_s)
    author_reserve = min(1800, max(1, policy["duration_s"] // 2))
    policy["final_model_s"] = min(int(final_model_s), policy["duration_s"] - author_reserve)
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


def profile_omitted_flags(profile):
    """Keep argument construction and input admission on the same capability boundary."""
    if profile is None or profile.seeded:
        return set()
    if profile.name.startswith("fresh10-tooling-v1/"):
        return set() if profile.allows("transform_library") else {"--required-shared-pass"}
    removed = {"--compiler-library-root", "--compiler-library-contract", "--compiler-library-contract-sha256",
               "--required-shared-pass"}
    if profile.level == 1:
        removed.update(("--full-compiler-root", "--full-compiler-manifest", "--full-compiler-manifest-sha256"))
    return removed


def profile_campaign_argv(profile, argv):
    """Apply the source and reuse requirements of the selected author treatment."""
    removed = profile_omitted_flags(profile)
    filtered, cursor = [], 0
    while cursor < len(argv):
        if argv[cursor] in removed:
            cursor += 2
        else:
            filtered.append(argv[cursor])
            cursor += 1
    if (profile is not None and not profile.seeded and not profile.name.startswith("fresh10-tooling-v1/")
            and "--compiler-access" in filtered):
        filtered[filtered.index("--compiler-access") + 1] = (
            "selected_library" if profile.level == 1 else "full_compiler"
        )
    return filtered


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
    from merlin_experiments.phase1.author_profiles import resolve as resolve_author_profile

    author_profile = resolve_author_profile(plan.get("author_profile", ""))
    manifest = None if author_profile and author_profile.level == 1 else ctx.path(plan["full_compiler_manifest"])
    library = restore_library(ctx.path(plan["library_selection"]))
    contract = output / "selected-library-contract.json"
    contract.write_text(json.dumps(library.contract.record(), sort_keys=True, indent=2) + "\n")
    references = planned_reference_manifest(ctx, plan, author_profile)

    cap = (lambda value: str(min(int(value), int(workers)))) if workers else str
    ctx.computed.update(
        {
            "DURATION_S": str(policy["duration_s"]),
            "RATE_LIMIT_WAITS": str(run["rate_limit_waits"]),
            "SIM_MAX_JOBS": cap(run["sim_max_jobs"]),
            "LIBRARY_ROOT": str(library.root),
            "LIBRARY_CONTRACT_SHA256": library.contract.sha256,
            "FULL_COMPILER_MANIFEST_SHA256": sha256(manifest) if manifest else "",
            "PERFORMANCE_SELECTION_PIN": json.dumps(timing),
        }
    )

    # 4. Campaign argv and environment, exactly as the spec spells them.
    raw_argv = profile_campaign_argv(author_profile, plan["argv"])
    argv = [ctx.x(item) for item in raw_argv]
    if author_profile is not None:
        argv += ["--author-profile", author_profile.name]
    reference = plan.get("reference_experiments") if ctx.reference_guidance == "on" else None
    if ctx.reference_guidance == "on" and reference is None:
        raise SpecError("reference guidance on requires a pinned reference_experiments selection")
    if reference is not None:
        from merlin_experiments.phase1.feedback.reference_experiments import invocation

        if (
            not isinstance(reference, dict)
            or set(reference) != {"path", "sha256"}
            or any(flag in argv for flag in ("--reference-experiments", "--reference-experiments-sha256"))
        ):
            raise SpecError("reference_experiments requires one explicit path/sha256 pin")
        reference = {"path": str(ctx.path(reference["path"]).resolve()), "sha256": ctx.x(reference["sha256"])}
        try:
            with invocation(reference["path"], reference["sha256"]):
                pass  # Reopen the complete selected closure before writing any enabled launch.
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise SpecError(f"reference experiment selection is invalid: {error}") from error
        argv += ["--reference-experiments", reference["path"], "--reference-experiments-sha256", reference["sha256"]]
    tasks = plan.get("reference_experiment_tasks", "on")
    if tasks not in ("on", "off") or "--reference-experiments-tasks" in argv:
        raise SpecError("reference_experiment_tasks must be on or off (the spec's argv must not pass it)")
    if reference is not None and tasks == "off":
        # Keep the selection's analysis grant; serve none of its prior tasks.
        argv += ["--reference-experiments-tasks", "off"]
    effort = helper_effort if helper_effort is not None else run.get("author_helper_effort")
    if effort and "--author-helper-effort" not in argv:
        argv += ["--author-helper-effort", str(effort)]
    environment = {key: ctx.x(value) for key, value in plan["environment"].items()}
    reference_map = pin_reference_map(ctx, plan, output, references)
    if reference_map is not None:
        from merlin_experiments.phase1.feedback.reference_map import ENV as REFERENCE_MAP_ENV

        if REFERENCE_MAP_ENV in environment:
            raise SpecError(f"select the reference map through plan.reference_map, not {REFERENCE_MAP_ENV}")
        environment[REFERENCE_MAP_ENV] = json.dumps(reference_map)
    recorder = command_recorder_pin(ctx)
    if recorder is not None:
        environment["MERLIN_COMMAND_RECORDER"] = json.dumps(recorder)
    if plan.get("ordering_selection"):
        selected = ctx.path(plan["ordering_selection"])
        if not selected.is_file():
            raise SpecError(f"no ordering selection at {selected}: run `fresh_campaign.py ordering` first")
        pinned = output / "completion-order-selection.json"
        if selected.resolve() != pinned.resolve():
            shutil.copyfile(selected, pinned)
        environment[ORDERING_ENV] = json.dumps({"path": str(pinned), "sha256": sha256(pinned)})
    if workers:
        environment = {k: cap(v) if k in WORKER_KEYS else v for k, v in environment.items()}
    for key in ("MERLIN_COMPUTE_CPUS", "MERLIN_CODEX_CPUS", "MERLIN_PREPARATION_CPUS"):
        if environment.get(key) != ctx.cpus:
            raise SpecError(f"{key} must be the reserved CPU list")
    _check_campaign(ctx, argv, plan)
    check_ordering_selection(environment, framework=ctx.path("${FRAMEWORK}"))
    check_capture_interpreter(environment, framework=ctx.path("${FRAMEWORK}"))

    # 5. Reservation command: the export's own wrapper, with the selected unit, state and log.
    launch = load_module("prepare_author_launch", ctx.path(plan["launch_modules"]))
    reservation = plan["reservation"]
    duration = policy["duration_s"]

    def reserve(argv, environment):
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
            # An unselected unit expands to "" and keeps the wrapper's run-id-derived name.
            value = ctx.x(reservation[key]) if reservation.get(key) else ""
            if value:
                set_wrapper_option(reserved, flag, value)
        return reserved

    reserved = reserve(argv, environment)
    launch.prepare_slot_roots(environment)

    framework = ctx.path("${FRAMEWORK}")
    record = ctx.x_tree(dict(plan.get("record") or {}))
    # Never carry historical roster counts into a newly prepared campaign.
    record.pop("capsules", None)
    inspection = ctx.path("${INPUTS}") / "inspect.json"
    seal = ctx.path("${RELEASE}") / "private/seal.json"
    if inspection.is_file() and seal.is_file():
        inspected, sealed = json.loads(inspection.read_text()), json.loads(seal.read_text())
        if Path(inspected.get("release", "")).resolve() != ctx.path("${RELEASE}").resolve() or inspected.get(
            "review_digest"
        ) != sealed.get("review_digest"):
            raise SpecError("capsule counts must come from this release's sealed inspection")
        counts = inspected["counts"]
        core = counts["public_admitted"]
        scalability = counts.get("public_scalability", {}).get("source_cases", 0)
        programs = counts.get("public_programs", {}).get("source_cases", 0)
        record["capsules"] = {
            "total_public": core + scalability + programs,
            "core": core,
            "scalability": scalability,
            "connected_programs": programs,
            "basis": "sealed admission inventory; not execution results",
        }
    record.update(
        {
            "reference_guidance": ctx.reference_guidance,
            **({"author_profile": author_profile.record()} if author_profile else {}),
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
            "reference_manifest": (
                None if references is None else {"path": str(references), "sha256": sha256(references)}
            ),
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
    if reference is not None:
        record["reference_experiments"] = reference
        if tasks == "off":
            record["reference_experiment_tasks"] = "off"
    if reference_map is not None:
        record["reference_map"] = reference_map
    input_selection = ctx.path(ctx.section("inputs")["output"]) / "inputs-selection.json"
    if input_selection.is_file():
        record["inputs_selection"] = {"path": str(input_selection), "sha256": sha256(input_selection)}
    for directory in (ctx.path("${RUN_STATE}"), ctx.path("${PRIVATE_ROOT}")):
        directory.mkdir(parents=True, exist_ok=True)
    private = ctx.path("${PRIVATE_ROOT}")
    if private.stat().st_mode & 0o077:
        os.chmod(private, 0o700)
    if ctx.spec.get("seed") is not None:
        # The fresh invariants above held for the unseeded launch; the seed rewrites it (fresh_seed.apply).
        load_module("fresh_seed", HERE).apply(
            ctx, record, validate_policy=_validate_policy, check_campaign=_check_campaign, reserve=reserve
        )
    if write:
        (output / "proposed-launch.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def pin_reference_map(ctx: Context, plan: dict, output: Path, references: Path | None) -> dict | None:
    """The operator-built reference map, verified and copied beside the launch: ``{path, sha256}``.

    The same bytes for every arm that names the same pin; it must describe this launch's reference
    manifest. Absent (or null), no map is staged and no feedback names a reference region."""
    selected = plan.get("reference_map")
    if selected is None:
        return None
    if references is None:
        raise SpecError("a reference map requires its reviewed reference manifest")
    if not isinstance(selected, dict) or set(selected) != {"path", "sha256"}:
        raise SpecError("reference_map requires one explicit path/sha256 pin")
    path, digest = ctx.path(selected["path"]), ctx.x(selected["sha256"])
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise SpecError(
            f"reference_map sha256 ({selected['sha256']} = {digest!r}) must be the 64-hex digest of "
            f"{path}: set it in the spec or pass --var NAME=SHA256 once the map is reviewed"
        )
    if not path.is_file():
        raise SpecError(f"reference map is unavailable: {path}")
    from merlin_experiments.phase1.feedback.reference_map import FILENAME, read_pin

    try:
        _, document = read_pin({"path": str(path.resolve()), "sha256": digest})
    except ValueError as error:
        raise SpecError(f"reference map is invalid: {error}") from error
    if document["reference_manifest_sha256"] != sha256(references):
        raise SpecError("reference map describes another reference manifest than this launch selects")
    pinned = output / FILENAME
    if pinned.resolve() != path.resolve():
        shutil.copyfile(path, pinned)
    if sha256(pinned) != digest:
        raise SpecError(f"reference map copy differs from its pin: {pinned}")
    return {"path": str(pinned), "sha256": digest}


def _check_campaign(ctx: Context, argv: list[str], plan: dict) -> None:
    """Fresh-launch invariants: no seed, no stale origin, every read input present."""
    leftover = [item for item in argv if "${" in item]
    if leftover:
        raise SpecError(f"unexpanded variables: {leftover[:3]}")
    from merlin_experiments.phase1.author_profiles import resolve

    profile = resolve(plan.get("author_profile", ""))
    omitted = profile_omitted_flags(profile)
    for flag in [f for f in plan.get("required_flags") or () if f not in omitted]:
        if flag not in argv:
            raise SpecError(f"fresh launch requires {flag}")
    for flag in plan.get("forbidden_flags") or ():
        if flag in argv:
            raise SpecError(f"fresh launch must not pass {flag}")
    stale = [item for item in argv if any(part in item for part in plan.get("forbidden_fragments") or ())]
    if stale:
        raise SpecError(f"stale origin paths remain: {stale[:3]}")
    for flag in [f for f in plan.get("input_flags") or () if f not in omitted]:
        value = argv[argv.index(flag) + 1]
        if not Path(value).exists():
            raise SpecError(f"missing input for {flag}: {value}")


ORDERING_ENV = "MERLIN_COMPLETION_ORDER_SELECTION"
ORDERING_BINDINGS = (("base_contract", "MERLIN_TARGET_CONTRACT"), ("base_facts", "MERLIN_RTL_FACTS"))


def check_ordering_selection(environment: dict, *, framework: Path) -> None:
    """An ordering selection binds exact facts, contract and reader bytes; the grader reports any other
    pairing as unknown and never checks a kernel, so refuse one this campaign would not select."""
    raw = environment.get(ORDERING_ENV)
    if not raw:
        return
    pin = json.loads(raw)
    path = Path(pin["path"])
    if not path.is_file() or sha256(path) != pin["sha256"]:
        raise SpecError(f"{ORDERING_ENV} does not match its pinned bytes: {path}")
    selection = json.loads(path.read_text())
    for key, selected_key in ORDERING_BINDINGS:
        selected = environment.get(selected_key)
        if not selected or not Path(selected).is_file() or sha256(selected) != selection[key]["sha256"]:
            raise SpecError(
                f"{ORDERING_ENV} binds {key} {selection[key]['path']}, not this campaign's {selected_key} "
                f"{selected}: prepare a selection from this export's facts and contract"
            )
    roots = [framework / "src", *sorted((framework / "packages").glob("*/src"))]
    for module, digest in selection["readers"].items():
        relative = Path(*module.split(".")).with_suffix(".py")
        found = [root / relative for root in roots if (root / relative).is_file()]
        if len(found) != 1 or sha256(found[0]) != digest:
            raise SpecError(
                f"{ORDERING_ENV} was prepared with a different reader {module}: prepare it from this export"
            )


CAPTURE_PYTHON_ENV = "MERLIN_M2M_PYTHON"


def check_capture_interpreter(environment: dict, *, framework: Path) -> None:
    """Full visible-model checks run their child under the model2MLIR capture interpreter. Probe that
    interpreter with the export's own dependency list, so a missing package stops the plan here
    instead of refusing every kernel of every full model check during the run."""
    python = environment.get(CAPTURE_PYTHON_ENV)
    if not python:
        return
    probe = "from merlin_experiments.phase1.brokers.model_check import import_probe; import_probe()"
    selected = dict(os.environ, **environment)
    try:
        done = subprocess.run(
            [python, "-c", probe], cwd=framework, env=selected, capture_output=True, text=True, timeout=300
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise SpecError(f"{CAPTURE_PYTHON_ENV} {python} cannot be probed: {exc}") from exc
    if done.returncode:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        raise SpecError(
            f"{CAPTURE_PYTHON_ENV} {python} cannot run the full model-check child "
            f"({lines[-1] if lines else done.returncode}): install the missing package into that environment"
        )


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
        # Only invocation identity and writable runtime state get a new name.
        # A run-specific frozen release can itself contain RUN_ID in its path;
        # rewriting descriptor/seal/source/reference pins would select nonexistent
        # inputs and invalidate the exact plan being checked.
        positions = [i for i, item in enumerate(command) if item == "--run-id"]
        if len(positions) != 1 or command[positions[0] + 1] != name:
            raise SpecError("preflight requires the plan's exact run-id option")
        command[positions[0] + 1] = preflight
        for flag in ("--codex-home-root",):
            if flag in command:
                position = command.index(flag)
                command[position + 1] = command[position + 1].replace(name, preflight)
        writable = {"TMPDIR", "MERLIN_BUNDLE_CAS", "MERLIN_COMPUTE_CPU_SLOTS", "MERLIN_GSIM_SLOT_ROOT"}
        command = [
            key + "=" + value.replace(name, preflight)
            if separator and key in writable else item
            for item in command for key, separator, value in (item.partition("="),)
        ]
        command += ["--preflight-only"]
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


def prepared_plan(ctx: Context) -> dict:
    """Refuse a plan prepared from another spec or mode before execution."""
    plan = json.loads((ctx.path("${ARTIFACTS}") / "proposed-launch.json").read_text())
    expected = dict(
        path=str(ctx.spec_path),
        sha256=sha256(ctx.spec_path),
        extends=[dict(path=str(p), sha256=sha256(p)) for p in ctx.spec_sources[1:]],
        unit=ctx.unit,
    )
    selected_mode = plan.get("reference_guidance", "on" if plan.get("reference_experiments") else "off")
    if (
        plan.get("launch_spec") != expected
        or plan.get("run_id") != ctx.run_id
        or selected_mode != ctx.reference_guidance
    ):
        raise SpecError("prepared plan differs from this launch spec; run plan and preflight for the selected mode")
    return plan


def host_health_line() -> str:
    """One line for the operator: host conditions that degraded earlier runs. Never refuses a launch."""
    try:
        warnings = load_module("run_watchdog", HERE).host_warnings()
    except (OSError, ValueError, KeyError, StopIteration) as error:
        return f"HOST HEALTH unknown: {error}"
    return "HOST HEALTH WARNING: " + "; ".join(warnings) if warnings else "HOST HEALTH ok"


def drop_inputs(root: Path, members) -> int:
    """Remove the listed sandbox input copies under ``root``; returns the bytes their names held alone.

    A preflight's sandbox holds a full copy of the run's declared input closure (8.6 GB, 2.8 GB of
    it unique); its ``snapshot.json`` records every member's digest, so the copy itself is
    regenerable and no record cites its bytes. Read-only directories are plain mode bits here. Links
    are removed as names, never followed, and a member that leaves ``root`` is refused.
    """
    root, freed = Path(root), 0
    for member in members or ():
        target = root / member
        if ".." in Path(member).parts or Path(member).is_absolute():
            raise SpecError(f"preflight drop member must stay inside its state: {member}")
        if target.is_symlink() or not target.is_dir():
            continue
        relative = target.relative_to(root)
        for parent in [*reversed(relative.parents), relative]:
            if parent != Path("."):  # the state root keeps its own mode
                os.chmod(root / parent, 0o755)
        for directory, names, files in os.walk(target):
            for name in names:
                if not os.path.islink(os.path.join(directory, name)):
                    os.chmod(os.path.join(directory, name), 0o755)
            for name in files:
                stat = os.lstat(os.path.join(directory, name))
                freed += stat.st_size if stat.st_nlink == 1 else 0
        shutil.rmtree(target)
    return freed


def cmd_preflight(ctx: Context, args) -> int:
    section = ctx.section("preflight")
    plan = prepared_plan(ctx)
    print(host_health_line())
    if plan.get("seed") is not None:
        # The runtime's preflight-only mode refuses a candidate: check the seed itself, then
        # preflight the environment with the plan's unseeded twin.
        seed = load_module("fresh_seed", HERE)
        print(json.dumps({"seed_check": seed.verify(ctx, plan)}))
        plan = seed.preflight_plan(plan, run=args.run)
    script = ctx.path(section["run_script" if args.run else "script"])
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(preflight_script(plan, run=args.run))
    script.chmod(0o755)
    print(json.dumps({"script": str(script), "mode": "unreserved run" if args.run else "preflight"}))
    if not args.execute or args.run:
        return 0
    # A preflight starts from empty preflight state: earlier attempts move aside, never deleted, except
    # for the sandbox input copies an item lists under ``drop`` (see drop_inputs).
    retired = ctx.path(section["retire_into"]) / str(int(time.time()))
    for item in section.get("retire") or ():
        source = ctx.path(item["path"])
        if source.exists():
            retired.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(retired / item["as"]))
            drop_inputs(retired / item["as"], item.get("drop"))
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
        # A passed preflight is never resumed; a failed one keeps its inputs for diagnosis until retired.
        freed = sum(drop_inputs(ctx.path(item["path"]), item.get("drop")) for item in section.get("retire") or ())
        print(f"released {freed >> 20} MiB of preflight sandbox input copies")
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
    plan = prepared_plan(ctx)
    if plan.get("seed") is not None:
        print(json.dumps({"seed_check": load_module("fresh_seed", HERE).verify(ctx, plan)}))
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
    if not args.no_watchdog:
        watchdog = load_module("run_watchdog", HERE).start_detached(
            planned_unit(ctx), set(cpu_list(ctx.cpus)), ctx.path("${ARTIFACTS}") / "watchdog.log"
        )
        print("started run watchdog pid", watchdog, "(log", ctx.path("${ARTIFACTS}") / "watchdog.log)")
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
    common.add_argument(
        "--reference-guidance",
        choices=("off", "on"),
        default=None,
        help="enable lifting/CCA reference tools and derived guidance; default: the spec's mode",
    )
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("export", parents=[common], help="frozen framework export of the branch HEAD")
    sub.add_parser("inputs", parents=[common], help="prepare author library, full compiler and corpus")
    sub.add_parser("_inputs", parents=[common], help=argparse.SUPPRESS)
    sub.add_parser("ordering", parents=[common], help="derive the completion-order selection from the run's facts")
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
    launch.add_argument("--no-watchdog", action="store_true", help="do not start run_watchdog.py for the unit")
    check = sub.add_parser("check", parents=[common], help="read-only campaign status")
    check.add_argument("--watch", action="store_true", help="return once the campaign needs attention")
    check.add_argument("--until", metavar="HHMM", help="wait until this UTC time, then report")
    check.add_argument("--console", action="store_true", help="include the experiment console dump")
    return p


COMMANDS = {
    "export": cmd_export,
    "inputs": cmd_inputs,
    "_inputs": cmd_inputs_stage,
    "ordering": cmd_ordering,
    "seal": cmd_seal,
    "plan": cmd_plan,
    "preflight": cmd_preflight,
    "launch": cmd_launch,
    "check": cmd_check,
}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    ctx = Context(
        args.spec,
        overrides=args.var,
        run_id=args.run_id,
        unit=args.unit,
        cpus=args.cpus,
        export_id=args.export_id,
        reference_guidance=args.reference_guidance,
    )
    return COMMANDS[args.command](ctx, args)


if __name__ == "__main__":
    raise SystemExit(main())
