"""Explicit author environments for a new study; historical experiment levels stay unchanged.

The judge's compiler_tool_profile remains a host-owned execution input. This separate
selection controls supplied author sources, advice and broker capabilities. It never
changes numerical, ISA, timing or privacy gates and never forbids independently authored code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

ENV = "MERLIN_AUTHOR_PROFILE"
SCHEMA = "merlin.author_profile.v1"
SCRATCH_NAMES = tuple("xs-el-v1/el" + str(level) for level in range(1, 5))
SEEDED_NAMES = ("xs-f6-v1/inherited", "xs-f6-v1/dense-epochs")
NO_EXPERT_NAMES = tuple("xs-el-no-expert-v1/el" + str(level) for level in range(1, 4))
from . import fresh10_profiles as F10

NAMES = (*SCRATCH_NAMES, *SEEDED_NAMES, *F10.NAMES, *F10.COVERAGE_NAMES, *NO_EXPERT_NAMES)
RECORD = "AUTHOR_PROFILE.json"
CPP_ENTRIES = tuple("merlin.targetgen.generate." + name for name in ("mlir_scaffold", "llvm_plan", "target_repo"))
LIGHT_ENTRIES = (*CPP_ENTRIES, "merlin.targetgen.generate.xdsl", *(
    "merlin.targetgen.oot_starterkit." + name
    for name in ("iface", "dialect", "llvm_context", "cmdbuf", "verify")))
# Exact reviewed support modules, not compiler-directory grants. Component preparation
# computes dependencies first, then fails if a future edit crosses this copy boundary.
LIGHT_SUPPORT = frozenset((
    "merlin", "merlin.common", "merlin.common.access", "merlin.common.artifacts",
    "merlin.common.paths", "merlin.common.yaml", "merlin.common.ir_lock",
    "merlin.common.dense_float_ir", "merlin.common.stimulus", "merlin.targetgen",
    "merlin.targetgen.generate", "merlin.targetgen.generate.typed_mlir",
    "merlin.targetgen.oot_starterkit", "merlin.targetgen.contract",
    "merlin.targetgen.contract.interface_emit", "merlin.targetgen.contract.linalg_iface",
    "merlin.compile", "merlin.compile.logical_abi", "merlin.runtime",
    "merlin.runtime.commandbuffer", "merlin.runtime.tensor", "merlin.xdsl_dialects",
    "merlin.xdsl_dialects._common", "merlin.xdsl_dialects.fp8", "merlin.xdsl_dialects.interface",
))
CPP_SUPPORT = frozenset((
    "merlin", "merlin.common", "merlin.common.access", "merlin.common.artifacts",
    "merlin.common.paths", "merlin.common.yaml", "merlin.targetgen",
    "merlin.targetgen.generate", "merlin.targetgen.generate.typed_mlir",
))


@dataclass(frozen=True)
class AuthorProfile:
    name: str
    level: int

    @property
    def seeded(self):
        return self.name in SEEDED_NAMES or self.coverage

    @property
    def fresh10(self):
        return self.name in F10.NAMES or self.coverage

    @property
    def coverage(self):
        """A seeded fresh10 continuation on the current reviewed compiler (CCA coverage study)."""
        return self.name in F10.COVERAGE_NAMES

    @property
    def raw_kernel_references(self):
        if self.name in NO_EXPERT_NAMES:
            return False
        return F10.resolve(self.name).raw_kernel_references if self.fresh10 else True

    def allows(self, capability):
        if self.fresh10:
            return F10.resolve(self.name).allows(capability)
        return self.modern

    @property
    def entries(self):
        if self.coverage:
            return None  # The complete current reviewed public compiler, as an ordinary seeded run.
        if self.fresh10:
            from . import fresh10_sources
            return fresh10_sources.entry_modules(self.name)
        return () if self.level == 1 else CPP_ENTRIES if self.level == 2 else LIGHT_ENTRIES if self.level == 3 else None

    @property
    def tools(self):
        if self.fresh10:
            return F10.TOOLS
        return (("isa_tools",) if self.seeded else ("isa_tools", "cca_tools") if self.level == 4
                else ("cpp_oot_generators",) if self.level > 1 else ())

    @property
    def modern(self):
        return self.level == 4 and not self.fresh10

    def record(self):
        if self.fresh10:
            return dict(schema=SCHEMA, name=self.name, level=self.level,
                        source_scope=("current_reviewed_full_public" if self.coverage
                                      else "reviewed_fresh10_capability_removal"),
                        baseline_source_revision=None if self.coverage else F10.BASE_REVISION,
                        entry_modules=None if self.entries is None else list(self.entries), tools=list(self.tools),
                        capabilities=F10.resolve(self.name).record(),
                        optimization_guidance=self.allows("catalog_advice"), live_cca=False,
                        prior_reference_tasks=False, historical_levels_unchanged=True,
                        **({"seeded": True, "raw_reference_sha256": sorted(F10.COVERAGE_RAW_REFERENCE_SHA256)}
                           if self.coverage else {}))
        return dict(schema=SCHEMA, name=self.name, level=self.level,
                    source_scope="explicit_inherited_closure" if self.seeded else "none" if self.level == 1 else "full_public" if self.modern else "component_closure",
                    entry_modules=None if self.entries is None else list(self.entries), tools=list(self.tools),
                    optimization_guidance=self.modern, live_cca=self.modern, prior_reference_tasks=False,
                    historical_levels_unchanged=True,
                    **({"raw_kernel_references": False, "expert_kernel_policy": "withheld"}
                       if self.name in NO_EXPERT_NAMES else {}))


def resolve(name):
    if not name:
        return None
    if name not in NAMES:
        raise ValueError("unknown versioned author profile: " + str(name))
    levels = dict(zip((*SCRATCH_NAMES, *SEEDED_NAMES), range(1, 7), strict=True))
    levels.update(zip(NO_EXPERT_NAMES, range(1, 4), strict=True))
    return AuthorProfile(name, 4 if name in (*F10.NAMES, *F10.COVERAGE_NAMES) else levels[name])


def current():
    """Host invocation restriction, inherited by its broker children; grants remain frozen bytes."""
    return resolve(os.environ.get(ENV, ""))


def modern_advice(capability="catalog_advice"):
    profile = current()
    return profile is None or profile.allows(capability)


def validate_options(options):
    profile = resolve(getattr(options, "author_profile", ""))
    if profile is None:
        return None
    if profile.fresh10:
        F10.require_preparable(profile.name)
    if options.sandbox != "bwrap" or options.compiler_execution != "snapshot_external_v1":
        raise ValueError("versioned author profiles require isolated snapshot execution")
    if profile.seeded:
        if not (options.seed_submission or options.preflight_only):
            raise ValueError("seeded author profile requires an explicit reviewed seed or empty preflight twin")
    elif not options.fresh_backend or options.seed_submission or options.qualify_submission or options.joint_epoch:
        raise ValueError("xs-el-v1 profiles require a fresh empty target backend")
    if options.with_tool or options.without_tool:
        raise ValueError("versioned author profiles cannot add or remove undeclared tools")
    if profile.level == 1 and (options.compiler_access == "full_compiler" or options.full_compiler_root):
        raise ValueError("EL1 supplies no Merlin compiler source selection")
    if profile.level > 1 and options.compiler_access != "full_compiler":
        raise ValueError("EL2-4 require their exact reviewed compiler source selection")
    if not profile.allows("reference_analysis") and options.reference_experiments:
        raise ValueError("reference analysis is not provided by this author profile")
    if profile.allows("reference_analysis") and (not options.reference_experiments or options.reference_experiments_tasks != "off"):
        raise ValueError("EL4 requires live CCA with prior reference tasks withheld")
    return profile


def verify_sources(profile, selection, *, bundle=None):
    from . import full_compiler_workspace as full

    if isinstance(profile, str):
        profile = resolve(profile)
    if profile.level == 1:
        if selection is not None:
            raise ValueError("EL1 supplies no Merlin source implementations")
        return
    if type(selection) is not full.FullCompilerSelection:
        raise ValueError("author profile requires its original source selection")
    selection.verify()
    document = json.loads(selection.manifest_path.read_bytes())
    if bundle is not None and bundle.get("author_profile_sources") != selection.record():
        raise ValueError("selected compiler sources differ from the sealed author profile")
    if profile.coverage:
        # The current reviewed full public compiler, prepared for exactly this profile.
        if document.get("author_profile") != profile.record():
            raise ValueError("compiler inventory belongs to another author profile")
        return
    if profile.seeded:
        return  # The sealed explicit source pin, not a new SDK version, binds this inherited closure.
    if document.get("author_profile") != profile.record():
        raise ValueError("compiler inventory belongs to another author profile")
    if profile.fresh10:
        from . import fresh10_sources
        fresh10_sources.verify_sources(profile.name, selection)
        return
    if profile.entries is not None:
        if document.get("entry_modules") != list(profile.entries):
            raise ValueError("component entry modules differ from the author profile")
        allowed = (CPP_SUPPORT | set(CPP_ENTRIES)) if profile.level == 2 else (LIGHT_SUPPORT | set(LIGHT_ENTRIES))
        extra = [path for path, _digest, module in selection.members if module not in allowed]
        if extra or selection.diagnostics:
            raise ValueError("component dependency closure exceeds the reviewed profile: " + repr(extra[:8]))


def prepare_sources(repo, destination, *, profile, review_id, source_selection=None):
    """Create only the selected public source closure; no target backend or legacy SDK."""
    from . import full_compiler_workspace as full

    profile = resolve(profile) if isinstance(profile, str) else profile
    if profile is None:
        raise ValueError("source preparation requires an explicit author profile")
    if profile.seeded and not profile.coverage:
        if isinstance(source_selection, dict):
            source_selection = full.select(**{key: source_selection[key]
                for key in ("root", "manifest_path", "manifest_sha256")})
        verify_sources(profile, source_selection)
        return source_selection
    if source_selection is not None:
        raise ValueError("scratch author profile cannot select an inherited source tree")
    if profile.level == 1:
        return None
    if profile.coverage:
        selection = full.prepare(repo, destination, review_id=review_id)
    elif profile.fresh10:
        from . import fresh10_sources
        selection = fresh10_sources.prepare_sources(repo, destination, profile.name, review_id)
    else:
        selection = full.prepare(repo, destination, review_id=review_id, modules=profile.entries,
                             include_runtime_support=profile.modern,
                             extra_modules=LIGHT_ENTRIES if profile.modern else ())
    manifest = selection.manifest_path
    document = json.loads(manifest.read_bytes())
    document["author_profile"] = profile.record()
    manifest.chmod(0o644)
    manifest.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n")
    manifest.chmod(0o444)
    selection = full.select(selection.root, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())
    verify_sources(profile, selection)
    return selection


def _source_package(path):
    """Recognize supplied Python framework source, including SDK copies outside a checkout."""
    path = Path(path)
    if path.suffix in {".pth", ".whl"}:
        return True
    for parent in (path, *path.parents):
        if parent.name in {"merlin", "merlin_experiments"} and (parent / "__init__.py").is_file():
            return True
    if path.is_dir():
        return any((path / name / "__init__.py").is_file() for name in ("merlin", "merlin_experiments"))
    return False


_EXPERT_REFERENCE_COMPONENTS = frozenset({
    "reference-kernels", "public-references", "reference-map.json", "reference_map.json",
    "reference-manifest.json", "reference-experiments", "reference-analysis",
    "reference-tasks", "reference_task_cards", "cca-artifacts", "cca-reports",
    "performance-baselines", "performance_baselines", "expert-targets", "expert_targets",
    "seed_performance",
})


def expert_reference_grant(row):
    """Recognize supplied reference containers and every declared/resolved alias.

    A run ID containing ``no-reference-kernels`` is not itself a reference bank.
    Ordinary workload semantics and an author's newly emitted kernels are retained.
    """
    for value in (row.get("path", ""), row.get("as", "")):
        if not value:
            continue
        path = Path(value)
        variants = (path, path.resolve()) if path.is_absolute() else (path,)
        if any({p.casefold() for p in candidate.parts} & _EXPERT_REFERENCE_COMPONENTS
               for candidate in variants):
            return True
        if path.is_absolute() and path.is_dir() and (path / "reference-manifest.json").is_file():
            return True
    return False


def _verify_no_expert_grants(bundle):
    allowed = bundle.get("allowed", ())
    if any(expert_reference_grant(row) for row in allowed):
        raise ValueError("no-expert author profile exposes supplied reference material")
    withheld = [Path(row["path"]).resolve() for row in bundle.get("host_inputs", ())
                if row.get("path") and expert_reference_grant(row)]
    for row in allowed:
        public = Path(row["path"]).resolve()
        if any(public == secret or public in secret.parents for secret in withheld):
            raise ValueError("no-expert author profile exposes a reference bank through an ancestor grant")


_REFERENCE_RECORDS = frozenset(("reference-manifest.json", "reference-selection.json"))


def _verify_coverage_references(bundle, *, content=True):
    """Grant exactly the coverage study's pinned raw executables, read-only, and nothing derived.

    Every supplied reference alias (bank, map, analysis, CCA product) counts as a grant, so a
    second one refuses. The one bank holds the pinned original bytes plus its staged records.
    """
    rows = [row for row in bundle.get("allowed", ()) if expert_reference_grant(row)]
    if len(rows) != 1 or rows[0].get("as") != "reference-kernels" or rows[0].get("mode") != "ro":
        raise ValueError("coverage profile requires exactly one read-only raw reference grant")
    root = Path(rows[0]["path"])
    root = root if root.is_absolute() else Path.cwd() / root
    if not root.is_dir():
        if content:
            raise ValueError("coverage raw reference grant is unavailable for verification: " + str(root))
        return
    expected = sorted(F10.COVERAGE_RAW_REFERENCE_SHA256)
    found = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("coverage raw reference grant contains an indirect path")
        if path.is_file() and path.relative_to(root).as_posix() not in _REFERENCE_RECORDS:
            found.append(hashlib.sha256(path.read_bytes()).hexdigest())
    manifest = json.loads((root / "reference-manifest.json").read_bytes())
    listed = sorted(member["sha256"] for origin in manifest.get("origins", ()) for member in origin.get("files", ()))
    if sorted(found) != expected or listed != expected:
        raise ValueError("coverage reference grant differs from the pinned raw executables")


def materialize_bundle(bundle, *, profile, common_allowed, source_selection=None):
    """Keep common public data explicit; retain old SDK leaves as host-only judge inputs.

    The operator passes the complete reviewed common-data allowlist. Inferring it by
    deleting one package prefix could leave SDK copies or directory grants exposed.
    The separate selected source tree is copied into the candidate, never mounted here.
    """
    profile = resolve(profile) if isinstance(profile, str) else profile
    verify_sources(profile, source_selection)
    from merlin.targetgen.tool_profile import verify_tool_profile

    verify_tool_profile(bundle)
    if not isinstance(common_allowed, (list, tuple)) or not common_allowed:
        raise ValueError("profile preparation requires an explicit common input allowlist")
    if any(not isinstance(row, dict) or not row.get("path") or _source_package(row["path"])
           for row in common_allowed):
        raise ValueError("common inputs cannot expose compiler SDK packages")
    selected = copy.deepcopy(bundle)
    selected["allowed"] = copy.deepcopy(list(common_allowed))
    selected["tools"] = list(profile.tools)
    selected["author_profile"] = profile.record()
    selected["author_profile_sources"] = None if source_selection is None else source_selection.record()
    hosts = selected.setdefault("host_inputs", [])
    existing = {row["path"] for row in hosts}
    public = {row["path"] for row in common_allowed}
    for row in bundle.get("allowed", ()):
        if row["path"] not in public and row["path"] not in existing:
            hosts.append(dict(path=row["path"], note="judge-only original input; withheld by author profile"))
            existing.add(row["path"])
    judge = selected.get("compiler_tool_profile")
    if judge is None:
        raise ValueError("versioned author profiles require an independent versioned judge profile")
    judge["treatment_tools"] = list(profile.tools)
    judge["sha256"] = hashlib.sha256(json.dumps(
        {key: value for key, value in judge.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    verify_tool_profile(selected, tools=profile.tools)
    if not profile.raw_kernel_references:
        _verify_no_expert_grants(selected)
    if profile.coverage:
        _verify_coverage_references(selected)
    return selected


def verify_bundle(profile, bundle):
    profile = resolve(profile) if isinstance(profile, str) else profile
    if bundle.get("author_profile") != profile.record() or tuple(bundle.get("tools", ())) != profile.tools:
        raise ValueError("bundle author profile differs from its original launch selection")
    if any(_source_package(row["path"]) for row in bundle.get("allowed", ())):
        raise ValueError("profile bundle exposes framework source outside its selected inventory")
    if not profile.raw_kernel_references:
        _verify_no_expert_grants(bundle)
    if profile.coverage:
        # Content was pinned when the release was prepared; a sealed bundle may be reopened elsewhere.
        _verify_coverage_references(bundle, content=False)


def check_broker_action(action):
    profile = current()
    if profile is not None and profile.fresh10:
        if not F10.broker_allowed(profile.name, action):
            raise FileNotFoundError("service is withheld by the selected fresh10 capability profile")
        return
    if not modern_advice() and action in {"common-transform", "common-check", "reference-analysis",
                                         "candidate-diagnostics"}:
        raise FileNotFoundError("analysis/optimization assistance is withheld by the selected author profile")


def stage_record(workspace, run_dir, profile, selection):
    verify_sources(profile, selection)
    document = {**profile.record(), "source_selection": None if selection is None else selection.record()}
    data = json.dumps(document, sort_keys=True, indent=2) + "\n"
    for root in (Path(workspace), Path(run_dir)):
        path = root / RECORD
        if path.exists() or path.is_symlink():
            if path.is_symlink() or path.read_text() != data:
                raise ValueError("author profile staging changed")
        else:
            path.write_text(data)
            path.chmod(0o444)
    return document


def task(profile):
    if profile.fresh10:
        from . import fresh10_disclosure
        return fresh10_disclosure.capability_text(profile.name)
    language = "C++/MLIR" if profile.level < 3 else "Python/xDSL or C++/MLIR"
    supplied = ("Keep the complete explicitly pinned inherited compiler and its library. "
                "Only the separately selected bounded consumer differs across the matched pair; "
                "current SDK refreshes, current catalogs and CCA are not supplied." if profile.seeded else {
        1: "Ordinary toolchains and public contracts are supplied. Implement the compiler without Merlin authoring tools.",
        2: "The selected compiler_library supplies C++ OOT scaffolding generators and their support dependencies only.",
        3: "The selected compiler_library supplies lightweight Python/xDSL dialect, parser, verifier and emission plumbing, "
           "including its generic logical-ABI, command-buffer and tensor/numerical support dependencies. "
           "Author all target lowerings, transformations and scheduling policies yourself.",
        4: "The complete reviewed public Merlin compiler sources, scheduling dialect, pass library and analysis tools "
           "are available. Use live reference analysis where useful; no prior generated task cards are supplied.",
    }[profile.level])
    origin = ("Continue the supplied reviewed compiler and preserve its working behavior. " if profile.seeded
              else "Start with an empty target backend. ")
    return ("# Compiler construction ablation\n\n"
            "Build one general compiler in submission/ for every selected public xs capsule. "
            + origin + "Preserve the admitted source semantics, precision, "
            "fusion boundaries and every live output. Implement the manifest and required entrypoints "
            "from the common backend contract.\n\n"
            + supplied + "\nLanguage/tooling environment: " + language + ".\n"
            + ("Declare language: cpp and a real C++/MLIR build block in manifest.yaml.\n" if profile.level < 3 else "")
            + ("Read AUTHOR_PROFILE.json for the exact API inventory. Import supplied components with "
               "submission/compiler_library on the Python module search path.\n" if profile.level in (2, 3) else "")
            + ("The source-only components have no packaged schema copy. Select the already supplied public "
               "schemas/command_buffer.schema.json explicitly with the schema_path argument to "
               "CommandBufferBuilder or verify.validate. You can include an identical copy of that public "
               "data file in the compiler package; workspace aliases are not candidate execution paths.\n"
               if profile.level == 3 else "")
            + "\n"
            + ("Expert kernel sources, ELFs, disassemblies, reference maps and CCA-derived artifacts "
               "are not supplied. Author all target kernel implementations yourself.\n"
               if not profile.raw_kernel_references else "")
            + ("Raw hardware and ordering evidence and the independent numerical/ISA/timing judge "
               if not profile.raw_kernel_references else
               "Raw hardware, ordering and reference evidence and the independent numerical/ISA/timing judge ")
            + "are common across profiles. Use only the granted inputs; hidden data and prior implementations "
            "are withheld. Production decisions derive from semantics, shapes and hardware facts, never case names.\n\n"
            "Use python agent_selfcheck.py for focused numerical and timing checks, immutable variants for "
            "alternatives and the supplied request/attach interfaces. Check qa/verdict.json source identities. "
            "A model estimate is provisional; unknown checks remain unknown. Keep working within the selected "
            "clock and helper/CPU limits. Freeze the compiler before final offline evaluation.\n")


def _mounted_python_packages(argv, names):
    """Find conventional package trees only beneath the actual bound runtimes.

    An alternate venv has its own site-packages even when the manager imports
    another installation. Never scan unrelated host environments or execute
    their Python/path hooks. The ordinary mask owner projects these sources
    through every bind alias and skips destinations already hidden.
    """
    from merlin.targetgen.sandbox import bwrap as BW

    found = set()
    for state, source, _destination in BW._mounts(argv):
        if state != "expose":
            continue
        source = Path(source).absolute()
        root = source if source.is_dir() else source.parent.parent if source.parent.name == "bin" else None
        if root is None:
            continue
        environments = {root, root / ".venv", root / "venv"}
        # The ordinary Python toolchain can bind the selected uv runtime store.
        for directory in (root / "python", root / "tools"):
            if directory.is_dir():
                environments.update(path for path in directory.iterdir() if path.is_dir())
        search = {root} if root.name in {"site-packages", "dist-packages"} else set()
        if root.name in names and root.parent.name in {"site-packages", "dist-packages"}:
            search.add(root.parent)
        for environment in environments:
            search.add(environment / "Lib/site-packages")
            for library in ("lib", "lib64"):
                for python in (environment / library).glob("python*"):
                    if python.is_dir():
                        search.update((python / "site-packages", python / "dist-packages"))
        for directory in search:
            for name in names:
                for path in (directory / name, directory / (name + ".py")):
                    if path.is_dir() or path.is_file():
                        found.update((path.absolute(), path.resolve()))
    return tuple(sorted(found))


def runtime_masks(argv, bundle):
    """Withhold ambient framework packages; keep only the staged compiler copy.

    Applied after all author/provider/toolchain binds, also for evaluated compilers.
    Prefixes come from actual installed package search locations, never guessed paths.
    The ordinary mask owner projects these through bind aliases as well.
    """
    import importlib.util

    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    selected = bundle.get("author_profile")
    if selected is None:
        return argv
    profile = resolve(selected.get("name"))
    verify_bundle(profile, bundle)
    names = ("merlin", "merlin_experiments", *(("xdsl",) if profile.level < 3 else ()))
    surfaces = []
    for name in names:
        spec = importlib.util.find_spec(name)
        if spec is None:
            continue
        roots = spec.submodule_search_locations or (() if spec.origin is None else (spec.origin,))
        # Extended namespace paths include optional distribution roots.
        import sys
        roots = getattr(sys.modules.get(name), "__path__", roots)
        for root in roots:
            path = Path(root).resolve()
            if path.exists():
                surfaces.append(AnswerSurface("author profile withheld runtime package", path,
                                              "dir" if path.is_dir() else "file", "hidden"))
    for path in _mounted_python_packages(argv, names):
        surfaces.append(AnswerSurface("author profile withheld bound-runtime package", path,
                                      "dir" if path.is_dir() else "file", "hidden"))
    return BW.apply_answer_masks(argv, surfaces)


def canonical_common(bundle, te, variant):
    """Explicit study selection of the generator's common information set.

    Use the release-local descriptor, then take actual generated rows so ordinary
    LLVM/facts/public-input relocation stays authoritative. No SDK-directory guess.
    """
    from merlin.targetgen.generate_bundles import _shared_allow

    legacy_task = te.experiment_resource("task").rstrip("/")
    wanted = {row["path"] for row in _shared_allow(te, variant)
              if row["path"].rstrip("/") != legacy_task}
    facts = bundle.get("selected_rtl_facts_file")
    facts_root = str(Path(facts).parent) if facts else None
    software = ((bundle.get("compiler_tool_profile") or {}).get("semantic_inputs") or {}).get("software_spec")
    chosen, found = [], set()
    for row in bundle["allowed"]:
        path = row["path"]
        direct = path in wanted
        llvm = row.get("note") == "LLVM/MLIR 23 toolchain"
        public = row.get("note") == "explicit public input"
        raw_facts = facts_root is not None and path.rstrip("/") == facts_root
        # The source owner already validated these exact pinned OS ABI files;
        # retaining their public role adds no compiler or formal authoring API.
        runtime = row.get("note") == "selected ordinary OS runtime; not formal authoring capability"
        semantics = row.get("note") == "admitted software semantics"
        if semantics and (not isinstance(software, dict) or path != software.get("path")
                or hashlib.sha256(Path(path).read_bytes()).hexdigest() != software.get("sha256")):
            raise ValueError("common software semantics differ from the exact admitted declaration")
        if direct or llvm or public or raw_facts or runtime or semantics:
            chosen.append(copy.deepcopy(row))
            if direct:
                found.add(path)
            if llvm:
                found.add("third_party/llvm-install/")
    missing = wanted - found
    if missing:
        raise ValueError("canonical common grants were not materialized: " + repr(sorted(missing)[:8]))
    return chosen


def stage_release_bundles(experiment, te, *, profile, source_selection, common_inputs):
    """Finish a newly generated, unsealed release through its existing scaffold owner."""
    import yaml

    profile = resolve(profile)
    if profile.fresh10:
        if common_inputs != "fresh10_common":
            raise ValueError("fresh10 profiles require their reviewed fresh10 common input selection")
    elif common_inputs != "canonical_common":
        raise ValueError("author profiles require explicit canonical_common input selection")
    records = []
    for path in sorted((Path(experiment) / "input_bundles").glob("*/input_bundle_manifest.yaml")):
        base = yaml.safe_load(path.read_text())
        if profile.fresh10:
            from . import fresh10_sources
            common = fresh10_sources.common_allowed(base, profile.name)
        else:
            common = canonical_common(base, te, base.get("variant", "public_v0"))
            if not profile.raw_kernel_references:
                common = [row for row in common if not expert_reference_grant(row)]
        selected = materialize_bundle(base, profile=profile, common_allowed=common,
                                      source_selection=source_selection)
        path.write_text(yaml.safe_dump(selected, sort_keys=False))
        directory = path.parent
        for key, name in (("allowed", "allowed_files.txt"), ("denied", "denied_files.txt")):
            (directory / name).write_text("".join(row["path"] + "\n" for row in selected[key]))
        (directory / "tools.txt").write_text("".join(name + "\n" for name in profile.tools))
        (directory / "STARTER_PROMPT.md").write_text(task(profile))
        (directory / "ALLOWED_MERLIN_TOOLS.md").write_text(task(profile))
        if profile.fresh10 or profile.name in NO_EXPERT_NAMES:
            # This independent copied document is discoverable even when the
            # full-model TASK path does not append it. Do not retain old CCA or
            # full-SDK advertisements outside the selected TASK disclosure.
            (directory / "TASK_ADDENDUM.md").write_text(task(profile))
        records.append(dict(bundle_id=selected["bundle_id"], profile=profile.record(),
                            common_inputs=common_inputs, public_paths=[row["path"] for row in common]))
    if not records:
        raise ValueError("profile release has no newly generated bundles")
    return records
