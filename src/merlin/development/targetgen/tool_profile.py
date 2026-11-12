"""Versioned common compiler facilities and immutable experiment tool policy.

Only the current common substrate is supported; a profile naming a retired one
is refused. Bundles without a profile select no common substrate. Grants name
exact sources; their frozen run snapshot supplies the bytes, not a mutable
installed package.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sysconfig
from pathlib import Path

from merlin.development.targetgen import tool_registry as TR


def _public_runtime_members(document: dict, resolve_member) -> set[str]:
    """Explicit selected OS runtime roles; never infer public grants from a prefix.

    This bounded role is for the ordinary dynamic loader and glibc ABI libraries
    already exposed by the sandbox's system runtime. It cannot publish compiler,
    proof, SDK or arbitrary system files. The selected manifest still pins them.
    """
    runtime = document.get("system_runtime_files", [])
    if not isinstance(runtime, list):
        raise ValueError("selected system_runtime_files must be a list")
    if not runtime:
        return set()
    files = document["files"]
    multiarch = sysconfig.get_config_var("MULTIARCH")
    if not isinstance(multiarch, str) or "/" in multiarch:
        raise ValueError("selected OS runtime requires a known canonical system ABI")
    names = {
        "libc.so.6",
        "libdl.so.2",
        "libm.so.6",
        "libpthread.so.0",
        "librt.so.1",
        "ld-linux-x86-64.so.2",
        "ld-linux-aarch64.so.1",
    }
    canonical = {
        (Path(root) / multiarch / name).resolve()
        for root in ("/lib", "/usr/lib")
        for name in names
        if (Path(root) / multiarch / name).is_file()
    }
    selected: set[str] = set()
    for row in runtime:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"} or row not in files:
            raise ValueError("selected runtime member must exactly match a pinned file member")
        path = resolve_member(row["path"])
        if path not in canonical or str(path) in selected:
            raise ValueError("selected runtime member is not a unique ordinary OS ABI library")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("selected runtime bytes differ from its manifest")
        if content[:4] != b"\x7fELF" or content[5:6] not in (b"\x01", b"\x02"):
            raise ValueError("selected runtime member must be an ELF shared object")
        order = "little" if content[5] == 1 else "big"
        if len(content) < 18 or int.from_bytes(content[16:18], order) != 3:
            raise ValueError("selected runtime member must be an ELF shared object")
        selected.add(str(path))
    return selected


SCHEMA = "merlin.compiler_tool_profile.v1"
# The single reviewed common compiler substrate; it binds the current shared SDK
# release. Earlier substrates are retired: a profile naming one is refused, and
# its frozen runs keep their archived environment.
CURRENT_COMMON_SUBSTRATE = "merlin.common_compiler_substrate.v20"
_SUBSTRATE_PREFIX = "merlin.common_compiler_substrate.v"
_PY = "merlin/python/merlin/"
COMMON_NAMESPACE_MODULES = (
    "merlin.__init__",
    "merlin.compiler.transforms.__init__",
    "merlin.verification.__init__",
    "merlin.compiler.semantic.__init__",
    "merlin.development.targetgen.__init__",
)
COMMON_PUBLIC_MODULES = (
    "merlin.compiler.transforms.certificates",
    "merlin.compiler.transforms.normalization",
    "merlin.compiler.transforms.loops",
    "merlin.compiler.transforms.planning",
    "merlin.compiler.transforms.validation",
    "merlin.compiler.transforms.pipeline",
    "merlin.compiler.transforms.client",
    "merlin.verification.lean_contract",
    "merlin.compiler.transforms.pure_scalar",
    "merlin.compiler.transforms.scalar_witness",
    "merlin.compiler.transforms.unrolling",
    "merlin.compiler.transforms.analysis_types",
    "merlin.compiler.transforms.analysis_memory",
    "merlin.compiler.transforms.analysis_liveness",
    "merlin.compiler.transforms.analyses",
    "merlin.compiler.transforms.resource_allocation",
    "merlin.compiler.semantic.allocation_types",
    "merlin.compiler.semantic.allocate",
    "merlin.compiler.semantic.rules",
    "merlin.compiler.semantic.model",
    "merlin.compiler.transforms.dense_planning",
    "merlin.compiler.transforms.dense_ir",
    "merlin.compiler.schedule.blocks",
)
AUTOMATIC_CHECKER_MODULES = (
    "merlin.verification.lean_export",
    "merlin.verification.lean_checker",
    "merlin.verification.lean_obligations",
    "merlin.analysis.perf.pass_observation",
    "merlin.verification.lean_resource_geometry",
    "merlin.verification.lean_state_effects",
    "merlin.verification.lean_state_projection",
    "merlin.verification.lean_checked",
    "merlin.verification.lean_emission_chain",
    "merlin.verification.lean_emission_context",
    "merlin.verification.lean_emission_products",
    "merlin.verification.lean_sat_runtime",
    "merlin.verification.lean_isa_semantics",
    "merlin.development.targetgen.isa_spec",
    "merlin.development.targetgen.isa_dialect_gen",
    "merlin.development.targetgen.compiler_profile_admission",
    "merlin.development.targetgen.tool_profile_toolchain",
    "merlin.development.targetgen.host_optimization_admission",
    "merlin.development.targetgen.common_transforms",
    "merlin.common.bundle_snapshot",
    "merlin.common.read_lease_digest",
    "merlin.development.targetgen.rtl.result_format_ceiling",
    "merlin.development.targetgen.rtl.result_format_validity",
    "merlin.common.bwrap_args",
    "merlin.compiler.lower.model.host_optimization_request",
    "merlin.compiler.lower.model.host_optimization_compile",
    "merlin.development.targetgen.rocc.semantics",
    "merlin.development.targetgen.rocc.normalized_operands",
    "merlin.development.targetgen.rocc.selected_operand_protocol",
    "merlin.development.targetgen.rtl.firrtl_operand_slices",
    "merlin.development.targetgen.rtl.firrtl_load_states",
    "merlin.development.targetgen.tool_profile_shared",
    "merlin.compiler.schedule.export",
)
AUTOMATIC_CHECKER_DATA = (
    "verification/lean/Merlin/Foundation.lean",
    "verification/lean/Merlin/ResourceGeometry.lean",
    "verification/lean/Merlin/StateEffects.lean",
)
# The proof domains of the current substrate (the formal coverage policy version).
_POLICY_VERSION = "3"
# The tool registry's compatibility path for the always-public formal coverage policy.
_LEAN_CONTRACT_GRANT = _PY + "verification/lean_contract.py"
_OPTIONAL_TOOLS = frozenset({"isa_dialect_generation", "lean_proof_authoring"})


def require_current_substrate(version) -> None:
    """Refuse every substrate other than the current one, naming a retired selection."""
    if version == CURRENT_COMMON_SUBSTRATE:
        return
    if isinstance(version, str) and version.startswith(_SUBSTRATE_PREFIX):
        raise ValueError(
            f"common compiler substrate {version} is retired; only {CURRENT_COMMON_SUBSTRATE} is supported"
        )
    raise ValueError("unknown common compiler substrate")


def substrate_inventory(version: str = CURRENT_COMMON_SUBSTRATE) -> tuple[tuple[str, ...], ...]:
    """Published source membership of the current substrate: namespaces, public and automatic modules."""
    require_current_substrate(version)
    from merlin.development.targetgen.tool_profile_shared import shared_inventory

    namespaces, public, _ = shared_inventory(COMMON_NAMESPACE_MODULES, COMMON_PUBLIC_MODULES)
    return namespaces, public, AUTOMATIC_CHECKER_MODULES


def substrate_data(version: str = CURRENT_COMMON_SUBSTRATE) -> tuple[str, ...]:
    require_current_substrate(version)
    return AUTOMATIC_CHECKER_DATA


def _digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _logical(module: str) -> str:
    return _PY + module.removeprefix("merlin.").replace(".", "/") + ".py"


def _source(root: Path, module: str) -> Path:
    selected = root / (module.replace(".", "/") + ".py")
    if selected.is_symlink() or not selected.is_file() or selected.resolve() != selected:
        raise ValueError(f"common compiler source is absent, linked or noncanonical: {module}")
    return selected


def _profile_inventory(profile: dict) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    from merlin.development.targetgen.tool_profile_shared import profile_inventory

    return profile_inventory(profile, COMMON_NAMESPACE_MODULES, COMMON_PUBLIC_MODULES)


def _implementation_members(profile: dict) -> tuple[str, ...]:
    """Exact private dependency extension of the selected release."""
    modules = profile.get("common_implementation_modules", [])
    if type(modules) is not list or modules != list(_profile_inventory(profile)[2]):
        raise ValueError("common compiler implementation dependency changed")
    return tuple(modules)


def common_tool_profile(
    arm: str, tools: tuple[str, ...], *, python_source_root: Path, version: str = CURRENT_COMMON_SUBSTRATE
) -> dict:
    """Bind actual common source bytes and host-owned formal coverage declarations."""
    from merlin.development.targetgen.tool_profile_shared import reviewed_shared_manifest
    from merlin.verification import lean_contract

    require_current_substrate(version)
    root = Path(python_source_root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("common substrate requires the explicit canonical selected Python source root")
    release = reviewed_shared_manifest(root)
    namespaces, public, implementation = _profile_inventory(
        {"common_substrate": version, "shared_scheduling_library": release}
    )
    modules = (*namespaces, *public, *implementation)
    sources = [
        {
            "module": module,
            "path": _logical(module),
            "sha256": hashlib.sha256(_source(root, module).read_bytes()).hexdigest(),
        }
        for module in modules
    ]
    automatic_sources = [
        {
            "module": module,
            "path": _logical(module),
            "sha256": hashlib.sha256(_source(root, module).read_bytes()).hexdigest(),
        }
        for module in AUTOMATIC_CHECKER_MODULES
    ]
    for relative in AUTOMATIC_CHECKER_DATA:
        source = root / "merlin" / relative
        if source.is_symlink() or not source.is_file() or source.resolve() != source:
            raise ValueError("automatic verifier data is absent, linked or noncanonical")
        automatic_sources.append(
            {"module": None, "path": _PY + relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    if (
        hashlib.sha256(Path(lean_contract.__file__).read_bytes()).hexdigest()
        != hashlib.sha256(_source(root, "merlin.verification.lean_contract").read_bytes()).hexdigest()
    ):
        raise ValueError("loaded formal coverage policy differs from the selected source release")
    policy = lean_contract.coverage_policy_record(version=_POLICY_VERSION)
    if policy.get("mode") != "strict_within_coverage":
        raise ValueError("new compiler substrate requires the host-owned strict formal coverage policy")
    record = {
        "schema": SCHEMA,
        "common_substrate": version,
        "arm": arm,
        "treatment_tools": list(tools),
        "common_public_modules": list(public),
        "automatic_checker_modules": list(AUTOMATIC_CHECKER_MODULES),
        "common_sources": sources,
        "automatic_checker_sources": automatic_sources,
        "proof_policy": copy.deepcopy(policy),
    }
    if implementation:
        record["common_implementation_modules"] = list(implementation)
    record["shared_scheduling_library"] = release
    record["shared_scheduling_resources"] = [
        {"path": str(root / name), "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
        for name in ("shared-library.json", "LICENSE", "NOTICE")
    ]
    return {**record, "sha256": _digest(record)}


def verify_tool_profile(bundle: dict, *, tools: tuple[str, ...] | None = None) -> dict | None:
    """Validate immutable inventory; a bundle without a profile selects no common substrate."""
    profile = bundle.get("compiler_tool_profile")
    if profile is None:
        if _OPTIONAL_TOOLS & set(bundle.get("tools") or ()) or bundle.get("arm") in {"merlin_isa", "merlin_lean"}:
            raise ValueError("EL5/EL6 require a versioned common compiler substrate")
        return None
    if not isinstance(profile, dict) or profile.get("schema") != SCHEMA:
        raise ValueError("unknown compiler tool profile schema")
    document = {key: value for key, value in profile.items() if key != "sha256"}
    if profile.get("sha256") != _digest(document):
        raise ValueError("compiler tool profile identity changed")
    require_current_substrate(profile.get("common_substrate"))
    namespaces, public, _ = _profile_inventory(profile)
    if profile.get("arm") != bundle.get("arm"):
        raise ValueError("compiler tool profile differs from its selected bundle")
    declared = profile.get("treatment_tools")
    if not isinstance(declared, list) or declared != bundle.get("tools"):
        raise ValueError("compiler tool inventory differs from its frozen profile")
    if tools is not None and list(tools) != declared:
        raise ValueError("launch tool choices differ from the immutable compiler tool profile")
    arm = str(bundle.get("arm"))
    expected = set(TR.ARM_TOOLS.get(arm, ())) & _OPTIONAL_TOOLS
    if set(declared) & _OPTIONAL_TOOLS != expected:
        raise ValueError("optional tool inventory contradicts its explicit experiment level")
    if profile.get("common_public_modules") != list(public):
        raise ValueError("common compiler inventory is incomplete or changed")
    if profile.get("automatic_checker_modules") != list(AUTOMATIC_CHECKER_MODULES):
        raise ValueError("automatic verification inventory changed")
    sources = profile.get("common_sources")
    expected_modules = (*namespaces, *public, *_implementation_members(profile))
    if not isinstance(sources, list) or [row.get("module") for row in sources if isinstance(row, dict)] != list(
        expected_modules
    ):
        raise ValueError("common compiler source membership changed")
    for row in sources:
        digest = row.get("sha256")
        if (
            row.get("path") != _logical(row["module"])
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("common compiler source identity is malformed")
    from merlin.development.targetgen.tool_profile_shared import validate_profile_release

    validate_profile_release(profile)
    automatic_sources = profile.get("automatic_checker_sources")
    expected_automatic = [(module, _logical(module)) for module in AUTOMATIC_CHECKER_MODULES] + [
        (None, _PY + relative) for relative in AUTOMATIC_CHECKER_DATA
    ]
    if (
        not isinstance(automatic_sources, list)
        or [(row.get("module"), row.get("path")) for row in automatic_sources if isinstance(row, dict)]
        != expected_automatic
    ):
        raise ValueError("automatic verifier source closure changed")
    for row in automatic_sources:
        digest = row.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("automatic verifier source identity is malformed")
    policy = profile.get("proof_policy") or {}
    from merlin.verification.lean_contract import coverage_policy_record

    expected_policy = coverage_policy_record(version=_POLICY_VERSION)
    if (
        policy.get("mode") != "strict_within_coverage"
        or policy.get("unknown_within_coverage") != "reject"
        or policy.get("outside_coverage") != "independent_checks_required"
        or policy != expected_policy
    ):
        raise ValueError("compiler tool profile weakens the formal preservation policy")
    toolchain = profile.get("lean_toolchain") or {}
    if toolchain.get("selection") not in {"not_selected", "explicit_manifest"} or toolchain.get("readiness") not in {
        "unavailable",
        "declared_unprobed",
    }:
        raise ValueError("compiler tool profile invents a formal toolchain qualification")
    width = profile.get("host_index_bits")
    if width is not None and (type(width) is not int or not 2 <= width <= 128):
        raise ValueError("compiler tool profile has an invalid reviewed host index width")
    from merlin.development.targetgen.public_codegen_budget import policy_from_profile

    policy_from_profile(profile)
    from merlin.verification.lean_isa_semantics import validate_semantic_inputs

    validate_semantic_inputs(profile.get("semantic_inputs"))
    return copy.deepcopy(profile)


def _withhold_author_packages(bundle: dict, python_source_root: Path) -> None:
    """Turn grants of author-inaccessible modules into denials; the SDK release omits them.

    Arm tool grants predate the withheld-package policy and name some operator files (the CCA spine)
    directly. The release does not stage those, and a grant of an absent file would fail the input
    snapshot, so the exact grant becomes a recorded denial. A directory grant stays, and every
    access-policy grader/oracle module or withheld package below it is recorded as a denial (deny-wins),
    so the policy is visible in the bundle even though the release omits the file. The generic analyses
    the remaining tools import are granted at the files the release stages for them.
    """
    from merlin.common.access import AUTHOR_WITHHELD_PREFIXES, MODULE_ACCESS, author_inaccessible
    from merlin.development.targetgen.tool_profile_shared import sdk_support_sources, staged_module

    selected = str(Path(python_source_root) / "merlin") + "/"
    kept, directories = [], []
    for row in bundle["allowed"]:
        path = row.get("path")
        prefix = next((p for p in (_PY, selected) if isinstance(path, str) and path.startswith(p)), None)
        relative = path[len(prefix) :] if prefix is not None else None
        if relative is not None and author_inaccessible(staged_module("merlin/" + relative)):
            bundle["denied"].append(
                {"path": path, "reason": "author-inaccessible module; reached only through its broker"}
            )
            continue
        kept.append(row)
        if relative is not None and relative.endswith("/"):
            directories.append((prefix, relative))
    bundle["allowed"] = kept
    origins = {name: "withheld package" for name in AUTHOR_WITHHELD_PREFIXES}
    origins.update((m, f"access-policy {e.origin} module") for e in MODULE_ACCESS for m in e.modules)
    source = Path(__file__).resolve().parents[2]  # this installed ``merlin`` package
    denied = {row.get("path") for row in bundle["denied"]}
    for name, why in origins.items():
        if not name.startswith("merlin."):
            continue
        stem = name.removeprefix("merlin.").replace(".", "/")
        relative = stem + "/" if (source / stem).is_dir() else stem + ".py"
        for prefix, directory in directories:
            if relative.startswith(directory) and prefix + relative not in denied:
                denied.add(prefix + relative)
                bundle["denied"].append({"path": prefix + relative, "reason": f"{why}; never an author input"})
    granted = {row.get("path") for row in kept}
    for relative in sdk_support_sources():
        if _PY + relative not in granted and selected + relative not in granted:
            bundle["allowed"].append(
                {"path": _PY + relative, "mode": "ro", "note": "generic analysis imported by granted tools"}
            )


def attach_common_substrate(
    bundle: dict,
    *,
    version: str,
    python_source_root: Path,
    lean_toolchain_manifest: Path | None = None,
    host_index_bits: int | None = None,
    software_spec_path: Path | None = None,
    facts_path: Path | None = None,
    target: str | None = None,
    public_codegen_resource_policy: dict | None = None,
) -> None:
    """Add reviewed common facilities while masking optional tools on every other level."""
    require_current_substrate(version)
    if host_index_bits is not None and (type(host_index_bits) is not int or not 2 <= host_index_bits <= 128):
        raise ValueError("reviewed host index width must be 2..128 bits or explicitly unknown")
    tools = tuple(bundle["tools"])
    profile = common_tool_profile(bundle["arm"], tools, python_source_root=python_source_root, version=version)
    namespaces, public, _ = _profile_inventory(profile)
    implementation = _implementation_members(profile)
    automatic = AUTOMATIC_CHECKER_MODULES
    # Deny-by-default isolation admits only these leaves. Parent denials would
    # also hide common leaves, so replace those historical blanket exclusions.
    bundle["denied"] = [row for row in bundle["denied"] if row.get("path") != "merlin/"]
    for module in (*namespaces, *public, *implementation):
        if any(
            row.get("path") in {_logical(module), str(_source(python_source_root, module))} for row in bundle["allowed"]
        ):
            continue
        bundle["allowed"].append({"path": _logical(module), "mode": "ro", "note": "common compiler infrastructure"})
    bundle["allowed"].extend(
        {"path": row["path"], "mode": "ro", "note": "shared compiler release identity and license"}
        for row in profile["shared_scheduling_resources"]
    )
    # Automatic verification is host-owned. These dependencies are sealed as
    # host inputs, never exposed by adding a general proof API to lower levels.
    if "lean_proof_authoring" not in tools:
        for module in automatic:
            if module in (*namespaces, *public, *implementation):
                continue
            if any(row.get("path") == str(_source(python_source_root, module)) for row in bundle["allowed"]):
                continue
            bundle["host_inputs"].append(
                {"path": str(_source(python_source_root, module)), "note": "host-owned automatic formal verification"}
            )
        bundle["host_inputs"].extend(
            {"path": str(python_source_root / "merlin" / relative), "note": "automatic formal verification lemmas"}
            for relative in AUTOMATIC_CHECKER_DATA
        )
    else:
        author_sources = [(_logical(module), _source(python_source_root, module)) for module in automatic]
        author_sources.extend(
            (_PY + relative, python_source_root / "merlin" / relative) for relative in AUTOMATIC_CHECKER_DATA
        )
        for logical, selected in author_sources:
            for row in bundle["allowed"]:
                if row.get("path") == logical:
                    row["path"] = str(selected)
            if not any(row.get("path") == str(selected) for row in bundle["allowed"]):
                bundle["allowed"].append(
                    {"path": str(selected), "mode": "ro", "note": "admitted formal author tool dependency"}
                )
    software_inputs = [
        row for row in bundle["host_inputs"] if row.get("note") == "selected software semantics; not hardware evidence"
    ]
    for row in software_inputs:
        bundle["host_inputs"].remove(row)
        path = str(software_spec_path) if software_spec_path is not None else row["path"]
        bundle["allowed"].append({"path": path, "mode": "ro", "note": "admitted software semantics"})
    public_grants = {_logical(module) for module in public}
    for tool_name in sorted(_OPTIONAL_TOOLS):
        tool = TR.spec(tool_name)
        if tool_name not in tools:
            bundle["denied"].extend(
                {"path": path, "reason": tool.deny_reason}
                for path in tool.bundle_paths
                if path != _LEAN_CONTRACT_GRANT and path not in public_grants
            )
    if "lean_proof_authoring" in tools:
        cli = str(_source(python_source_root, "merlin.verification.lean_cli"))
        bundle["allowed"].append({"path": cli, "mode": "ro", "note": "existing Lean author CLI implementation"})
    else:
        cli = str(python_source_root / "merlin/verification/lean_cli.py")
        bundle["denied"].append({"path": cli, "reason": "interactive Lean proof development requires EL6"})
    _withhold_author_packages(bundle, python_source_root)
    from merlin.development.targetgen.tool_profile_toolchain import attach_selected_toolchain

    toolchain = attach_selected_toolchain(bundle, tools, lean_toolchain_manifest)
    if lean_toolchain_manifest is not None:
        from merlin.development.targetgen.tool_profile_toolchain import require_packaged_mathematics

        require_packaged_mathematics(
            lean_toolchain_manifest,
            python_source_root,
            ("Foundation", "ResourceGeometry", "StateEffects"),
            mathematics_directory=python_source_root / "merlin" / Path(AUTOMATIC_CHECKER_DATA[0]).parent,
        )
    profile = {key: value for key, value in profile.items() if key != "sha256"}
    profile["lean_toolchain"] = toolchain
    profile["host_index_bits"] = host_index_bits
    if public_codegen_resource_policy is not None:
        from merlin.development.targetgen.public_codegen_budget import FIELD, validate_policy

        profile[FIELD] = validate_policy(public_codegen_resource_policy)
    from merlin.verification.lean_isa_semantics import select_isa_semantics

    profile["semantic_inputs"] = select_isa_semantics(
        software_spec_path=software_spec_path, facts_path=facts_path, target=target
    ).record()
    for key in ("software_spec", "rtl_facts"):
        selected = profile["semantic_inputs"][key]
        if selected is None:
            continue
        path = Path(selected["path"])
        covered = any(
            row.get("path") == str(path)
            or (str(row.get("path", "")).endswith("/") and path.is_relative_to(Path(row["path"])))
            for row in (*bundle["allowed"], *bundle["host_inputs"])
        )
        if not covered:
            bundle["host_inputs"].append(
                {"path": str(path), "note": "original formal semantic input; never a new agent grant"}
            )
    profile["sha256"] = _digest(profile)
    bundle["compiler_tool_profile"] = profile
    verify_tool_profile(bundle)


def tool_profile_prompt(bundle: dict) -> str:
    profile = verify_tool_profile(bundle)
    if profile is None:
        return ""
    from merlin.compiler.schedule.library_sources import CURRENT_SDK_VERSION

    sdk = CURRENT_SDK_VERSION
    lines = [
        "\n\n## Common compiler infrastructure and preservation policy\n",
        "The versioned common substrate is available in every experiment level: "
        + ", ".join(f"`{module}`" for module in profile["common_public_modules"])
        + ".",
        "Use its documented generic transformations and analyses. Their automatic verification "
        "is common infrastructure, not an additional level treatment.",
        "Use `merlin.compiler.transforms.client.apply_transform(workspace, source_ir, options)` through the "
        "original host broker for verified common transformations. Submit only IR and documented "
        "parameters; the host selects admitted semantics, formal coverage and the frozen proof toolchain.",
        "Within the admitted formal domain, successful Lean checking is required. Missing tools, "
        "unknown obligations, failed proofs and timeouts cannot discharge covered obligations. "
        "Outside that domain, independent correctness checks remain required and formal coverage "
        "must be reported explicitly. The admitted semantics and coverage policy are immutable.",
        "Tool presence describes an interface, not a successful runtime qualification. "
        "Report actual availability and proof outcomes separately from numerical and performance results.",
        'Use `merlin.compiler.transforms.client.check_transform(workspace, before_ir, after_ir, entry="...")` '
        "to submit actual compiler rewrites for covered host checking. The original host selects "
        "software/ISA semantics and effect premises from sealed inputs. A result establishes only "
        "its reported formal scope; hardware, numerical execution and performance need independent evidence.",
        "The original command-buffer schema accepts an optional source-bound "
        "`host_optimization_request`. Only bounded documented choices are permitted; "
        "the original reviewed software profile independently supplies any required host "
        "semantic and runtime permissions. Omission of those permissions leaves the request "
        "unavailable. Numerical, instruction-policy and complete execution gates still apply.",
        "During original compiler entrypoint calls, `MERLIN_COMMON_WORKSPACE` names "
        "the isolated request workspace for the same public transformation/check client. "
        "Replies establish only their reported scope.",
        "Shared dense scheduling is available as `merlin.compiler.transforms.dense_planning`: "
        "provide explicit primitive geometry, available storage after reservations, "
        "supported block limits and a target static-command price. The planner returns "
        "a bounded choice, not a numerical or execution proof. Emit and verify its "
        "`merlin.compiler.transforms.dense_ir.DenseScheduleOp` against the realized commands. "
        "Version 4 retains version 3's formal coverage policy.",
        "Shared `merlin.compiler.schedule.blocks` provides checked synchronous "
        "matmul/convolution schedules. Supply explicit fact-derived Geometry; "
        "check_residency verifies operand contents, ordered reductions, accumulator "
        "lifetimes and complete output stores. It does not establish asynchronous "
        "completion or external view aliasing. Project emitted operations explicitly; "
        "load counts alone cannot establish lifetime conformance. Version 5 keeps "
        "version 4's automatic checkers and formal coverage policy.",
        f"Every V{sdk} treatment receives "
        "the same shared scheduling SDK: "
        "tiling and reuse, "
        "temporary storage, supported fusion/copy elimination, persistent configuration, "
        "device spills, complete row reductions and final program checks. Phase 2 may "
        "edit reusable shape/resource-based policy code to select legal alternatives. "
        "Keep numerical policy, hardware limits, qualification identities and verification "
        "gates fixed; checker work limits are not performance objectives. Check actual "
        "decoded output and complete outputs/state before performance comparisons. "
        "The exact original Phase 1 release is immutable; older runs cannot acquire "
        f"these grants. V{sdk} also makes generic ISA parser/generator "
        "algorithms common "
        "infrastructure, so historical ISA-tool ablations need a new interpretation. "
        "No target facts, instruction meanings, execution qualification or runtime tools "
        "follow from SDK availability. Missing capabilities remain unavailable.",
        "Discover current executable scheduling APIs with "
        f"`merlin.compiler.schedule.api_catalog.scheduling_capabilities({sdk})` or "
        "the selected `shared-library.json` scheduling_capabilities entry. "
        "It lists segmented-K issue ordering, exact final wide-load composition, "
        "array-weight reuse and source-axis loop transforms with actual symbols "
        "and required proofs. Emit the checked actual kernel; checking projections "
        "and discovery metadata supply no target execution admission.",
        "V19 additionally provides a source-checked final-K producer/readout frontier on the original "
        "full accumulator view. It requires a separate target RAW retirement and pending-write read "
        "readiness capability, with original store snapshots/backpressure and a final full drain. "
        "It supplies no fixed-cycle spacing or loop-carried event ownership.",
        "V22 keeps every V21 scheduling API, including loop-carried completion events and "
        "effect-checked schedule composition. Operator-only kernel packages and access-policy "
        "grader/oracle modules are not part of this SDK; the CCA calls remain available "
        "through their staged broker clients.",
        "V24 additionally exposes source-checked shared-RHS sibling products through "
        "the scheduling dialect and the ordinary registered-pass A/B route. Bind original "
        "member operands and all outputs, immutable storage, readout contracts and selected "
        "ordering laws. Missing premises retain drains or decline; recheck emitted effects "
        "and complete numerical results before comparing whole-call cycles.",
        f"V{sdk} uses canonical compiler.schedule, compiler.ir and compiler.selection owners. "
        "Earlier SDK source selections remain immutable and require their archived environment; "
        "this namespace migration grants no new target semantics or acceptance authority.",
    ]
    for name in profile["treatment_tools"]:
        if name in _OPTIONAL_TOOLS:
            lines.append(f"- `{name}`: {TR.spec(name).blurb}")
    return "\n".join(lines) + "\n"
