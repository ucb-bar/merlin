"""Versioned common compiler facilities and immutable experiment tool policy.

Historical manifests have no profile and retain their original interpretation.
Only newly prepared releases select this substrate. Grants name exact sources;
their frozen run snapshot supplies the bytes, not a mutable installed package.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import sysconfig
from pathlib import Path

from . import tool_registry as TR


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
COMMON_SUBSTRATE_V1 = "merlin.common_compiler_substrate.v1"
COMMON_SUBSTRATE_V2 = "merlin.common_compiler_substrate.v2"
COMMON_SUBSTRATE_V3 = "merlin.common_compiler_substrate.v3"
COMMON_SUBSTRATE_V4 = "merlin.common_compiler_substrate.v4"
COMMON_SUBSTRATE_V5 = "merlin.common_compiler_substrate.v5"
COMMON_SUBSTRATE_V6 = "merlin.common_compiler_substrate.v6"
_ISA_SUBSTRATES = frozenset({COMMON_SUBSTRATE_V3, COMMON_SUBSTRATE_V4, COMMON_SUBSTRATE_V5, COMMON_SUBSTRATE_V6})
_PY = "merlin/python/merlin/"
COMMON_PUBLIC_MODULES = (
    "merlin.transforms.certificates",
    "merlin.transforms.normalization",
    "merlin.transforms.loops",
    "merlin.transforms.planning",
    "merlin.transforms.validation",
    "merlin.transforms.pipeline",
    "merlin.transforms.client",
    "merlin.verify.lean_contract",
)
_COMMON_IMPLEMENTATION_MODULES = ("merlin.transforms.pipeline_state",)
_DENSE_FLOAT_IMPLEMENTATION_MODULES = ("merlin.common.__init__", "merlin.common.dense_float_ir")
COMMON_NAMESPACE_MODULES = (
    "merlin.__init__",
    "merlin.transforms.__init__",
    "merlin.verify.__init__",
)
AUTOMATIC_CHECKER_MODULES = (
    "merlin.verify.lean_export",
    "merlin.verify.lean_checker",
)
AUTOMATIC_CHECKER_DATA = ("verify/lean/Merlin/Foundation.lean",)
COMMON_PUBLIC_MODULES_V2 = (
    *COMMON_PUBLIC_MODULES,
    "merlin.transforms.pure_scalar",
    "merlin.transforms.scalar_witness",
    "merlin.transforms.unrolling",
    "merlin.transforms.analysis_types",
    "merlin.transforms.analysis_memory",
    "merlin.transforms.analysis_liveness",
    "merlin.transforms.analyses",
    "merlin.transforms.resource_allocation",
    "merlin.semantic_compiler.allocation_types",
    "merlin.semantic_compiler.allocate",
    "merlin.semantic_compiler.rules",
    "merlin.semantic_compiler.model",
)
COMMON_NAMESPACE_MODULES_V2 = (*COMMON_NAMESPACE_MODULES, "merlin.semantic_compiler.__init__")
AUTOMATIC_CHECKER_MODULES_V2 = (
    *AUTOMATIC_CHECKER_MODULES,
    "merlin.verify.lean_obligations",
    "merlin.perf.pass_observation",
)
COMMON_PUBLIC_MODULES_V4 = (
    *COMMON_PUBLIC_MODULES_V2,
    "merlin.transforms.dense_planning",
    "merlin.transforms.dense_ir",
)
COMMON_PUBLIC_MODULES_V5 = (*COMMON_PUBLIC_MODULES_V4, "merlin.compile.scheduling.block_schedule")
COMMON_NAMESPACE_MODULES_V3 = (*COMMON_NAMESPACE_MODULES_V2, "merlin.targetgen.__init__")
AUTOMATIC_CHECKER_MODULES_V3 = (
    *AUTOMATIC_CHECKER_MODULES_V2,
    "merlin.verify.lean_resource_geometry",
    "merlin.verify.lean_state_effects",
    "merlin.verify.lean_state_projection",
    "merlin.verify.lean_checked",
    "merlin.verify.lean_emission_chain",
    "merlin.verify.lean_emission_context",
    "merlin.verify.lean_emission_products",
    "merlin.verify.lean_sat_runtime",
    "merlin.verify.lean_isa_semantics",
    "merlin.targetgen.isa_spec",
    "merlin.targetgen.isa_dialect_gen",
    "merlin.targetgen.compiler_profile_admission",
    "merlin.targetgen.tool_profile_toolchain",
    "merlin.targetgen.host_optimization_admission",
    "merlin.targetgen.common_transforms",
    "merlin.common.bundle_snapshot",
    "merlin.common.read_lease_digest",
    "merlin.targetgen.rtl.result_format_ceiling",
    "merlin.targetgen.rtl.result_format_validity",
    "merlin.common.bwrap_args",
    "merlin.llvmlower.host_optimization_request",
    "merlin.llvmlower.host_optimization_compile",
    "merlin.targetgen.rocc.semantics",
    "merlin.targetgen.rocc.normalized_operands",
    "merlin.targetgen.rocc.selected_operand_protocol",
    "merlin.targetgen.rtl.firrtl_operand_slices",
    "merlin.targetgen.rtl.firrtl_load_states",
)
AUTOMATIC_CHECKER_DATA_V3 = (
    *AUTOMATIC_CHECKER_DATA,
    "verify/lean/Merlin/ResourceGeometry.lean",
    "verify/lean/Merlin/StateEffects.lean",
)
_OPTIONAL_TOOLS = frozenset({"isa_dialect_generation", "lean_proof_authoring"})


def substrate_inventory(version: str) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Published source membership is interpreted by its explicit substrate version."""
    if version == COMMON_SUBSTRATE_V1:
        return COMMON_NAMESPACE_MODULES, COMMON_PUBLIC_MODULES, AUTOMATIC_CHECKER_MODULES
    if version == COMMON_SUBSTRATE_V2:
        return COMMON_NAMESPACE_MODULES_V2, COMMON_PUBLIC_MODULES_V2, AUTOMATIC_CHECKER_MODULES_V2
    if version == COMMON_SUBSTRATE_V3:
        return COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V2, AUTOMATIC_CHECKER_MODULES_V3
    if version == COMMON_SUBSTRATE_V4:
        return COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V4, AUTOMATIC_CHECKER_MODULES_V3
    if version == COMMON_SUBSTRATE_V5:
        return COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V5, AUTOMATIC_CHECKER_MODULES_V3
    if version == COMMON_SUBSTRATE_V6:
        from .tool_profile_shared import shared_inventory

        namespaces, public, _ = shared_inventory(COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V5)
        automatic = (*AUTOMATIC_CHECKER_MODULES_V3, "merlin.targetgen.tool_profile_shared",
                     "merlin.compile.scheduling.export")
        return namespaces, public, automatic
    raise ValueError("unknown common compiler substrate")


def substrate_data(version: str) -> tuple[str, ...]:
    substrate_inventory(version)
    return AUTOMATIC_CHECKER_DATA_V3 if version in _ISA_SUBSTRATES else AUTOMATIC_CHECKER_DATA


def _policy_version(version: str) -> str:
    # V4--V6 extend authoring tools without changing the V3 proof domains.
    return {
        COMMON_SUBSTRATE_V1: "1", COMMON_SUBSTRATE_V2: "2",
        COMMON_SUBSTRATE_V3: "3", COMMON_SUBSTRATE_V4: "3", COMMON_SUBSTRATE_V5: "3", COMMON_SUBSTRATE_V6: "3",
    }[version]


def _digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _logical(module: str) -> str:
    return _PY + module.removeprefix("merlin.").replace(".", "/") + ".py"


def _source(root: Path, module: str) -> Path:
    selected = root / (module.replace(".", "/") + ".py")
    if selected.is_symlink() or not selected.is_file() or selected.resolve() != selected:
        raise ValueError(f"common compiler source is absent, linked or noncanonical: {module}")
    return selected


def _implementation_members(profile: dict) -> tuple[str, ...]:
    """Exact private dependency extension; older frozen profiles omit it."""
    modules = profile.get("common_implementation_modules", [])
    expected = (
        [],
        list(_COMMON_IMPLEMENTATION_MODULES),
        list(_DENSE_FLOAT_IMPLEMENTATION_MODULES),
        [*_COMMON_IMPLEMENTATION_MODULES, *_DENSE_FLOAT_IMPLEMENTATION_MODULES],
    )
    if profile.get("common_substrate") == COMMON_SUBSTRATE_V6:
        from .tool_profile_shared import profile_inventory

        expected = (list(profile_inventory(profile, COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V5)[2]),)
    if type(modules) is not list or modules not in expected:
        raise ValueError("common compiler implementation dependency changed")
    return tuple(modules)


def _selected_implementation_members(root: Path) -> tuple[str, ...]:
    # Legacy source releases define their state in pipeline.py itself. New
    # releases reexport it; transport that same code without adding a public
    # import permission or reinterpreting an older frozen source inventory.
    implementation = ()
    tree = ast.parse(_source(root, "merlin.transforms.pipeline").read_bytes())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            (node.level == 1 and node.module == "pipeline_state")
            or (node.level == 0 and node.module == "merlin.transforms.pipeline_state")
            or (node.module == "merlin.transforms" and any(a.name == "pipeline_state" for a in node.names))
        ):
            implementation = _COMMON_IMPLEMENTATION_MODULES
            break
    # Bit-preserving certificate IO is another exact private source dependency.
    # Include its namespace only for releases whose selected certificate source
    # imports the helper; historical profiles keep their existing inventories.
    tree = ast.parse(_source(root, "merlin.transforms.certificates").read_bytes())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Import)
            and any(a.name == "merlin.common.dense_float_ir" for a in node.names)
            or isinstance(node, ast.ImportFrom)
            and (
                (node.level == 0 and node.module == "merlin.common.dense_float_ir")
                or (node.level == 2 and node.module == "common.dense_float_ir")
                or (
                    (node.level == 0 and node.module == "merlin.common" or node.level == 2 and node.module == "common")
                    and any(a.name == "dense_float_ir" for a in node.names)
                )
            )
        ):
            return (*implementation, *_DENSE_FLOAT_IMPLEMENTATION_MODULES)
    return implementation


def common_tool_profile(
    arm: str, tools: tuple[str, ...], *, python_source_root: Path, version: str = COMMON_SUBSTRATE_V1
) -> dict:
    """Bind actual common source bytes and host-owned formal coverage declarations."""
    from merlin.verify import lean_contract

    root = Path(python_source_root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("common substrate requires the explicit canonical selected Python source root")
    namespaces, public, automatic = substrate_inventory(version)
    implementation = _selected_implementation_members(root)
    release = None
    if version == COMMON_SUBSTRATE_V6:
        from .tool_profile_shared import profile_inventory, reviewed_shared_manifest

        release = reviewed_shared_manifest(root)
        namespaces, public, implementation = profile_inventory(
            {"shared_scheduling_library": release}, COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V5
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
        for module in automatic
    ]
    for relative in substrate_data(version):
        source = root / "merlin" / relative
        if source.is_symlink() or not source.is_file() or source.resolve() != source:
            raise ValueError("automatic verifier data is absent, linked or noncanonical")
        automatic_sources.append(
            {"module": None, "path": _PY + relative, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    if (
        hashlib.sha256(Path(lean_contract.__file__).read_bytes()).hexdigest()
        != hashlib.sha256(_source(root, "merlin.verify.lean_contract").read_bytes()).hexdigest()
    ):
        raise ValueError("loaded formal coverage policy differs from the selected source release")
    policy = lean_contract.coverage_policy_record(version=_policy_version(version))
    if policy.get("mode") != "strict_within_coverage":
        raise ValueError("new compiler substrate requires the host-owned strict formal coverage policy")
    record = {
        "schema": SCHEMA,
        "common_substrate": version,
        "arm": arm,
        "treatment_tools": list(tools),
        "common_public_modules": list(public),
        "automatic_checker_modules": list(automatic),
        "common_sources": sources,
        "automatic_checker_sources": automatic_sources,
        "proof_policy": copy.deepcopy(policy),
    }
    if implementation:
        record["common_implementation_modules"] = list(implementation)
    if release is not None:
        record["shared_scheduling_library"] = release
        record["shared_scheduling_resources"] = [
            {"path": str(root / name), "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in ("shared-library.json", "LICENSE", "NOTICE")
        ]
    return {**record, "sha256": _digest(record)}


def verify_tool_profile(bundle: dict, *, tools: tuple[str, ...] | None = None) -> dict | None:
    """Validate immutable inventory; legacy bundles retain their exact policy."""
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
    namespaces, public, automatic = substrate_inventory(profile.get("common_substrate"))
    if profile.get("common_substrate") == COMMON_SUBSTRATE_V6:
        from .tool_profile_shared import profile_inventory

        namespaces, public, _ = profile_inventory(profile, COMMON_NAMESPACE_MODULES_V3, COMMON_PUBLIC_MODULES_V5)
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
    if profile.get("automatic_checker_modules") != list(automatic):
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
    if profile["common_substrate"] == COMMON_SUBSTRATE_V6:
        from .tool_profile_shared import validate_profile_release

        validate_profile_release(profile)
    elif "shared_scheduling_library" in profile or "shared_scheduling_resources" in profile:
        raise ValueError("shared compiler release is outside the original frozen substrate")
    automatic_sources = profile.get("automatic_checker_sources")
    expected_automatic = [(module, _logical(module)) for module in automatic] + [
        (None, _PY + relative) for relative in substrate_data(profile["common_substrate"])
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
    from merlin.verify.lean_contract import coverage_policy_record

    expected_policy = coverage_policy_record(version=_policy_version(profile["common_substrate"]))
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
    from .public_codegen_budget import policy_from_profile

    policy_from_profile(profile)
    if profile["common_substrate"] in _ISA_SUBSTRATES:
        from merlin.verify.lean_isa_semantics import validate_semantic_inputs

        validate_semantic_inputs(profile.get("semantic_inputs"))
    return copy.deepcopy(profile)


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
    namespaces, public, automatic = substrate_inventory(version)
    if host_index_bits is not None and (type(host_index_bits) is not int or not 2 <= host_index_bits <= 128):
        raise ValueError("reviewed host index width must be 2..128 bits or explicitly unknown")
    tools = tuple(bundle["tools"])
    profile = common_tool_profile(bundle["arm"], tools, python_source_root=python_source_root, version=version)
    # Deny-by-default isolation admits only these leaves. Parent denials would
    # also hide common leaves, so replace those historical blanket exclusions.
    bundle["denied"] = [row for row in bundle["denied"] if row.get("path") not in {"merlin/", _PY + "verify/"}]
    # The historical verification arm granted a directory before Lean modules
    # existed. Fresh bundles retain those historical tools as exact leaves so
    # that the new proof-authoring capability cannot leak through that grant.
    historical_verify = [row for row in bundle["allowed"] if row.get("path") == _PY + "verify/"]
    for row in historical_verify:
        bundle["allowed"].remove(row)
        for source in sorted((python_source_root / "merlin/verify").glob("*.py")):
            if not source.name.startswith("lean_") and "merlin.verify." + source.stem not in automatic:
                bundle["allowed"].append({**row, "path": _PY + "verify/" + source.name})
    for module in (*namespaces, *public, *_implementation_members(profile)):
        if version == COMMON_SUBSTRATE_V6 and any(
            row.get("path") in {_logical(module), str(_source(python_source_root, module))}
            for row in bundle["allowed"]
        ):
            continue
        bundle["allowed"].append({"path": _logical(module), "mode": "ro", "note": "common compiler infrastructure"})
    if version == COMMON_SUBSTRATE_V6:
        bundle["allowed"].extend(
            {"path": row["path"], "mode": "ro", "note": "shared compiler release identity and license"}
            for row in profile["shared_scheduling_resources"]
        )
    # Automatic verification is host-owned. These dependencies are sealed as
    # host inputs, never exposed by adding a general proof API to lower levels.
    if "lean_proof_authoring" not in tools:
        for module in automatic:
            if version == COMMON_SUBSTRATE_V6 and module in (*namespaces, *public, *_implementation_members(profile)):
                continue
            if version in _ISA_SUBSTRATES and any(
                row.get("path") == str(_source(python_source_root, module)) for row in bundle["allowed"]
            ):
                continue
            bundle["host_inputs"].append(
                {"path": str(_source(python_source_root, module)), "note": "host-owned automatic formal verification"}
            )
        bundle["host_inputs"].extend(
            {"path": str(python_source_root / "merlin" / relative), "note": "automatic formal verification lemmas"}
            for relative in substrate_data(version)
        )
    else:
        author_sources = [(_logical(module), _source(python_source_root, module)) for module in automatic]
        author_sources.extend(
            (_PY + relative, python_source_root / "merlin" / relative) for relative in substrate_data(version)
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
        path = (
            str(software_spec_path)
            if version in _ISA_SUBSTRATES and software_spec_path is not None
            else row["path"]
        )
        bundle["allowed"].append({"path": path, "mode": "ro", "note": "admitted software semantics"})
    for tool_name in sorted(_OPTIONAL_TOOLS):
        tool = TR.spec(tool_name)
        if tool_name not in tools:
            bundle["denied"].extend(
                {"path": path, "reason": tool.deny_reason}
                for path in tool.bundle_paths
                if path != _logical("merlin.verify.lean_contract")
                and not (version == COMMON_SUBSTRATE_V6 and path in {_logical(module) for module in public})
            )
    from .tool_profile_toolchain import attach_selected_toolchain

    toolchain = attach_selected_toolchain(bundle, tools, lean_toolchain_manifest)
    if version in _ISA_SUBSTRATES and lean_toolchain_manifest is not None:
        from .tool_profile_toolchain import require_packaged_mathematics

        require_packaged_mathematics(
            lean_toolchain_manifest, python_source_root, ("Foundation", "ResourceGeometry", "StateEffects")
        )
    profile = {key: value for key, value in profile.items() if key != "sha256"}
    profile["lean_toolchain"] = toolchain
    profile["host_index_bits"] = host_index_bits
    if public_codegen_resource_policy is not None:
        from .public_codegen_budget import FIELD, validate_policy

        profile[FIELD] = validate_policy(public_codegen_resource_policy)
    if version in _ISA_SUBSTRATES:
        from merlin.verify.lean_isa_semantics import select_isa_semantics

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
    elif software_spec_path is not None or facts_path is not None or target is not None:
        raise ValueError("original formal semantic selection requires substrate V3")
    profile["sha256"] = _digest(profile)
    bundle["compiler_tool_profile"] = profile
    verify_tool_profile(bundle)


def inherited_tool_profile(phase1_bundle: dict, phase2_bundle: dict) -> dict | None:
    """Phase 2 must reuse the exact admitted Phase 1 grants and proof policy."""
    before = verify_tool_profile(phase1_bundle)
    after = verify_tool_profile(phase2_bundle)
    if before != after:
        raise ValueError("Phase 2 changed the frozen Phase 1 compiler tool profile")
    for key in ("allowed", "denied", "host_inputs", "selected_rtl_facts_file"):
        if phase1_bundle.get(key) != phase2_bundle.get(key):
            raise ValueError("Phase 2 changed the frozen Phase 1 target/spec/runtime grants")
    return before


def tool_profile_prompt(bundle: dict) -> str:
    profile = verify_tool_profile(bundle)
    if profile is None:
        return ""
    lines = [
        "\n\n## Common compiler infrastructure and preservation policy\n",
        "The versioned common substrate is available in every experiment level: "
        + ", ".join(f"`{module}`" for module in profile["common_public_modules"])
        + ".",
        "Use its documented generic transformations and analyses. Their automatic verification "
        "is common infrastructure, not an additional level treatment.",
        "Use `merlin.transforms.client.apply_transform(workspace, source_ir, options)` through the "
        "original host broker for verified common transformations. Submit only IR and documented "
        "parameters; the host selects admitted semantics, formal coverage and the frozen proof toolchain.",
        "Within the admitted formal domain, successful Lean checking is required. Missing tools, "
        "unknown obligations, failed proofs and timeouts cannot discharge covered obligations. "
        "Outside that domain, independent correctness checks remain required and formal coverage "
        "must be reported explicitly. The admitted semantics and coverage policy are immutable.",
        "Tool presence describes an interface, not a successful runtime qualification. "
        "Report actual availability and proof outcomes separately from numerical and performance results.",
    ]
    if profile["common_substrate"] in _ISA_SUBSTRATES:
        lines.append(
            "Use `merlin.transforms.client.check_transform(workspace, before_ir, after_ir, entry=\"...\")` "
            "to submit actual compiler rewrites for covered host checking. The original host selects "
            "software/ISA semantics and effect premises from sealed inputs. A result establishes only "
            "its reported formal scope; hardware, numerical execution and performance need independent evidence."
        )
    if profile["common_substrate"] in _ISA_SUBSTRATES:
        lines.append(
            "The original command-buffer schema accepts an optional source-bound "
            "`host_optimization_request`. Only bounded documented choices are permitted; "
            "the original reviewed software profile independently supplies any required host "
            "semantic and runtime permissions. Omission of those permissions leaves the request "
            "unavailable. Numerical, instruction-policy and complete execution gates still apply."
        )
    if profile["common_substrate"] in _ISA_SUBSTRATES:
        lines.append(
            "During original compiler entrypoint calls, `MERLIN_COMMON_WORKSPACE` names "
            "the isolated request workspace for the same public transformation/check client. "
            "Replies establish only their reported scope."
        )
    if profile["common_substrate"] in {COMMON_SUBSTRATE_V4, COMMON_SUBSTRATE_V5, COMMON_SUBSTRATE_V6}:
        lines.append(
            "Shared dense scheduling is available as `merlin.transforms.dense_planning`: "
            "provide explicit primitive geometry, available storage after reservations, "
            "supported block limits and a target static-command price. The planner returns "
            "a bounded choice, not a numerical or execution proof. Emit and verify its "
            "`merlin.transforms.dense_ir.DenseScheduleOp` against the realized commands. "
            "Version 4 retains version 3's formal coverage policy."
        )
    if profile["common_substrate"] in {COMMON_SUBSTRATE_V5, COMMON_SUBSTRATE_V6}:
        lines.append(
            "Shared `merlin.compile.scheduling.block_schedule` provides checked synchronous "
            "matmul/convolution schedules. Supply explicit fact-derived Geometry; "
            "check_residency verifies operand contents, ordered reductions, accumulator "
            "lifetimes and complete output stores. It does not establish asynchronous "
            "completion or external view aliasing. Project emitted operations explicitly; "
            "load counts alone cannot establish lifetime conformance. Version 5 keeps "
            "version 4's automatic checkers and formal coverage policy."
        )
    if profile["common_substrate"] == COMMON_SUBSTRATE_V6:
        lines.append(
            "Every V6 treatment receives the same shared scheduling SDK: tiling and reuse, "
            "temporary storage, supported fusion/copy elimination, persistent configuration, "
            "device spills, complete row reductions and final program checks. Phase 2 may "
            "edit reusable shape/resource-based policy code to select legal alternatives. "
            "Keep numerical policy, hardware limits, qualification identities and verification "
            "gates fixed; checker work limits are not performance objectives. Check actual "
            "decoded output and complete outputs/state before performance comparisons. "
            "The exact original Phase 1 release is immutable; older runs cannot acquire "
            "these grants. V6 also makes generic ISA parser/generator algorithms common "
            "infrastructure, so historical ISA-tool ablations need a new interpretation. "
            "No target facts, instruction meanings, execution qualification or runtime tools "
            "follow from SDK availability. Missing capabilities remain unavailable."
        )
    for name in profile["treatment_tools"]:
        if name in _OPTIONAL_TOOLS:
            lines.append(f"- `{name}`: {TR.spec(name).blurb}")
    return "\n".join(lines) + "\n"
