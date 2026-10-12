"""Versioned common compiler facilities and immutable experiment tool policy.

Historical manifests have no profile and retain their original interpretation.
Only newly prepared releases select this substrate. Grants name exact sources;
their frozen run snapshot supplies the bytes, not a mutable installed package.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from . import tool_registry as TR

SCHEMA = "merlin.compiler_tool_profile.v1"
COMMON_SUBSTRATE_V1 = "merlin.common_compiler_substrate.v1"
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
_OPTIONAL_TOOLS = frozenset({"isa_dialect_generation", "lean_proof_authoring"})


def _digest(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _logical(module: str) -> str:
    return _PY + module.removeprefix("merlin.").replace(".", "/") + ".py"


def _source(root: Path, module: str) -> Path:
    selected = root / (module.replace(".", "/") + ".py")
    if selected.is_symlink() or not selected.is_file() or selected.resolve() != selected:
        raise ValueError(f"common compiler source is absent, linked or noncanonical: {module}")
    return selected


def common_tool_profile(arm: str, tools: tuple[str, ...], *, python_source_root: Path) -> dict:
    """Bind actual common source bytes and host-owned formal coverage declarations."""
    from merlin.verify import lean_contract

    root = Path(python_source_root)
    if not root.is_absolute() or root.resolve() != root or not root.is_dir():
        raise ValueError("common substrate requires the explicit canonical selected Python source root")
    modules = (*COMMON_NAMESPACE_MODULES, *COMMON_PUBLIC_MODULES)
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
        != hashlib.sha256(_source(root, "merlin.verify.lean_contract").read_bytes()).hexdigest()
    ):
        raise ValueError("loaded formal coverage policy differs from the selected source release")
    policy = lean_contract.coverage_policy_record()
    if policy.get("mode") != "strict_within_coverage":
        raise ValueError("new compiler substrate requires the host-owned strict formal coverage policy")
    record = {
        "schema": SCHEMA,
        "common_substrate": COMMON_SUBSTRATE_V1,
        "arm": arm,
        "treatment_tools": list(tools),
        "common_public_modules": list(COMMON_PUBLIC_MODULES),
        "automatic_checker_modules": list(AUTOMATIC_CHECKER_MODULES),
        "common_sources": sources,
        "automatic_checker_sources": automatic_sources,
        "proof_policy": copy.deepcopy(policy),
    }
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
    if profile.get("common_substrate") != COMMON_SUBSTRATE_V1 or profile.get("arm") != bundle.get("arm"):
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
    if profile.get("common_public_modules") != list(COMMON_PUBLIC_MODULES):
        raise ValueError("common compiler inventory is incomplete or changed")
    if profile.get("automatic_checker_modules") != list(AUTOMATIC_CHECKER_MODULES):
        raise ValueError("automatic verification inventory changed")
    sources = profile.get("common_sources")
    expected_modules = (*COMMON_NAMESPACE_MODULES, *COMMON_PUBLIC_MODULES)
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
    if (
        policy.get("mode") != "strict_within_coverage"
        or policy.get("unknown_within_coverage") != "reject"
        or policy.get("outside_coverage") != "independent_checks_required"
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
    return copy.deepcopy(profile)


def attach_common_substrate(
    bundle: dict,
    *,
    version: str,
    python_source_root: Path,
    lean_toolchain_manifest: Path | None = None,
    host_index_bits: int | None = None,
) -> None:
    """Add reviewed common facilities while masking optional tools on every other level."""
    if version != COMMON_SUBSTRATE_V1:
        raise ValueError(f"unknown common compiler substrate {version!r}")
    if host_index_bits is not None and (type(host_index_bits) is not int or not 2 <= host_index_bits <= 128):
        raise ValueError("reviewed host index width must be 2..128 bits or explicitly unknown")
    tools = tuple(bundle["tools"])
    profile = common_tool_profile(bundle["arm"], tools, python_source_root=python_source_root)
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
            if not source.name.startswith("lean_"):
                bundle["allowed"].append({**row, "path": _PY + "verify/" + source.name})
    for module in (*COMMON_NAMESPACE_MODULES, *COMMON_PUBLIC_MODULES):
        bundle["allowed"].append({"path": _logical(module), "mode": "ro", "note": "common compiler infrastructure"})
    # Automatic verification is host-owned. These dependencies are sealed as
    # host inputs, never exposed by adding a general proof API to lower levels.
    if "lean_proof_authoring" not in tools:
        for module in AUTOMATIC_CHECKER_MODULES:
            bundle["host_inputs"].append({"path": _logical(module), "note": "host-owned automatic formal verification"})
        bundle["host_inputs"].extend(
            {"path": _PY + relative, "note": "automatic formal verification lemmas"}
            for relative in AUTOMATIC_CHECKER_DATA
        )
    software_inputs = [
        row for row in bundle["host_inputs"] if row.get("note") == "selected software semantics; not hardware evidence"
    ]
    for row in software_inputs:
        bundle["host_inputs"].remove(row)
        bundle["allowed"].append({"path": row["path"], "mode": "ro", "note": "admitted software semantics"})
    for tool_name in sorted(_OPTIONAL_TOOLS):
        tool = TR.spec(tool_name)
        if tool_name not in tools:
            bundle["denied"].extend(
                {"path": path, "reason": tool.deny_reason}
                for path in tool.bundle_paths
                if path != _logical("merlin.verify.lean_contract")
            )
    toolchain = {"selection": "not_selected", "readiness": "unavailable"}
    if lean_toolchain_manifest is not None:
        selected = Path(lean_toolchain_manifest)
        if (
            not selected.is_absolute()
            or selected.is_symlink()
            or selected.resolve() != selected
            or not selected.is_file()
        ):
            raise ValueError("Lean toolchain manifest must be an ordinary canonical selected file")
        document = json.loads(selected.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema") != "merlin.lean_toolchain.v1":
            raise ValueError("unknown selected Lean toolchain manifest schema")
        executable = document.get("executable")
        files = document.get("files")
        if not isinstance(executable, dict) or not isinstance(files, list) or not files:
            raise ValueError("Lean toolchain requires its explicit executable and closed file membership")
        roots = document.get("library_roots")
        if not isinstance(roots, list) or not roots or any(not isinstance(root, str) or not root for root in roots):
            raise ValueError("Lean toolchain requires its explicit import roots")
        for name in ("timeout_s", "memory_mb"):
            budget = document.get(name, 120 if name == "timeout_s" else 1024)
            if type(budget) is not int or budget < 1:
                raise ValueError("Lean toolchain requires positive execution budgets")

        def resolve_member(raw: str) -> Path:
            path = Path(raw)
            if ".." in path.parts or not raw:
                raise ValueError("Lean toolchain member path is invalid")
            path = path if path.is_absolute() else selected.parent / path
            if path.is_symlink() or path.resolve() != path:
                raise ValueError("Lean toolchain member is linked or noncanonical")
            return path

        rows = [executable, *files]
        paths: list[str] = []
        import_roots = [resolve_member(root) for root in roots]
        missing = any(not root.is_dir() for root in import_roots)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Lean toolchain contains an invalid file member")
            raw, digest = row.get("path"), row.get("sha256")
            if (
                not isinstance(raw, str)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("Lean toolchain member requires an explicit path and SHA-256")
            path = resolve_member(raw)
            if path.exists():
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("selected Lean toolchain bytes differ from its manifest")
            else:
                missing = True
            if str(path) not in paths:
                paths.append(str(path))
        role = "allowed" if "lean_proof_authoring" in tools else "host_inputs"
        for path in (str(selected), *paths):
            row = {"path": path, "note": "selected formal verification toolchain"}
            if path == str(selected):
                row["note"] = "selected formal verification toolchain manifest"
            if role == "allowed":
                row["mode"] = "ro"
            bundle[role].append(row)
        toolchain = {
            "selection": "explicit_manifest",
            "manifest_sha256": hashlib.sha256(selected.read_bytes()).hexdigest(),
            "executable_sha256": executable["sha256"],
            "dependency_revision": document.get("dependency_revision"),
            "readiness": "unavailable" if missing else "declared_unprobed",
        }
    profile = {key: value for key, value in profile.items() if key != "sha256"}
    profile["lean_toolchain"] = toolchain
    profile["host_index_bits"] = host_index_bits
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
    for name in profile["treatment_tools"]:
        if name in _OPTIONAL_TOOLS:
            lines.append(f"- `{name}`: {TR.spec(name).blurb}")
    return "\n".join(lines) + "\n"
