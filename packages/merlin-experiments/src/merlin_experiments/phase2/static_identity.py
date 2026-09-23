"""Compiler and portable byte identities, and admission of exact static cache reuse.

Native callers supply source roots and authority bindings; this owner neither
discovers a checkout nor imports evaluated compiler modules. Sealed iteration JSON
is decoded from the same bytes whose original journal digest was checked.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from merlin.benchharness import hash_tree

from . import authoring as AUTHORING
from . import contracts as P2_CONTRACTS
from .broker_evidence import _is_sha256
from .contracts import sha256_file as _sha256_file


def compiler_dependency_content_sha256(record: Mapping[str, Any]) -> str:
    """Portable identity for compiler-affecting dependency bytes, excluding snapshot paths."""
    required = ("candidate_sha256", "shared_sources", "selected_lazy_exports")
    if any(name not in record for name in required) or not _is_sha256(record["candidate_sha256"]):
        raise ValueError("compiler dependency record is incomplete")
    return P2_CONTRACTS.document_sha256({name: record[name] for name in required})


def _verified_relative_source_hashes(record: Mapping[str, Any], *, source_root: Path) -> dict[str, str]:
    """Rebind one host policy to source-relative bytes after checking every named file.

    Absolute paths are retained in receipts for auditability, but are not semantic identity.  A
    prior policy is portable only when the current verifier can prove that each pinned absolute
    path was a real file below its sealed source snapshot and still has the recorded bytes.
    """
    sources = record.get("sources")
    root = Path(source_root).resolve()
    if record.get("schema") != "global_host_verification_policy_v1" or not isinstance(sources, Mapping):
        raise ValueError("host verification policy is malformed")
    relative: dict[str, str] = {}
    for raw_path, digest in sources.items():
        if not isinstance(raw_path, str) or not _is_sha256(digest):
            raise ValueError("host verification policy source identity is malformed")
        path = Path(raw_path)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError("host verification policy source is absent, linked, or non-absolute")
        try:
            name = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("host verification policy source escaped its sealed source root") from exc
        if name in relative or _sha256_file(path) != digest:
            raise ValueError("host verification policy source bytes changed")
        relative[name] = digest
    if not relative:
        raise ValueError("host verification policy has no source identities")
    return dict(sorted(relative.items()))


def host_policy_content_sha256(record: Mapping[str, Any], *, source_root: Path) -> str:
    """Path-neutral, byte-exact identity used only for cross-run static-analysis reuse."""
    if record.get("schema") in ("global_host_verification_policy_v2", "global_host_verification_policy_v3"):
        from .host_policy import content_sha256

        return content_sha256(record, source_root=source_root)
    if record.get("schema") == "global_host_verification_policy_v1":
        return P2_CONTRACTS.document_sha256(_verified_relative_source_hashes(record, source_root=source_root))
    raise ValueError("host verification policy is malformed")


def portable_machine_build_policy(record: Mapping[str, Any] | None, *, verify_files: bool) -> Mapping[str, Any] | None:
    """Remove executable spellings while retaining and optionally verifying their bytes."""
    if record is None:
        return None

    def normalize(value: Any) -> Any:
        if isinstance(value, Mapping):
            if "path" in value or "resolved_path" in value:
                digest = value.get("sha256")
                raw_path = value.get("path")
                if not _is_sha256(digest) or not isinstance(raw_path, str):
                    raise ValueError("machine build policy path lacks an exact content identity")
                path = Path(raw_path)
                if verify_files and (path.is_symlink() or not path.is_file() or _sha256_file(path) != digest):
                    raise ValueError("machine build policy executable or implementation changed")
                return {key: normalize(item) for key, item in value.items() if key not in ("path", "resolved_path")}
            return {key: normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return copy.deepcopy(value)

    return normalize(record)


def portable_phase1_binding(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if record is None:
        return None
    result = copy.deepcopy(dict(record))
    result.pop("run_dir", None)
    return result


def portable_optimization_baseline_binding(record: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(record))
    dependencies = result.pop("compiler_dependencies", None)
    result.pop("path", None)
    result["compiler_dependencies_content_sha256"] = compiler_dependency_content_sha256(dependencies)
    return result


def portable_edit_authority(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if record is None:
        return None
    result = copy.deepcopy(dict(record))
    result.pop("seed_path", None)
    return result


def portable_historical_reference(record: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if record is None:
        return None
    result = copy.deepcopy(dict(record))
    result.pop("path", None)
    return result


DYNAMIC_EVIDENCE_KEYS = frozenset(
    {
        "probe_receipts",
        "semantic_receipts",
        "context_receipts",
        "paired_context_receipts",
        "source_contraction_preparation_receipts",
        "source_pair_receipts",
        "decision_feedback",
        "relative_semantic_evidence",
        "global_performance_claim",
        "global_speedup_proven",
        "full_model_cycles",
        "elapsed_seconds",
        "wall_seconds",
        "build_wall_seconds",
        "run_wall_seconds",
        "emission_wall_seconds",
        "observed_analysis_wall_seconds",
        "functional_gate",
    }
)


def static_only_copy(value: Any) -> Any:
    """Copy analytical evidence while excluding measurements and decision/semantic feedback."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if key in DYNAMIC_EVIDENCE_KEYS or key == "emission_execution":
                continue
            result[key] = static_only_copy(item)
        return result
    if isinstance(value, list):
        return [static_only_copy(item) for item in value]
    if isinstance(value, tuple):
        return [static_only_copy(item) for item in value]
    return copy.deepcopy(value)


def compiler_dependency_record(candidate: Path, *, shared_source_root: Path) -> dict[str, Any]:
    """Hash candidate code plus its statically resolved trusted Merlin import closure.

    This does not import candidate modules to discover dependencies. Shared helpers remain shared,
    but an edit to one invalidates the compiler identity just as an edit inside the package does.
    External Python/toolchain installations belong to the launch environment identity separately.
    """
    from merlin.perf.static_imports import imported_attribute_paths, resolve_lazy_export

    root = shared_source_root.resolve()
    if not root.is_dir() or not (root / "__init__.py").is_file():
        raise ValueError("compiler shared-source root must identify an existing Merlin package")
    pending: list[tuple[Path, str]] = [
        (path, "") for path in candidate.rglob("*.py") if "__pycache__" not in path.parts
    ]
    scanned: set[Path] = set()
    shared: dict[str, str] = {}
    requested: set[str] = set()
    lazy_exports: dict[str, str] = {}

    def enqueue(module: str) -> None:
        if module != "merlin" and not module.startswith("merlin."):
            return
        if module in requested:
            return
        requested.add(module)
        parts = module.split(".")[1:]
        choices = (
            (root.joinpath(*parts).with_suffix(".py"), root.joinpath(*parts) / "__init__.py")
            if parts
            else (root / "__init__.py",)
        )
        for source in choices:
            if source.is_file():
                if source.is_symlink() or (source.resolve() != root and root not in source.resolve().parents):
                    raise ValueError("shared compiler import escapes the trusted Merlin source root")
                pending.append((source, module))
                for count in range(len(parts)):
                    parent = root.joinpath(*parts[:count]) / "__init__.py"
                    if parent.is_file():
                        pending.append((parent, ".".join(("merlin", *parts[:count]))))
                return
        # A from-import may name an exported class rather than a source file. Resolve just
        # that requested lazy symbol, without importing __init__ or granting sibling modules.
        package, _, symbol = module.rpartition(".")
        initializer = root.joinpath(*package.split(".")[1:]) / "__init__.py"
        if not initializer.is_file():
            return
        if initializer.is_symlink() or root not in initializer.resolve().parents:
            raise ValueError("shared compiler import escapes the trusted Merlin source root")
        resolution = resolve_lazy_export(initializer.read_bytes(), package=package, symbol=symbol)
        if resolution.status == "unresolved":
            raise ValueError(f"unresolved shared lazy import {module}: {resolution.reason}")
        if resolution.status == "resolved":
            assert resolution.module is not None
            lazy_exports[module] = resolution.module
            enqueue(package)
            enqueue(resolution.module)

    while pending:
        source, module = pending.pop()
        source = source.resolve()
        if source in scanned:
            continue
        scanned.add(source)
        payload = source.read_bytes()
        if root in source.parents:
            shared[source.relative_to(root).as_posix()] = hashlib.sha256(payload).hexdigest()
        tree = ast.parse(payload, filename=str(source))
        package = module if source.name == "__init__.py" else module.rpartition(".")[0]
        bindings: dict[str, set[str]] = {}
        import_functions: set[str] = {"__import__"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    enqueue(alias.name)
                    bindings.setdefault(alias.asname or alias.name.split(".")[0], set()).add(
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                if node.level:
                    if not package:
                        continue  # candidate-local relative import; all its .py files are scanned
                    prefix = package.split(".")[: len(package.split(".")) - node.level + 1]
                    name = ".".join((*prefix, name)) if name else ".".join(prefix)
                enqueue(name)
                for alias in node.names:
                    enqueue(name + "." + alias.name)
                    if alias.name != "*":
                        bindings.setdefault(alias.asname or alias.name, set()).add(name + "." + alias.name)
                    if name == "importlib" and alias.name == "import_module":
                        import_functions.add(alias.asname or alias.name)
        for path in sorted(imported_attribute_paths(tree, bindings)):
            enqueue(path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and (
                    (isinstance(node.func, ast.Name) and node.func.id in import_functions)
                    or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
                )
            ):
                enqueue(node.args[0].value)
    body = {
        "candidate_sha256": hash_tree(candidate)["sha256"],
        "shared_source_root": str(root),
        "shared_sources": dict(sorted(shared.items())),
        "selected_lazy_exports": dict(sorted(lazy_exports.items())),
    }
    return {
        "schema": "compiler_implementation_dependencies_v1",
        **body,
        "compiler_implementation_sha256": P2_CONTRACTS.document_sha256(body),
        "scope": "candidate and static Merlin source import closure; toolchain identity is separate",
    }


def immutable_reusable_iteration(
    row: Mapping[str, Any],
    *,
    binding: Mapping[str, Any],
    output: Path,
    record_sha256: Mapping[int, str],
    compiler_shared_source_root: Path,
    capsule_sha256s: Sequence[str],
    portfolio_sha256: str,
) -> dict[str, Any] | None:
    """Load and revalidate one prior ready iteration; malformed cache entries are misses."""
    if not capsule_sha256s:
        return None
    iteration = row.get("iteration")
    if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
        return None
    path = output / f"iteration_{iteration:04d}.json"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            return None
        payload = path.read_bytes()
        if record_sha256.get(iteration) != hashlib.sha256(payload).hexdigest():
            return None
        source = json.loads(payload.decode("utf-8"))
        if not isinstance(source, dict):
            return None
        if (
            source.get("schema") != "global_perf_iteration_v1"
            or source.get("iteration") != iteration
            or source.get("analysis_reuse_binding") != binding
            or source.get("candidate_sha256") != binding["candidate_sha256"]
            or source.get("compiler_dependencies") != binding["compiler_dependencies"]
            or source.get("readiness", {}).get("status") != "ready_for_probe_admission"
            or "iteration_wall_budget_exceeded" in source.get("readiness", {}).get("blockers", ())
            or not isinstance(source.get("elapsed_seconds"), (int, float))
            or isinstance(source.get("elapsed_seconds"), bool)
            or not math.isfinite(source["elapsed_seconds"])
            or not isinstance(source.get("allocated_seconds"), (int, float))
            or isinstance(source.get("allocated_seconds"), bool)
            or not math.isfinite(source["allocated_seconds"])
            or source["elapsed_seconds"] > source["allocated_seconds"]
            or P2_CONTRACTS.document_sha256(source.get("analysis")) != P2_CONTRACTS.document_sha256(row.get("analysis"))
        ):
            return None
        submitted = Path(source["submitted_snapshot"])
        output = output.resolve()
        if (
            submitted.is_symlink()
            or not submitted.is_dir()
            or not submitted.resolve().is_relative_to(output)
            or submitted.stat().st_mode & 0o222
            or any(path.is_symlink() or path.stat().st_mode & 0o222 for path in submitted.rglob("*"))
        ):
            return None
        AUTHORING.assert_candidate_sealable(submitted)
        if (
            hash_tree(submitted)["sha256"] != source["candidate_sha256"]
            or compiler_dependency_record(submitted, shared_source_root=compiler_shared_source_root)
            != source["compiler_dependencies"]
        ):
            return None
        analysis = source.get("analysis") or {}
        portfolio = source.get("portfolio") or {}
        if (
            analysis.get("candidate_sha256") != source["candidate_sha256"]
            or analysis.get("workload", {}).get("capsule_sha256") != capsule_sha256s[0]
            or portfolio.get("candidate_sha256") != source["candidate_sha256"]
            or portfolio.get("portfolio_sha256") != portfolio_sha256
            or portfolio.get("members_total") != len(capsule_sha256s)
        ):
            return None
        expected_members = list(capsule_sha256s)
        members = portfolio.get("members")
        if not isinstance(members, list) or len(members) != len(expected_members):
            return None
        actual_members = [
            member.get("identity", {}).get("capsule_sha256") for member in members if isinstance(member, Mapping)
        ]
        if actual_members != expected_members:
            return None
        for index, (capsule_sha256, member) in enumerate(zip(capsule_sha256s, members, strict=True)):
            member_analysis = analysis if index == 0 else member.get("analysis")
            if (
                not isinstance(member_analysis, Mapping)
                or member_analysis.get("candidate_sha256") != source["candidate_sha256"]
                or member_analysis.get("workload", {}).get("capsule_sha256") != capsule_sha256
                or member.get("readiness", {}).get("status") != "ready_for_probe_admission"
            ):
                return None
        return source
    except (KeyError, OSError, TypeError, ValueError, P2_CONTRACTS.StageGateError):
        return None


def find_reusable_iteration(
    rows: Sequence[Mapping[str, Any]],
    *,
    binding: Mapping[str, Any],
    output: Path,
    record_sha256: Mapping[int, str],
    compiler_shared_source_root: Path,
    capsule_sha256s: Sequence[str],
    portfolio_sha256: str,
) -> dict[str, Any] | None:
    """Return the newest exact valid static record, skipping malformed cache entries."""
    for row in reversed(rows):
        if (
            row.get("candidate_sha256") == binding["candidate_sha256"]
            and row.get("compiler_dependencies") == binding["compiler_dependencies"]
        ):
            source = immutable_reusable_iteration(
                row,
                binding=binding,
                output=output,
                record_sha256=record_sha256,
                compiler_shared_source_root=compiler_shared_source_root,
                capsule_sha256s=capsule_sha256s,
                portfolio_sha256=portfolio_sha256,
            )
            if source is not None:
                return source
    return None
